import torch
import triton
import triton.language as tl
import torch.nn.functional as F
from packaging import version

from typing import Optional

PAD_SLOT_ID = -1

TRITON3 = version.parse(triton.__version__) >= version.parse("3.0.0")

if TRITON3:

    @triton.jit
    def softplus(dt):
        dt = tl.where(dt <= 20.0, tl.math.log(tl.math.exp(dt) + 1), dt)
        return dt

else:

    @triton.jit
    def softplus(dt):
        dt = tl.where(dt <= 20.0, tl.math.log1p(tl.exp(dt)), dt)
        return dt


@triton.heuristics({"HAS_DT_BIAS": lambda args: args["dt_bias_ptr"] is not None})
@triton.heuristics({"HAS_D": lambda args: args["D_ptr"] is not None})
@triton.heuristics({"HAS_Z": lambda args: args["z_ptr"] is not None})
@triton.heuristics(
    {
        "HAS_STATE_BATCH_INDICES": lambda args: args["state_batch_indices_ptr"]
        is not None
    }
)
@triton.heuristics(
    {"BLOCK_SIZE_DSTATE": lambda args: triton.next_power_of_2(args["dstate"])}
)
@triton.jit
def _selective_scan_update_kernel(
    state_ptr,
    x_ptr,
    dt_ptr,
    dt_bias_ptr,
    A_ptr,
    B_ptr,
    C_ptr,
    D_ptr,
    z_ptr,
    out_ptr,
    state_batch_indices_ptr,
    pad_slot_id,
    batch,
    nheads,
    dim,
    dstate,
    nheads_ngroups_ratio,
    stride_state_batch,
    stride_state_head,
    stride_state_dstate,  # innermost after transpose
    stride_state_dim,      # second innermost after transpose
    stride_x_batch,
    stride_x_head,
    stride_x_dim,
    stride_dt_batch,
    stride_dt_head,
    stride_dt_dim,
    stride_dt_bias_head,
    stride_dt_bias_dim,
    stride_A_head,
    stride_A_dim,
    stride_A_dstate,
    stride_B_batch,
    stride_B_group,
    stride_B_dstate,
    stride_C_batch,
    stride_C_group,
    stride_C_dstate,
    stride_D_head,
    stride_D_dim,
    stride_z_batch,
    stride_z_head,
    stride_z_dim,
    stride_out_batch,
    stride_out_head,
    stride_out_dim,
    DT_SOFTPLUS: tl.constexpr,
    TIE_HDIM: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    HAS_DT_BIAS: tl.constexpr,
    HAS_D: tl.constexpr,
    HAS_Z: tl.constexpr,
    HAS_STATE_BATCH_INDICES: tl.constexpr,
    BLOCK_SIZE_DSTATE: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_b = tl.program_id(axis=1)
    pid_h = tl.program_id(axis=2)

    # Batch-head pointer adjustments
    if HAS_STATE_BATCH_INDICES:
        state_batch_indices_ptr += pid_b
        state_batch_idx = tl.load(state_batch_indices_ptr).to(tl.int64)
        state_ptr += state_batch_idx * stride_state_batch + pid_h * stride_state_head
    else:
        state_ptr += pid_b * stride_state_batch + pid_h * stride_state_head

    x_ptr += pid_b * stride_x_batch + pid_h * stride_x_head
    dt_ptr += pid_b * stride_dt_batch + pid_h * stride_dt_head
    if HAS_DT_BIAS:
        dt_bias_ptr += pid_h * stride_dt_bias_head
    A_ptr += pid_h * stride_A_head
    B_ptr += pid_b * stride_B_batch + (pid_h // nheads_ngroups_ratio) * stride_B_group
    C_ptr += pid_b * stride_C_batch + (pid_h // nheads_ngroups_ratio) * stride_C_group
    if HAS_Z:
        z_ptr += pid_b * stride_z_batch + pid_h * stride_z_head
    out_ptr += pid_b * stride_out_batch + pid_h * stride_out_head

    # Persistent kernel loop: each program processes multiple dim blocks
    total_dim_blocks = tl.cdiv(dim, BLOCK_SIZE_M)
    num_programs_dim = tl.num_programs(0)
    offs_n = tl.arange(0, BLOCK_SIZE_DSTATE)

    # Load B and C once (they are constant across dim blocks)
    B_ptrs = B_ptr + offs_n * stride_B_dstate
    B = tl.load(B_ptrs, mask=offs_n < dstate, other=0.0).to(tl.float32)
    C_ptrs = C_ptr + offs_n * stride_C_dstate
    C = tl.load(C_ptrs, mask=offs_n < dstate, other=0.0).to(tl.float32)

    if TIE_HDIM:
        dt = tl.load(dt_ptr).to(tl.float32)
        if HAS_DT_BIAS:
            dt += tl.load(dt_bias_ptr).to(tl.float32)
        if DT_SOFTPLUS:
            dt = softplus(dt)
        A = tl.load(A_ptr).to(tl.float32)
        dA = tl.exp(A * dt)
        dB = B * dt
        for block_idx in range(pid_m, total_dim_blocks, num_programs_dim):
            offs_m = block_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            mask_m = offs_m < dim
            state_mask = (offs_m[:, None] < dim) & (offs_n[None, :] < dstate)
            if HAS_STATE_BATCH_INDICES:
                state_mask = state_mask & (state_batch_idx != pad_slot_id)

            # State pointer: innermost dstate, then dim
            state_ptrs = state_ptr + (
                offs_n[None, :] * stride_state_dstate + offs_m[:, None] * stride_state_dim
            )
            x_ptrs = x_ptr + offs_m * stride_x_dim
            out_ptrs = out_ptr + offs_m * stride_out_dim

            state = tl.load(state_ptrs, mask=state_mask, other=0.0)
            x = tl.load(x_ptrs, mask=mask_m, other=0.0).to(tl.float32)

            state = state * dA + dB * x[:, None]
            tl.store(state_ptrs, state, mask=state_mask)

            out = tl.sum(state * C[None, :], axis=1)
            if HAS_D:
                D_ptrs = D_ptr + offs_m * stride_D_dim
                D = tl.load(D_ptrs, mask=mask_m, other=0.0).to(tl.float32)
                out += x * D
            if HAS_Z:
                z_ptrs = z_ptr + offs_m * stride_z_dim
                z = tl.load(z_ptrs, mask=mask_m, other=0.0).to(tl.float32)
                out *= z * tl.sigmoid(z)
            tl.store(out_ptrs, out, mask=mask_m)
    else:
        for block_idx in range(pid_m, total_dim_blocks, num_programs_dim):
            offs_m = block_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            mask_m = offs_m < dim
            state_mask = (offs_m[:, None] < dim) & (offs_n[None, :] < dstate)
            if HAS_STATE_BATCH_INDICES:
                state_mask = state_mask & (state_batch_idx != pad_slot_id)

            # State pointer: innermost dstate, then dim
            state_ptrs = state_ptr + (
                offs_n[None, :] * stride_state_dstate + offs_m[:, None] * stride_state_dim
            )
            x_ptrs = x_ptr + offs_m * stride_x_dim
            dt_ptrs = dt_ptr + offs_m * stride_dt_dim
            if HAS_DT_BIAS:
                dt_bias_ptrs = dt_bias_ptr + offs_m * stride_dt_bias_dim
            A_ptrs = A_ptr + (
                offs_m[:, None] * stride_A_dim + offs_n[None, :] * stride_A_dstate
            )
            out_ptrs = out_ptr + offs_m * stride_out_dim

            state = tl.load(state_ptrs, mask=state_mask, other=0.0)
            x = tl.load(x_ptrs, mask=mask_m, other=0.0).to(tl.float32)
            dt = tl.load(dt_ptrs, mask=mask_m, other=0.0).to(tl.float32)
            if HAS_DT_BIAS:
                dt += tl.load(dt_bias_ptrs, mask=mask_m, other=0.0).to(tl.float32)
            if DT_SOFTPLUS:
                dt = softplus(dt)
            A = tl.load(
                A_ptrs, mask=(offs_m[:, None] < dim) & (offs_n[None, :] < dstate), other=0.0
            ).to(tl.float32)
            dA = tl.exp(A * dt[:, None])
            dB = B[None, :] * dt[:, None]

            state = state * dA + dB * x[:, None]
            tl.store(state_ptrs, state, mask=state_mask)

            out = tl.sum(state * C[None, :], axis=1)
            if HAS_D:
                D_ptrs = D_ptr + offs_m * stride_D_dim
                D = tl.load(D_ptrs, mask=mask_m, other=0.0).to(tl.float32)
                out += x * D
            if HAS_Z:
                z_ptrs = z_ptr + offs_m * stride_z_dim
                z = tl.load(z_ptrs, mask=mask_m, other=0.0).to(tl.float32)
                out *= z * tl.sigmoid(z)
            tl.store(out_ptrs, out, mask=mask_m)


def selective_state_update(
    state,
    x,
    dt,
    A,
    B,
    C,
    D=None,
    z=None,
    dt_bias=None,
    dt_softplus=False,
    state_batch_indices=None,
    pad_slot_id=PAD_SLOT_ID,
    out=None,
):

    # Make sure all tensors have 4D shapes (or appropriate dims) for indexing
    if state.dim() == 3:
        state = state.unsqueeze(1)
    if x.dim() == 2:
        x = x.unsqueeze(1)
    if dt.dim() == 2:
        dt = dt.unsqueeze(1)
    if A.dim() == 2:
        A = A.unsqueeze(0)
    if B.dim() == 2:
        B = B.unsqueeze(1)
    if C.dim() == 2:
        C = C.unsqueeze(1)
    if D is not None and D.dim() == 1:
        D = D.unsqueeze(0)
    if z is not None and z.dim() == 2:
        z = z.unsqueeze(1)
    if dt_bias is not None and dt_bias.dim() == 1:
        dt_bias = dt_bias.unsqueeze(0)
    if out.dim() == 2:
        out = out.unsqueeze(1)

    batch, nheads, dim, dstate = state.shape
    assert x.shape == (batch, nheads, dim)
    assert dt.shape == x.shape
    assert A.shape == (nheads, dim, dstate)
    ngroups = B.shape[1]
    assert nheads % ngroups == 0, "nheads must be divisible by ngroups"
    assert B.shape == (batch, ngroups, dstate)
    assert C.shape == B.shape
    if D is not None:
        assert D.shape == (nheads, dim)
    if z is not None:
        assert z.shape == x.shape
    if dt_bias is not None:
        assert dt_bias.shape == (nheads, dim)
    if state_batch_indices is not None:
        assert state_batch_indices.shape == (batch,)
    assert out.shape == x.shape

    # Transpose state from (B, H, dim, dstate) to (B, H, dstate, dim)
    # This creates a contiguous tensor in the new layout for better coalescing
    state_t = state.permute(0, 1, 3, 2).contiguous()
    # Compute strides for the transposed state
    s_strides = state_t.stride()

    # Heuristic BLOCK_SIZE_M and num_warps based on dstate
    if dstate <= 16:
        BLOCK_SIZE_M, num_warps = 32, 4
    elif dstate <= 32:
        BLOCK_SIZE_M, num_warps = 16, 4
    elif dstate <= 64:
        BLOCK_SIZE_M, num_warps = 8, 4
    elif dstate <= 128:
        BLOCK_SIZE_M, num_warps = 4, 4
    else:
        BLOCK_SIZE_M, num_warps = 4, 8

    total_dim_blocks = triton.cdiv(dim, BLOCK_SIZE_M)
    dim_programs = max(1, total_dim_blocks // 4)
    grid = lambda META: (dim_programs, batch, nheads)

    z_strides = (z.stride(0), z.stride(1), z.stride(2)) if z is not None else (0, 0, 0)

    tie_hdim = (
        A.stride(-1) == 0
        and A.stride(-2) == 0
        and dt.stride(-1) == 0
        and dt_bias.stride(-1) == 0
    )

    # Launch kernel with transposed state and its strides
    _selective_scan_update_kernel[grid](
        state_t,
        x,
        dt,
        dt_bias,
        A,
        B,
        C,
        D,
        z,
        out,
        state_batch_indices,
        pad_slot_id,
        batch,
        nheads,
        dim,
        dstate,
        nheads // ngroups,
        s_strides[0],  # stride_state_batch
        s_strides[1],  # stride_state_head
        s_strides[2],  # stride_state_dstate (innermost)
        s_strides[3],  # stride_state_dim
        x.stride(0),
        x.stride(1),
        x.stride(2),
        dt.stride(0),
        dt.stride(1),
        dt.stride(2),
        *(dt_bias.stride(0), dt_bias.stride(1)) if dt_bias is not None else (0, 0),
        A.stride(0),
        A.stride(1),
        A.stride(2),
        B.stride(0),
        B.stride(1),
        B.stride(2),
        C.stride(0),
        C.stride(1),
        C.stride(2),
        *(D.stride(0), D.stride(1)) if D is not None else (0, 0),
        z_strides[0],
        z_strides[1],
        z_strides[2],
        out.stride(0),
        out.stride(1),
        out.stride(2),
        dt_softplus,
        tie_hdim,
        BLOCK_SIZE_M,
        num_warps=num_warps,
    )

    # Copy the updated state back to the original layout to preserve the mutation contract
    state.copy_(state_t.permute(0, 1, 3, 2))