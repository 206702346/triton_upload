

import math

import torch
import triton
import triton.language as tl

@triton.jit
def softplus(dt):
    dt = tl.where(dt <= 20.0, tl.math.log(tl.math.exp(dt) + 1), dt)
    return dt

@triton.jit
def _chunk_state_fwd_kernel(

    x_ptr,
    b_ptr,
    states_ptr,
    dt_ptr,
    dA_cumsum_ptr,
    seq_idx_ptr,

    hdim,
    dstate,
    chunk_size,
    batch,
    seqlen,
    nheads_ngroups_ratio,

    stride_x_batch,
    stride_x_seqlen,
    stride_x_head,
    stride_x_hdim,
    stride_b_batch,
    stride_b_seqlen,
    stride_b_head,
    stride_b_dstate,
    stride_states_batch,
    stride_states_chunk,
    stride_states_head,
    stride_states_hdim,
    stride_states_dstate,
    stride_dt_batch,
    stride_dt_chunk,
    stride_dt_head,
    stride_dt_csize,
    stride_dA_cs_batch,
    stride_dA_cs_chunk,
    stride_dA_cs_head,
    stride_dA_cs_csize,
    stride_seq_idx_batch,
    stride_seq_idx_seqlen,

    HAS_SEQ_IDX: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr = 16,
    BLOCK_SIZE_N: tl.constexpr = 16,
    BLOCK_SIZE_K: tl.constexpr = 16,
):
    pid_bc = tl.program_id(axis=1).to(tl.int64)
    pid_c = pid_bc // batch
    pid_b = pid_bc - pid_c * batch
    pid_h = tl.program_id(axis=2)
    num_pid_n = tl.cdiv(dstate, BLOCK_SIZE_N)
    pid_m = tl.program_id(axis=0) // num_pid_n
    pid_n = tl.program_id(axis=0) % num_pid_n
    b_ptr += (
        pid_b * stride_b_batch
        + pid_c * chunk_size * stride_b_seqlen
        + (pid_h // nheads_ngroups_ratio) * stride_b_head
    )
    x_ptr += (
        pid_b * stride_x_batch
        + pid_c * chunk_size * stride_x_seqlen
        + pid_h * stride_x_head
    )
    dt_ptr += pid_b * stride_dt_batch + pid_c * stride_dt_chunk + pid_h * stride_dt_head
    dA_cumsum_ptr += (
        pid_b * stride_dA_cs_batch
        + pid_c * stride_dA_cs_chunk
        + pid_h * stride_dA_cs_head
    )
    if HAS_SEQ_IDX:
        seq_idx_ptr += (
            pid_b * stride_seq_idx_batch + pid_c * chunk_size * stride_seq_idx_seqlen
        )

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    x_ptrs = x_ptr + (
        offs_m[:, None] * stride_x_hdim + offs_k[None, :] * stride_x_seqlen
    )
    b_ptrs = b_ptr + (
        offs_n[None, :] * stride_b_dstate + offs_k[:, None] * stride_b_seqlen
    )
    dt_ptrs = dt_ptr + offs_k * stride_dt_csize
    dA_cs_last = tl.load(dA_cumsum_ptr + (chunk_size - 1) * stride_dA_cs_csize).to(
        tl.float32
    )
    dA_cumsum_ptrs = dA_cumsum_ptr + offs_k * stride_dA_cs_csize
    if HAS_SEQ_IDX:
        seq_idx_ptrs = seq_idx_ptr + offs_k * stride_seq_idx_seqlen

    chunk_size_limit = min(chunk_size, seqlen - pid_c * chunk_size)
    if HAS_SEQ_IDX:
        seq_idx_last = tl.load(
            seq_idx_ptr + (chunk_size_limit - 1) * stride_seq_idx_seqlen
        )

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, chunk_size_limit, BLOCK_SIZE_K):
        x = tl.load(
            x_ptrs,
            mask=(offs_m[:, None] < hdim) & (offs_k[None, :] < chunk_size_limit - k),
            other=0.0,
        )
        b = tl.load(
            b_ptrs,
            mask=(offs_k[:, None] < chunk_size_limit - k) & (offs_n[None, :] < dstate),
            other=0.0,
        ).to(tl.float32)
        dA_cs_k = tl.load(
            dA_cumsum_ptrs, mask=offs_k < chunk_size_limit - k, other=0.0
        ).to(tl.float32)
        if HAS_SEQ_IDX:
            seq_idx_k = tl.load(
                seq_idx_ptrs, mask=offs_k < chunk_size_limit - k, other=-1
            )
        dt_k = tl.load(dt_ptrs, mask=offs_k < chunk_size_limit - k, other=0.0).to(
            tl.float32
        )
        if not HAS_SEQ_IDX:
            scale = tl.exp(dA_cs_last - dA_cs_k) * dt_k
        else:
            scale = tl.where(
                seq_idx_k == seq_idx_last, tl.exp(dA_cs_last - dA_cs_k) * dt_k, 0.0
            )
        b *= scale[:, None]
        b = b.to(x_ptr.dtype.element_ty)
        acc += tl.dot(x, b)
        x_ptrs += BLOCK_SIZE_K * stride_x_seqlen
        b_ptrs += BLOCK_SIZE_K * stride_b_seqlen
        dt_ptrs += BLOCK_SIZE_K * stride_dt_csize
        dA_cumsum_ptrs += BLOCK_SIZE_K * stride_dA_cs_csize
        if HAS_SEQ_IDX:
            seq_idx_ptrs += BLOCK_SIZE_K * stride_seq_idx_seqlen
    states = acc.to(states_ptr.dtype.element_ty)

    states_ptr += (
        pid_b * stride_states_batch
        + pid_c * stride_states_chunk
        + pid_h * stride_states_head
    )
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    states_ptrs = states_ptr + (
        offs_m[:, None] * stride_states_hdim + offs_n[None, :] * stride_states_dstate
    )
    c_mask = (offs_m[:, None] < hdim) & (offs_n[None, :] < dstate)
    tl.store(states_ptrs, states, mask=c_mask)

def _chunk_state_fwd(
    B, x, dt, dA_cumsum, seq_idx=None, states=None, states_in_fp32=True
):
    batch, seqlen, nheads, headdim = x.shape
    _, _, nchunks, chunk_size = dt.shape
    _, _, ngroups, dstate = B.shape
    assert nheads % ngroups == 0
    assert B.shape == (batch, seqlen, ngroups, dstate)
    assert dt.shape == (batch, nheads, nchunks, chunk_size)
    assert dA_cumsum.shape == dt.shape
    if seq_idx is not None:
        assert seq_idx.shape == (batch, seqlen)
    if states is not None:
        assert states.shape == (batch, nchunks, nheads, headdim, dstate)
    else:
        states_dtype = torch.float32 if states_in_fp32 else B.dtype
        states = torch.empty(
            (batch, nchunks, nheads, headdim, dstate),
            device=B.device,
            dtype=states_dtype,
        )
    grid = lambda META: (
        triton.cdiv(headdim, META["BLOCK_SIZE_M"])
        * triton.cdiv(dstate, META["BLOCK_SIZE_N"]),
        batch * nchunks,
        nheads,
    )
    _chunk_state_fwd_kernel[grid](
        x,
        B,
        states,
        dt,
        dA_cumsum,
        seq_idx,
        headdim,
        dstate,
        chunk_size,
        batch,
        seqlen,
        nheads // ngroups,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        x.stride(3),
        B.stride(0),
        B.stride(1),
        B.stride(2),
        B.stride(-1),
        states.stride(0),
        states.stride(1),
        states.stride(2),
        states.stride(3),
        states.stride(4),
        dt.stride(0),
        dt.stride(2),
        dt.stride(1),
        dt.stride(3),
        dA_cumsum.stride(0),
        dA_cumsum.stride(2),
        dA_cumsum.stride(1),
        dA_cumsum.stride(3),
        *(
            (seq_idx.stride(0), seq_idx.stride(1))
            if seq_idx is not None
            else (0, 0)
        ),
        HAS_SEQ_IDX=seq_idx is not None,
    )
    return states
