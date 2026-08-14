import torch
import triton
import triton.language as tl

@triton.jit
def _lse_global_kernel(
    lses_ptr,
    vlse_ptr,
    lses_stride_N,
    lses_stride_B,
    lses_stride_H,
    N_ROUNDED: tl.constexpr,
):
    batch_idx = tl.program_id(0).to(tl.int64)
    head_idx = tl.program_id(1).to(tl.int64)

    TILE_N: tl.constexpr = 256  # wider tile, fully covered by padding
    lse_max = -float("inf")
    lse_sum = 0.0

    # All tiles are complete because N_ROUNDED is a multiple of TILE_N
    for start in range(0, N_ROUNDED, TILE_N):
        offsets = start + tl.arange(0, TILE_N)
        lse_offsets = (offsets * lses_stride_N
                       + batch_idx * lses_stride_B
                       + head_idx * lses_stride_H)
        lse_raw = tl.load(lses_ptr + lse_offsets)   # no mask needed
        # Clamp out any NaN or +inf that could break online LSE
        lse_tile = tl.where((lse_raw != lse_raw) | (lse_raw == float("inf")),
                            -float("inf"), lse_raw)

        tile_max = tl.max(lse_tile, axis=0)
        new_max = tl.maximum(lse_max, tile_max)
        # stable exponential recombination
        lse_sum = lse_sum * tl.exp(lse_max - new_max) + tl.sum(tl.exp(lse_tile - new_max), axis=0)
        lse_max = new_max

    lse_global = tl.where(lse_sum <= 0, -float("inf"), tl.log(lse_sum) + lse_max)

    H = tl.num_programs(1)
    vlse_offset = batch_idx * H + head_idx
    tl.store(vlse_ptr + vlse_offset, lse_global)


@triton.jit
def _correct_attn_cp_out_kernel(
    outputs_ptr,
    new_output_ptr,
    lses_ptr,
    vlse_ptr,
    outputs_stride_B,
    outputs_stride_H,
    outputs_stride_D,
    lses_stride_N,
    lses_stride_B,
    lses_stride_H,
    lse_idx,
    HEAD_DIM: tl.constexpr,
    TILE_D: tl.constexpr,
    N_ROUNDED: tl.constexpr,
):
    batch_idx = tl.program_id(0).to(tl.int64)
    head_idx = tl.program_id(1).to(tl.int64)

    # Load the specific LSE value (always valid because lse_idx < original N)
    lse_idx_offsets = (lse_idx * lses_stride_N
                       + batch_idx * lses_stride_B
                       + head_idx * lses_stride_H)
    lse_tmp = tl.load(lses_ptr + lse_idx_offsets)

    H = tl.num_programs(1)
    vlse_offset = batch_idx * H + head_idx
    lse_global = tl.load(vlse_ptr + vlse_offset)

    factor = tl.where(lse_global == -float("inf"), 0.0,
                     tl.exp(lse_tmp - lse_global))

    # Tile the head dimension inside the kernel
    for d_start in range(0, HEAD_DIM, TILE_D):
        d_offsets = d_start + tl.arange(0, TILE_D)
        mask = d_offsets < HEAD_DIM

        output_offsets = (batch_idx * outputs_stride_B
                          + head_idx * outputs_stride_H
                          + d_offsets * outputs_stride_D)
        output = tl.load(outputs_ptr + output_offsets, mask=mask, other=0.0)
        output = output * factor
        tl.store(new_output_ptr + output_offsets, output, mask=mask)


def correct_attn_cp_out(outputs, lses, lse_idx):
    B, H, D = outputs.shape
    N = lses.shape[0]
    # Pad N to the next multiple of 256 so that every tile is full and mask‑free
    N_ROUNDED = ((N + 255) // 256) * 256

    if N < N_ROUNDED:
        pad_shape = (N_ROUNDED - N,) + lses.shape[1:]
        padding = torch.full(pad_shape, float("-inf"), dtype=lses.dtype, device=lses.device)
        lses = torch.cat([lses, padding], dim=0)

    # Permute to (B, H, N) with contiguous N dimension (stride 1)
    lses = lses.permute(1, 2, 0).contiguous()

    new_output = torch.empty_like(outputs, device=outputs.device)
    vlse = torch.empty((B, H), device=outputs.device, dtype=torch.float32)

    # First pass: global LSE per (batch, head)
    grid_lse = (B, H)
    _lse_global_kernel[grid_lse](
        lses,
        vlse,
        lses.stride(2),  # N stride = 1 after contiguous
        lses.stride(0),  # B stride
        lses.stride(1),  # H stride
        N_ROUNDED=N_ROUNDED,
    )

    # Second pass: rescale outputs
    _correct_attn_cp_out_kernel[(B, H)](
        outputs,
        new_output,
        lses,
        vlse,
        outputs.stride(0),
        outputs.stride(1),
        outputs.stride(2),
        lses.stride(2),
        lses.stride(0),
        lses.stride(1),
        lse_idx,
        HEAD_DIM=D,
        TILE_D=64,
        N_ROUNDED=N_ROUNDED,
    )

    return new_output, vlse