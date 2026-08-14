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
    # Linearize the (batch, head) launch grid while remaining compatible with
    # both 1D and legacy 2D launches.
    pid_linear = (
        tl.program_id(0).to(tl.int64) * tl.num_programs(1).to(tl.int64)
        + tl.program_id(1).to(tl.int64)
    )
    H = lses_stride_B // lses_stride_H
    batch_idx = pid_linear // H
    head_idx = pid_linear % H

    TILE_N: tl.constexpr = 128
    lse_max = -float("inf")
    lse_sum = 0.0

    for start in range(0, N_ROUNDED, TILE_N):
        offsets = start + tl.arange(0, TILE_N)
        lse_offsets = (
            offsets * lses_stride_N
            + batch_idx * lses_stride_B
            + head_idx * lses_stride_H
        )
        mask = offsets < N_ROUNDED
        lse_raw = tl.load(lses_ptr + lse_offsets, mask=mask, other=float("-inf"))
        lse_tile = tl.where(
            (lse_raw != lse_raw) | (lse_raw == float("inf")),
            -float("inf"),
            lse_raw,
        )

        tile_max = tl.max(lse_tile, axis=0)
        new_max = tl.maximum(lse_max, tile_max)

        # Avoid NaN from (-inf) - (-inf) when an entire tile is invalid.
        scale = tl.where(
            new_max == -float("inf"),
0,
            tl.exp(lse_max - new_max),
        )
        safe_diff = tl.where(
            new_max == -float("inf"),
            -float("inf"),
            lse_tile - new_max,
        )
        tile_sum = tl.sum(tl.exp(safe_diff), axis=0)
        lse_sum = lse_sum * scale + tile_sum
        lse_max = new_max

    lse_global = tl.where(
        lse_sum <= 0,
        -float("inf"),
        tl.log(lse_sum) + lse_max,
    )

    tl.store(vlse_ptr + pid_linear, lse_global)


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
    # Same linearized grid mapping as the LSE kernel.
    pid_linear = (
        tl.program_id(0).to(tl.int64) * tl.num_programs(1).to(tl.int64)
        + tl.program_id(1).to(tl.int64)
    )
    H = lses_stride_B // lses_stride_H
    batch_idx = pid_linear // H
    head_idx = pid_linear % H

    lse_idx_offsets = (
        lse_idx * lses_stride_N
        + batch_idx * lses_stride_B
        + head_idx * lses_stride_H
    )
    lse_tmp = tl.load(lses_ptr + lse_idx_offsets)

    lse_global = tl.load(vlse_ptr + pid_linear)
    factor = tl.where(
        lse_global == -float("inf"),
0,
        tl.exp(lse_tmp - lse_global),
    )

    for d_start in range(0, HEAD_DIM, TILE_D):
        d_offsets = d_start + tl.arange(0, TILE_D)
        mask = d_offsets < HEAD_DIM

        output_offsets = (
            batch_idx * outputs_stride_B
            + head_idx * outputs_stride_H
            + d_offsets * outputs_stride_D
        )
        output = tl.load(outputs_ptr + output_offsets, mask=mask, other=0.0)
        output = output * factor
        tl.store(new_output_ptr + output_offsets, output, mask=mask)


def correct_attn_cp_out(outputs, lses, lse_idx):
    B, H, D = outputs.shape
    N = lses.shape[0]
    N_ROUNDED = triton.next_power_of_2(N)

    if N < N_ROUNDED:
        pad_shape = (N_ROUNDED - N,) + lses.shape[1:]
        padding = torch.full(pad_shape, float("-inf"), dtype=lses.dtype, device=lses.device)
        lses = torch.cat([lses, padding], dim=0)

    # Permute to (B, H, N) with contiguous N.
    lses = lses.permute(1, 2, 0).contiguous()

    new_output = torch.empty_like(outputs, device=outputs.device)
    vlse = torch.empty((B, H), device=outputs.device, dtype=torch.float32)

    grid = (B * H,)

    # First pass: global LSE for every (batch, head).
    _lse_global_kernel[grid](
        lses,
        vlse,
        lses.stride(2),
        lses.stride(0),
        lses.stride(1),
        N_ROUNDED=N_ROUNDED,
    )

    # Second pass: apply the correction using the same 1D grid.
    _correct_attn_cp_out_kernel[grid](
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