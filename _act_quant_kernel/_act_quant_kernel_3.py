from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

# ----------------------------------------------------------------------
# Cross-row quantization kernel: one scale per (BLOCK_M x group_size) block
# ----------------------------------------------------------------------

@triton.jit
def _act_quant_kernel(
    X_ptr,
    Y_ptr,
    S_ptr,
    M,
    N,
    group_size: tl.constexpr,
    round_scale: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Quantization kernel that computes a single scale factor for each
    (BLOCK_M x BLOCK_N) block.  Scales are stored directly (row‑major)
    at index  block_row_id * num_groups + block_col_id.
    """
    pid_m = tl.program_id(0)           # block row index
    pid_n = tl.program_id(1)           # block column index (group index)

    fp8_min = -448.0
    fp8_max = 448.0
    fp8_max_inv = 1.0 / fp8_max

    row_start = pid_m * BLOCK_M
    col_start = pid_n * group_size     # group_size is BLOCK_N

    rows = row_start + tl.arange(0, BLOCK_M)
    cols = col_start + tl.arange(0, BLOCK_N)

    row_mask = rows < M
    col_mask = cols < N
    mask = row_mask[:, None] & col_mask[None, :]

    # Load the tile
    x_ptrs = X_ptr + rows[:, None] * N + cols[None, :]
    x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)

    # Single scale for the whole block (max over rows and columns)
    x_abs = tl.abs(x)
    row_max = tl.max(x_abs, axis=1)                 # (BLOCK_M,)
    block_max = tl.max(row_max, axis=0)             # scalar

    amax = tl.maximum(block_max, 1e-4)

    if round_scale:
        log_val = tl.log2(amax * fp8_max_inv)
        log_ceil = tl.ceil(log_val)
        scale = tl.exp2(log_ceil)
    else:
        scale = amax * fp8_max_inv

    # Scale and clamp
    y = x / scale
    y = tl.minimum(tl.maximum(y, fp8_min), fp8_max)

    # Store quantized values
    y_ptrs = Y_ptr + rows[:, None] * N + cols[None, :]
    tl.store(y_ptrs, y, mask=mask)

    # Store the single scale for this block
    num_groups = N // group_size
    s_ptrs = S_ptr + pid_m * num_groups + pid_n
    tl.store(s_ptrs, scale)


# ----------------------------------------------------------------------
# Dummy transpose kernel – kept for compatibility (required function)
# ----------------------------------------------------------------------

@triton.jit
def _scale_transpose_kernel(
    S_temp_ptr,
    S_ptr,
    M,
    num_groups: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Transpose kernel (kept for compatibility).  Not called in the
    cross‑row grouping pipeline – scales are already stored in the
    correct layout.
    """
    pid = tl.program_id(0)
    start = pid * BLOCK_SIZE
    offsets = start + tl.arange(0, BLOCK_SIZE)
    total = M * num_groups
    mask = offsets < total

    # Convert linear index to (row, group)
    row = offsets % M
    group = offsets // M

    # Load from temp buffer (group * M + row)
    val = tl.load(S_temp_ptr + group * M + row, mask=mask)

    # Store to final layout (row * num_groups + group)
    tl.store(S_ptr + row * num_groups + group, val, mask=mask)


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------

def act_quant(
    x: torch.Tensor, block_size: int = 128, scale_fmt: Optional[str] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantizes the input tensor `x` using **cross‑row** block‑wise
    quantization.  A single scale factor is computed for every
    (BLOCK_M x group_size) tile, which reduces scale memory by a
    factor of BLOCK_M compared to per‑row scaling.

    Args:
        x (torch.Tensor): Contiguous input tensor.  Last dimension must
                          be divisible by `block_size`.
        block_size (int, optional): Column block size.  Default: 128.
        scale_fmt (Optional[str], optional): If not None, round the
                                              scale to a power of two.
                                              Default: None.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - Quantized tensor (float16), same shape as `x`.
            - Scale tensor (float32) of shape
              (ceil(M / BLOCK_M), num_groups) where M = * x.numel() / N
              and num_groups = N // block_size.  BLOCK_M = 4 is used.
    """
    assert x.is_contiguous(), "Input tensor must be contiguous"
    assert (
        x.size(-1) % block_size == 0
    ), f"Last dimension must be divisible by block_size (block_size={block_size})"

    N = x.size(-1)
    x_flat = x.view(-1, N)
    M = x_flat.size(0)
    num_groups = N // block_size

    y = torch.empty_like(x, dtype=torch.float16)
    y_flat = y.view(-1, N)

    # Scale shape: (num_block_rows, num_groups)
    BLOCK_M = 4
    num_block_rows = (M + BLOCK_M - 1) // BLOCK_M
    s = x.new_empty((num_block_rows, num_groups), dtype=torch.float32)

    BLOCK_N = block_size
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, block_size))
    round_scale = scale_fmt is not None

    _act_quant_kernel[grid](
        x_flat,
        y_flat,
        s,
        M,
        N,
        group_size=block_size,
        round_scale=round_scale,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_stages=0 if round_scale else 2,
    )

    return y, s