from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

@triton.jit
def _act_quant_kernel(
    X_ptr,
    Y_ptr,
    S_temp_ptr,
    M,
    N,
    group_size: tl.constexpr,
    round_scale: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Triton kernel for activation quantization with coalesced scale writes.
    Scales are written to a temporary buffer in column-major order (groups × rows)
    to achieve coalesced stores when BLOCK_M > 1.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    fp8_min = -448.0
    fp8_max = 448.0
    fp8_max_inv = 1.0 / fp8_max

    row_start = pid_m * BLOCK_M
    col_start = pid_n * group_size

    rows = row_start + tl.arange(0, BLOCK_M)
    cols = col_start + tl.arange(0, BLOCK_N)

    row_mask = rows < M
    col_mask = cols < N
    mask = row_mask[:, None] & col_mask[None, :]

    x_ptrs = X_ptr + rows[:, None] * N + cols[None, :]
    x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)

    x_abs = tl.abs(x)
    amax = tl.max(x_abs, axis=1)

    amax = tl.maximum(amax, 1e-4)

    if round_scale:
        log_val = tl.log2(amax * fp8_max_inv)
        log_ceil = tl.ceil(log_val)
        scale = tl.exp2(log_ceil)
    else:
        scale = amax * fp8_max_inv

    scale_broadcast = scale[:, None]
    y = x / scale_broadcast
    y = tl.minimum(tl.maximum(y, fp8_min), fp8_max)

    y_ptrs = Y_ptr + rows[:, None] * N + cols[None, :]
    tl.store(y_ptrs, y, mask=mask)

    # Store scales coalesced: group_id * M + row_id
    # Each block writes BLOCK_M contiguous elements starting at
    # group * M + row_start. This is coalesced.
    s_temp_ptrs = S_temp_ptr + pid_n * M + rows
    s_mask = row_mask
    tl.store(s_temp_ptrs, scale, mask=s_mask)


@triton.jit
def _scale_transpose_kernel(
    S_temp_ptr,
    S_ptr,
    M,
    num_groups: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Transpose the temporary scale buffer (num_groups × M) into row-major (M × num_groups).
    Each block processes a tile of BLOCK_SIZE elements.
    """
    pid = tl.program_id(0)
    start = pid * BLOCK_SIZE
    offsets = start + tl.arange(0, BLOCK_SIZE)
    total = M * num_groups
    mask = offsets < total

    # Convert linear index (offsets) to (row, group)
    row = offsets % M
    group = offsets // M

    # Load from temp: group * M + row
    val = tl.load(S_temp_ptr + group * M + row, mask=mask)

    # Store to final: row * num_groups + group
    tl.store(S_ptr + row * num_groups + group, val, mask=mask)


def act_quant(
    x: torch.Tensor, block_size: int = 128, scale_fmt: Optional[str] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantizes the input tensor `x` using block-wise quantization with Triton.
    Uses a two-pass approach to ensure coalesced scale writes: first stores scales
    in a temporary transposed buffer, then transposes them into the final row-major format.

    Args:
        x (torch.Tensor): The input tensor to be quantized. Must be contiguous and its
                          last dimension size must be divisible by `block_size`.
        block_size (int, optional): The size of the blocks for quantization. Default is 128.
        scale_fmt (Optional[str], optional): Format of the scale. Default is None.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - The quantized tensor with dtype `torch.float8_e4m3fn`.
            - A tensor of scaling factors with dtype `torch.float32` in row-major order.
    """
    assert x.is_contiguous(), "Input tensor must be contiguous"
    assert (
        x.size(-1) % block_size == 0
    ), f"Last dimension size must be divisible by block_size (block_size={block_size})"

    N = x.size(-1)
    x_flat = x.view(-1, N)
    M = x_flat.size(0)
    num_groups = N // block_size

    y = torch.empty_like(x, dtype=torch.float16)
    y_flat = y.view(-1, N)

    # Final scale tensor: row-major (M x num_groups)
    s = x.new_empty(*x.size()[:-1], num_groups, dtype=torch.float32)
    s_flat = s.view(-1, num_groups)

    # Temporary buffer for coalesced scale writes: shape (num_groups, M)
    s_temp = x.new_empty(num_groups, M, dtype=torch.float32)

    BLOCK_M = 4  # Increased from 1 to exploit coalesced writes
    BLOCK_N = block_size
    grid_quant = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, block_size))
    round_scale = scale_fmt is not None

    # First pass: quantize and write scales coalesced to s_temp
    _act_quant_kernel[grid_quant](
        x_flat,
        y_flat,
        s_temp,
        M,
        N,
        group_size=block_size,
        round_scale=round_scale,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_stages=0 if round_scale else 2,
    )

    # Second pass: transpose s_temp into s_flat (row-major)
    # Tune BLOCK_SIZE for efficiency (e.g., 256)
    BLOCK_SIZE = 256
    grid_transpose = (triton.cdiv(M * num_groups, BLOCK_SIZE),)
    _scale_transpose_kernel[grid_transpose](
        s_temp,
        s_flat,
        M,
        num_groups,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return y, s