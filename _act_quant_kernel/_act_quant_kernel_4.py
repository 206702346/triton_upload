from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

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
    Quantization kernel with direct storage of scales to S_ptr in row-major layout.
    Each program instance processes BLOCK_M rows and group_size columns.
    With BLOCK_M=1, the scale store is a single element, avoiding strided writes.
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

    # Store scale directly to S_ptr (row-major layout)
    # Address: row * num_groups + group (group = pid_n)
    num_groups = N // group_size
    s_ptrs = S_ptr + rows * num_groups + pid_n
    s_mask = row_mask
    tl.store(s_ptrs, scale, mask=s_mask)


@triton.jit
def _scale_transpose_kernel(
    S_temp_ptr,
    S_ptr,
    M,
    num_groups: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Transpose kernel kept for compatibility; not used in the current pipeline.
    Provided to satisfy the required top-level function contract.
    """
    pid = tl.program_id(0)
    start = pid * BLOCK_SIZE
    offsets = start + tl.arange(0, BLOCK_SIZE)
    total = M * num_groups
    mask = offsets < total

    row = offsets % M
    group = offsets // M

    val = tl.load(S_temp_ptr + group * M + row, mask=mask)
    tl.store(S_ptr + row * num_groups + group, val, mask=mask)


def act_quant(
    x: torch.Tensor, block_size: int = 128, scale_fmt: Optional[str] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantizes the input tensor `x` using block-wise quantization with Triton.
    Scales are stored directly to the output buffer in row-major order,
    eliminating the need for a separate transpose step.

    Args:
        x (torch.Tensor): Contiguous input tensor. Last dimension must be divisible by block_size.
        block_size (int, optional): Block size for quantization. Default: 128.
        scale_fmt (Optional[str], optional): If not None, round scale to power of two. Default: None.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Quantized tensor (float16) and scale tensor (float32).
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

    # Use BLOCK_M=1 to ensure coalesced scale writes (single element per block)
    BLOCK_M = 1
    BLOCK_N = block_size
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, block_size))
    round_scale = scale_fmt is not None

    _act_quant_kernel[grid](
        x_flat,
        y_flat,
        s_flat,
        M,
        N,
        group_size=block_size,
        round_scale=round_scale,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_stages=0 if round_scale else 2,
    )

    return y, s