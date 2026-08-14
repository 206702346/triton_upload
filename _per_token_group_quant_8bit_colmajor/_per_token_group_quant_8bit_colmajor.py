from typing import Any, Dict, List, Optional, Tuple

import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": BLOCK_M}, num_warps=num_warps)
        for BLOCK_M in [1, 2, 4, 8, 16]
        for num_warps in [1, 2, 4]
    ],
    key=["M", "N", "group_size"],
)
@triton.jit
def _per_token_group_quant_8bit_colmajor(

    y_ptr,
    y_q_ptr,
    y_s_ptr,
    group_size,

    M,
    N,

    y_row_stride,
    y_q_row_stride,
    y_s_col_stride,

    eps,

    bit8_min,
    bit8_max,

    BLOCK: tl.constexpr,
    SCALE_UE8M0: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    """A Triton-accelerated function to perform per-token-group
    quantization on a tensor.
    This function converts the tensor values into float8 values.
    """

    pid = tl.program_id(0)

    num_groups_per_row = tl.cdiv(N, group_size)

    m_start = pid * BLOCK_M
    m_end = min(m_start + BLOCK_M, M)

    for m in range(m_start, m_end):

        for group_idx in range(num_groups_per_row):

            col_start = group_idx * group_size
            col_end = min(col_start + group_size, N)
            actual_group_size = col_end - col_start

            if actual_group_size > 0:

                y_row_ptr = y_ptr + m * y_row_stride
                y_q_row_ptr = y_q_ptr + m * y_q_row_stride

                y_s_ptr_local = y_s_ptr + group_idx * y_s_col_stride + m

                _absmax = eps.to(tl.float32)

                cols = tl.arange(0, BLOCK)
                for chunk_start in range(0, actual_group_size, BLOCK):
                    chunk_end = min(chunk_start + BLOCK, actual_group_size)
                    chunk_size = chunk_end - chunk_start
                    
                    if chunk_size > 0:
                        mask = cols < chunk_size
                        y_chunk = tl.load(y_row_ptr + col_start + chunk_start + cols, mask=mask, other=0.0).to(tl.float32)

                        chunk_max = tl.max(tl.abs(y_chunk))
                        _absmax = tl.maximum(_absmax, chunk_max)

                y_s = _absmax / bit8_max
                
                if SCALE_UE8M0:
                    y_s = tl.exp2(tl.ceil(tl.log2(tl.abs(y_s))))

                for chunk_start in range(0, actual_group_size, BLOCK):
                    chunk_end = min(chunk_start + BLOCK, actual_group_size)
                    chunk_size = chunk_end - chunk_start
                    
                    if chunk_size > 0:
                        mask = cols < chunk_size
                        y_chunk = tl.load(y_row_ptr + col_start + chunk_start + cols, mask=mask, other=0.0).to(tl.float32)
                        
                        y_q_chunk = tl.clamp(y_chunk / y_s, bit8_min, bit8_max).to(y_q_ptr.dtype.element_ty)

                        tl.store(y_q_row_ptr + col_start + chunk_start + cols, y_q_chunk, mask=mask)

                tl.store(y_s_ptr_local, y_s)

def per_token_group_quant_8bit_colmajor(y: torch.Tensor, group_size: int, eps: float = 1e-5, scale_ue8m0: bool = False):
    """Perform per-token-group quantization on a tensor.
    
    Args:
        y: Input tensor to quantize
        group_size: Size of each quantization group
        eps: Small value to avoid division by zero
        scale_ue8m0: Whether to use UE8M0 scaling
    
    Returns:
        Tuple of (quantized_tensor, scales)
    """
    assert y.is_contiguous(), "Input tensor must be contiguous"
    
    y_shape = y.shape
    y = y.view(-1, y_shape[-1])
    M, N = y.shape

    num_groups_per_row = (N + group_size - 1) // group_size

    y_q = torch.empty_like(y, dtype=torch.int8)
    y_s = torch.empty((num_groups_per_row, M), dtype=torch.float32, device=y.device)

    bit8_min = -128.0
    bit8_max = 127.0

    BLOCK = triton.next_power_of_2(group_size)

    SCALE_UE8M0 = scale_ue8m0

    MAX_GRID_SIZE = 20

    default_block_m = 4
    grid = (min(triton.cdiv(M, default_block_m), MAX_GRID_SIZE),)

    _per_token_group_quant_8bit_colmajor[grid](
        y, 
        y_q, 
        y_s, 
        group_size,
        M,
        N,
        y.stride(0),
        y_q.stride(0),
        y_s.stride(0),
        eps,
        bit8_min,
        bit8_max,
        BLOCK=BLOCK,
        SCALE_UE8M0=SCALE_UE8M0,
    )

    y_q = y_q.view(y_shape)
    
    return y_q, y_s