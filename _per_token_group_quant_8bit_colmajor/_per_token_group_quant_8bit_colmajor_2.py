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
    pid0 = tl.program_id(0)  # group index
    pid1 = tl.program_id(1)  # row block index

    m_start = pid1 * BLOCK_M
    m_end = min(m_start + BLOCK_M, M)
    actual_m = m_end - m_start

    col_start = pid0 * group_size
    col_end = min(col_start + group_size, N)
    actual_group_size = col_end - col_start

    if actual_group_size > 0 and actual_m > 0:
        row_offsets = tl.arange(0, BLOCK_M) * y_row_stride
        col_offsets = tl.arange(0, BLOCK)
        mask = (m_start + tl.arange(0, BLOCK_M)[:, None] < M) & (col_start + col_offsets[None, :] < N)
        y_data = tl.load(y_ptr + col_start + row_offsets[:, None] + col_offsets[None, :], mask=mask, other=0.0).to(tl.float32)
        _absmax = tl.max(tl.abs(y_data), axis=1)
        y_s = tl.maximum(_absmax / bit8_max, eps)
        if SCALE_UE8M0:
            y_s = tl.exp2(tl.ceil(tl.log2(tl.abs(y_s))))
        inv_scale = 1.0 / y_s
        quantized = tl.clamp(y_data * inv_scale[:, None], bit8_min, bit8_max).to(y_q_ptr.dtype.element_ty)
        tl.store(y_q_ptr + col_start + row_offsets[:, None] + col_offsets[None, :], quantized, mask=mask)
        row_indices = m_start + tl.arange(0, BLOCK_M)
        scale_mask = row_indices < M
        y_s_ptr_local = y_s_ptr + pid0 * y_s_col_stride + row_indices
        tl.store(y_s_ptr_local, y_s, mask=scale_mask)


def per_token_group_quant_8bit_colmajor(y: torch.Tensor, group_size: int, eps: float = 1e-5, scale_ue8m0: bool = False):
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

    default_block_m = 4
    grid = (num_groups_per_row, triton.cdiv(M, default_block_m))

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