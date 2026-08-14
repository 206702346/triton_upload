from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Optional

import numpy as np
import torch
import triton
import triton.language as tl

PAD_SLOT_ID = -1
logger = None


@triton.jit
def _convert_req_index_to_global_index_kernel(
    req_id_ptr,
    block_table_ptr,
    token_indices_ptr,
    out_ptr,
    max_num_blocks_per_req: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    num_topk_tokens,
    bt_stride0,
    bt_stride1,
    ti_stride0,
    ti_stride1,
    out_stride0,
    out_stride1,
):
    token_id = tl.program_id(0)
    req = tl.load(req_id_ptr + token_id)

    ti_base = token_indices_ptr + token_id * ti_stride0
    out_base = out_ptr + token_id * out_stride0

    num_blocks = num_topk_tokens // BLOCK_N
    for tile_id in tl.range(0, num_blocks):
        indice_id = tile_id * BLOCK_N + tl.arange(0, BLOCK_N)

        ti_ptr = ti_base + indice_id * ti_stride1
        # Alignment hint: the underlying token_indices contiguous tensor is 16‑byte aligned.
        tl.multiple_of(ti_ptr, [16])
        tok = tl.load(ti_ptr)

        block_id = tok // BLOCK_SIZE
        # avoid second division by computing remainder as subtraction
        inblock_off = tok - block_id * BLOCK_SIZE

        condition = (tok >= 0) & (block_id < max_num_blocks_per_req)
        bt_ptr = block_table_ptr + req * bt_stride0 + block_id * bt_stride1
        base = tl.load(bt_ptr, mask=condition, other=0)

        out_val = tl.where(condition, base * BLOCK_SIZE + inblock_off, -1)

        out_ptr_ij = out_base + indice_id * out_stride1
        # Alignment hint: the underlying output tensor is 16‑byte aligned.
        tl.multiple_of(out_ptr_ij, [16])
        tl.store(out_ptr_ij, out_val)


def triton_convert_req_index_to_global_index(
    req_id: torch.Tensor,
    block_table: torch.Tensor,
    token_indices: torch.Tensor,
    BLOCK_SIZE: int = 128,
    NUM_TOPK_TOKENS: int = 2048,
    BLOCK_N: int = 512,
):
    assert req_id.dtype == torch.int32
    assert block_table.dtype == torch.int32
    assert token_indices.dtype == torch.int32
    assert token_indices.shape[1] == NUM_TOPK_TOKENS
    assert NUM_TOPK_TOKENS % BLOCK_N == 0, (
        f"NUM_TOPK_TOKENS ({NUM_TOPK_TOKENS}) must be divisible by BLOCK_N ({BLOCK_N})"
    )

    num_tokens = req_id.shape[0]
    num_requests, max_num_blocks_per_req = block_table.shape
    total_elements = num_tokens * NUM_TOPK_TOKENS

    # Choose a tile size that keeps 4 elements/thread when there is enough work.
    # The kernel now launches one program per token and internally loops over tiles.
    if total_elements >= 8192:
        chosen_block_n = BLOCK_N
        chosen_num_warps = 16
        for candidate in (1024, 512, 256, 128):
            if NUM_TOPK_TOKENS % candidate == 0:
                chosen_block_n = candidate
                chosen_num_warps = candidate // 128
                break
    else:
        chosen_block_n = BLOCK_N
        chosen_num_warps = max(1, min(16, BLOCK_N // 32))
        if chosen_num_warps * 32 > chosen_block_n:
            chosen_num_warps = 16

    req_id_c = req_id.contiguous()
    block_table_c = block_table.contiguous()
    token_indices_c = token_indices.contiguous()
    out = torch.empty_like(token_indices_c)

    bt_stride0, bt_stride1 = block_table_c.stride()
    ti_stride0, ti_stride1 = token_indices_c.stride()
    out_stride0, out_stride1 = out.stride()

    grid = (num_tokens,)

    _convert_req_index_to_global_index_kernel[grid](
        req_id_c,
        block_table_c,
        token_indices_c,
        out,
        max_num_blocks_per_req,
        BLOCK_SIZE,
        chosen_block_n,
        NUM_TOPK_TOKENS,
        bt_stride0,
        bt_stride1,
        ti_stride0,
        ti_stride1,
        out_stride0,
        out_stride1,
        num_warps=chosen_num_warps,
        num_stages=2,
    )
    return out