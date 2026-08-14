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
    global_base_ptr,            # precomputed block_table * BLOCK_SIZE
    token_indices_ptr,
    out_ptr,
    max_num_blocks_per_req: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    bt_stride0,
    bt_stride1,
    ti_stride0,
    ti_stride1,
    out_stride0,
    out_stride1,
):
    token_id = tl.program_id(0)
    tile_id = tl.program_id(1)

    indice_id = tile_id * BLOCK_N + tl.arange(0, BLOCK_N)

    req = tl.load(req_id_ptr + token_id)
    ti_ptr = token_indices_ptr + token_id * ti_stride0 + indice_id * ti_stride1
    tok = tl.load(ti_ptr)

    block_id = tok // BLOCK_SIZE
    inblock_off = tok - block_id * BLOCK_SIZE

    condition = (tok >= 0) & (block_id < max_num_blocks_per_req)
    # load precomputed base address (block_table[req, block_id] * BLOCK_SIZE)
    base = tl.load(global_base_ptr + req * bt_stride0 + block_id * bt_stride1, mask=condition, other=0)

    out_val = tl.where(condition, base + inblock_off, -1)

    out_ptr_ij = out_ptr + token_id * out_stride0 + indice_id * out_stride1
    tl.store(out_ptr_ij, out_val)


def triton_convert_req_index_to_global_index(
    req_id: torch.Tensor,
    block_table: torch.Tensor,
    token_indices: torch.Tensor,
    BLOCK_SIZE: int = 128,
    NUM_TOPK_TOKENS: int = 2048,
    BLOCK_N: int = 256,
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
    tiles_per_row = NUM_TOPK_TOKENS // BLOCK_N

    # Precompute global_base = block_table * BLOCK_SIZE  (int32)
    global_base = torch.mul(block_table, BLOCK_SIZE, out=torch.empty_like(block_table)).contiguous()

    req_id_c = req_id.contiguous()
    token_indices_c = token_indices.contiguous()
    out = torch.empty_like(token_indices_c)

    bt_stride0, bt_stride1 = global_base.stride()
    ti_stride0, ti_stride1 = token_indices_c.stride()
    out_stride0, out_stride1 = out.stride()

    grid = (num_tokens, tiles_per_row)

    _convert_req_index_to_global_index_kernel[grid](
        req_id_c,
        global_base,
        token_indices_c,
        out,
        max_num_blocks_per_req,
        BLOCK_SIZE,
        BLOCK_N,
        bt_stride0,
        bt_stride1,
        ti_stride0,
        ti_stride1,
        out_stride0,
        out_stride1,
        num_warps=4,
        num_stages=2,
    )
    return out