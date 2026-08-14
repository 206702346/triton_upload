

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Optional

import numpy as np
import torch
import triton
import triton.language as tl

PAD_SLOT_ID = -1

logger = None

"""
NOTE: FlashMLA Sparse uses an fp8 cache with the following format

In the "FP8 with scale" format, each token's KV cache is 656 Bytes,
structured as:
-   **First 512 bytes:** The "quantized NoPE" part, containing 512
    `float8_e4m3` values.
-   **Next 16 bytes:** Scale factors, containing 4 `float32` values.
    The first `float32` is the scale for the first 128 `float8_e4m3` values,
    the second for the next 128, and so on.
-   **Last 128 bytes:** The "RoPE" part, containing 64 `bfloat16` values. This
    part is not quantized for accuracy.
"""

@triton.jit
def _convert_req_index_to_global_index_kernel(
    req_id_ptr,
    block_table_ptr,
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

    is_invalid_tok = tok < 0

    block_id = tok // BLOCK_SIZE
    inblock_off = tok % BLOCK_SIZE

    valid_block = (block_id < max_num_blocks_per_req) & (block_id >= 0)
    bt_ptr = block_table_ptr + req * bt_stride0 + block_id * bt_stride1
    base = tl.load(bt_ptr, mask=valid_block, other=0)

    out_val = tl.where(
        is_invalid_tok | (~valid_block), -1, base * BLOCK_SIZE + inblock_off
    )

    out_ptr_ij = out_ptr + token_id * out_stride0 + indice_id * out_stride1
    tl.store(out_ptr_ij, out_val)

def triton_convert_req_index_to_global_index(
    req_id: torch.Tensor,
    block_table: torch.Tensor,
    token_indices: torch.Tensor,
    BLOCK_SIZE: int = 64,
    NUM_TOPK_TOKENS: int = 2048,
    BLOCK_N: int = 128,
):
    """
    out[token_id, indice_id] =
        block_table[req_id[token_id],
            token_indices[token_id, indice_id] // BLOCK_SIZE] * BLOCK_SIZE
        + token_indices[token_id, indice_id] % BLOCK_SIZE

    Only when token_indices[token_id, indice_id] == -1 do we output -1.
    For safety, we also output -1 if the derived block_id would be
        out-of-bounds.
    """
    assert req_id.dtype == torch.int32
    assert block_table.dtype == torch.int32
    assert token_indices.dtype == torch.int32
    assert token_indices.shape[1] == NUM_TOPK_TOKENS
    assert NUM_TOPK_TOKENS % BLOCK_N == 0, (
        f"NUM_TOPK_TOKENS ({NUM_TOPK_TOKENS}) must be divisible byBLOCK_N ({BLOCK_N})"
    )

    num_tokens = req_id.shape[0]
    num_requests, max_num_blocks_per_req = block_table.shape
    tiles_per_row = NUM_TOPK_TOKENS // BLOCK_N

    req_id_c = req_id.contiguous()
    block_table_c = block_table.contiguous()
    token_indices_c = token_indices.contiguous()
    out = torch.empty_like(token_indices_c)

    bt_stride0, bt_stride1 = block_table_c.stride()
    ti_stride0, ti_stride1 = token_indices_c.stride()
    out_stride0, out_stride1 = out.stride()

    grid = (num_tokens, tiles_per_row)

    _convert_req_index_to_global_index_kernel[grid](
        req_id_c,
        block_table_c,
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
    )
    return out
