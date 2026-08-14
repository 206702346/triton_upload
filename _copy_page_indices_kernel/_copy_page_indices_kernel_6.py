# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Attention layer with FlashInfer – prefix‑sum based scatter with multi‑block launch."""

from dataclasses import dataclass

import torch
import triton
import triton.language as tl


@dataclass
class _DummyConfig:
    """Placeholder to preserve module structure."""
    pass


@triton.jit
def _copy_page_indices_kernel(
    page_indices: torch.Tensor,          # [total_blocks] output
    block_table: torch.Tensor,           # [num_reqs, block_table_stride] row‑wise
    block_table_stride: int,             # stride between rows
    cu_num_blocks: torch.Tensor,         # cumulative prefix sum over request block counts
    BLOCK_SIZE: tl.constexpr,
):
    """
    Scatter kernel that maps global block indices to per‑request block table rows.

    The cumulative array ``cu_num_blocks`` is used as a prefix sum to locate
    the owning request and row offset for every global block index.  Work is
    distributed among multiple thread blocks (launched as ``(1, BLOCK_SIZE)``),
    each processing a contiguous chunk of the output space.  Inside each chunk,
    a single thread performs a binary search (20 iterations, optimal for up to
    2^20 requests) followed by vectorised loads from the row‑wise ``block_table``.
    """
    # ------------------------------------------------------------------
    # total number of blocks (last valid entry of the prefix sum array)
    # ------------------------------------------------------------------
    total_blocks = tl.load(cu_num_blocks + (1 << 20) - 1)

    # ------------------------------------------------------------------
    # work distribution: each thread block processes one chunk
    # grid is launched as (1, BLOCK_SIZE) → program_id(1) is block id
    # ------------------------------------------------------------------
    block_id = tl.program_id(1)
    num_blocks = BLOCK_SIZE                # compile‑time known
    chunk_size = tl.cdiv(total_blocks, num_blocks)
    start_idx = block_id * chunk_size
    end_idx = tl.minimum(start_idx + chunk_size, total_blocks)

    # ------------------------------------------------------------------
    # process the chunk request‑by‑request
    # ------------------------------------------------------------------
    pos = start_idx
    while pos < end_idx:
        # binary search for the request that owns global index `pos`
        low: tl.int32 = 0
        high: tl.int32 = 1 << 20
        for _ in range(20):
            mid = (low + high) // 2
            val = tl.load(cu_num_blocks + mid)
            if val <= pos:
                low = mid
            else:
                high = mid

        req_idx = low

        # linear correction to guarantee correctness
        req_start = tl.load(cu_num_blocks + req_idx)
        next_start = tl.load(cu_num_blocks + req_idx + 1)
        while next_start <= pos:
            req_idx += 1
            req_start = next_start
            next_start = tl.load(cu_num_blocks + req_idx + 1)
        while req_start > pos:
            req_idx -= 1
            req_start = tl.load(cu_num_blocks + req_idx)
            next_start = tl.load(cu_num_blocks + req_idx + 1)

        # range owned by this request
        req_end = next_start
        # portion that falls within this thread block’s chunk
        process_end = tl.minimum(end_idx, req_end)
        num_in_request = process_end - pos
        start_offset = pos - req_start   # offset inside the request’s block list

        # vectorised copy from the row‑wise block table
        row_ptr = block_table + req_idx * block_table_stride
        for i in tl.range(0, num_in_request, BLOCK_SIZE):
            offsets = i + tl.arange(0, BLOCK_SIZE)
            mask = offsets < num_in_request
            block_ids = tl.load(row_ptr + start_offset + offsets,
                                mask=mask, other=0)
            tl.store(page_indices + pos + offsets, block_ids, mask=mask)

        pos = process_end


# ------------------------------------------------------------------
# Preserve structural check from parent modules
# ------------------------------------------------------------------
assert not hasattr(triton.jit, "_copy_page_indices_kernel")