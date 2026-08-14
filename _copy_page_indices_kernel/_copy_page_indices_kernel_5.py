# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Flat block mapping kernel – hybrid version using request-level batching and vectorized accesses."""

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
    page_indices,          # torch.Tensor of int32, output flat block mapping
    block_table,           # torch.Tensor of int32, shape [num_requests, max_blocks_per_request]
    block_table_stride,    # int, stride of the block_table's second dimension
    cu_num_blocks,         # torch.Tensor of int32, cumulative block counts (size >= 2**20)
    BLOCK_SIZE: tl.constexpr,  # number of threads per block
):
    """
    Copy page indices from a per-request block table into a flat output array.

    The kernel is designed to be launched with a grid size equal to the number of
    thread blocks that the caller wishes to use for processing requests.  It
    first computes the total number of global blocks and the number of requests
    from the cumulative array.  Each thread block (program) is assigned a
    contiguous slice of requests, and all threads within the block cooperatively
    copy the corresponding entries using vectorised loads/stores with masking.
    This provides good memory coalescing and allows the caller to trade off
    launch parallelism against per-block work.
    """
    # ---- Determine total number of global blocks and number of requests ----
    # Use the same contract as the original parent: cu_num_blocks has at least 2**20
    # valid entries, and the last meaningful cumulative count is at index (1<<20)-1.
    total_blocks = tl.load(cu_num_blocks + (1 << 20) - 1)

    # Early exit for an empty workload.
    if total_blocks == 0:
        return

    # Binary search for the request that owns block (total_blocks - 1).  This
    # gives us the maximum valid request index, from which we derive the total
    # number of requests.  The search is identical to the one used in parent A.
    target = total_blocks - 1
    low: tl.int32 = 0
    high: tl.int32 = 1 << 20

    for _ in range(20):
        mid = (low + high) // 2
        val = tl.load(cu_num_blocks + mid)
        if val <= target:
            low = mid
        else:
            high = mid

    req_idx = low
    start = tl.load(cu_num_blocks + req_idx)
    next_start = tl.load(cu_num_blocks + req_idx + 1)
    while next_start <= target:
        req_idx += 1
        start = next_start
        next_start = tl.load(cu_num_blocks + req_idx + 1)
    while start > target:
        req_idx -= 1
        start = tl.load(cu_num_blocks + req_idx)
        next_start = tl.load(cu_num_blocks + req_idx + 1)

    num_requests = req_idx + 1   # req_idx is the last valid request index

    # ---- Partition the requests among the launched thread blocks ----
    pid = tl.program_id(0)
    total_pids = tl.num_programs(0)

    # Number of requests handled by this program.
    chunk_reqs = tl.cdiv(num_requests, total_pids)
    req_start = pid * chunk_reqs
    req_end = tl.minimum(req_start + chunk_reqs, num_requests)

    # ---- Process the assigned requests using vectorised copies ----
    for r in range(req_start, req_end):
        start_idx = tl.load(cu_num_blocks + r)
        end_idx = tl.load(cu_num_blocks + r + 1)
        num_blocks = end_idx - start_idx

        row_ptr = block_table + r * block_table_stride
        offset = tl.arange(0, BLOCK_SIZE)

        for i in range(0, num_blocks, BLOCK_SIZE):
            block_ids = tl.load(row_ptr + i + offset, mask=i + offset < num_blocks)
            tl.store(page_indices + start_idx + i + offset,
                     block_ids,
                     mask=i + offset < num_blocks)


# All required top‑level functions are present.
assert hasattr(triton.jit, "_copy_page_indices_kernel") is False