import torch
import triton
import triton.language as tl


@triton.jit
def _count_expert_num_tokens(
    topk_ids_ptr,
    expert_num_tokens_ptr,
    num_experts: tl.constexpr,
    topk_numel: tl.constexpr,
    expert_map_ptr,
    HAS_EXPERT_MAP: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Double‑buffered version of the global‑map counting kernel.
    Prefetches the next block’s topk_ids while the current one is being processed,
    hiding global memory latency.
    """
    curr_expert = tl.program_id(0)
    if curr_expert >= num_experts:
        return

    offsets = tl.arange(0, BLOCK_SIZE)
    num_blocks = tl.cdiv(topk_numel, BLOCK_SIZE)
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.int32)
    base_ptr = topk_ids_ptr + offsets

    # Double‑buffering: two registers for expert_ids
    cur_ids = tl.zeros((BLOCK_SIZE,), dtype=tl.int32)
    next_ids = tl.zeros((BLOCK_SIZE,), dtype=tl.int32)

    # Prefetch the very first block
    mask0 = offsets < topk_numel
    cur_ids = tl.load(base_ptr, mask=mask0, other=-1)

    # Process all blocks except the last one, prefetching the next in parallel
    for x in range(num_blocks - 1):
        # Issue the load for the next block
        next_start = (x + 1) * BLOCK_SIZE
        mask_next = offsets < (topk_numel - next_start)
        next_ids = tl.load(base_ptr + next_start, mask=mask_next, other=-1)

        # Map the current block (already loaded) and accumulate
        expert_ids = cur_ids
        if HAS_EXPERT_MAP:
            map_mask = expert_ids >= 0
            expert_ids = tl.load(expert_map_ptr + expert_ids, mask=map_mask, other=-1)
        has_curr = (expert_ids == curr_expert).to(tl.int32)
        acc += has_curr

        # Swap buffers: current becomes next for the next iteration
        cur_ids, next_ids = next_ids, cur_ids

    # Process the last block (already loaded into cur_ids after the final swap)
    last_x = num_blocks - 1
    block_start_last = last_x * BLOCK_SIZE
    mask_last = offsets < (topk_numel - block_start_last)
    # cur_ids already holds the last block’s data; mask is not strictly needed for
    # the match test (out‑of‑bounds entries are -1 and never equal to curr_expert)
    expert_ids = cur_ids
    if HAS_EXPERT_MAP:
        map_mask = expert_ids >= 0
        expert_ids = tl.load(expert_map_ptr + expert_ids, mask=map_mask, other=-1)
    has_curr = (expert_ids == curr_expert).to(tl.int32)
    acc += has_curr

    tl.store(expert_num_tokens_ptr + curr_expert, tl.sum(acc))


@triton.jit
def _count_expert_num_tokens_smem(
    topk_ids_ptr,
    expert_num_tokens_ptr,
    num_experts: tl.constexpr,
    topk_numel: tl.constexpr,
    expert_map_ptr,
    MAP_NUMEL: tl.constexpr,   # number of entries in expert_map (must be > 0)
    BLOCK_SIZE: tl.constexpr,
):
    """
    Double‑buffered SMEM kernel: the expert map lives in shared memory,
    and the topk_ids stream is double‑buffered to overlap global loads.
    """
    curr_expert = tl.program_id(0)
    if curr_expert >= num_experts:
        return

    # Load expert_map into shared memory cooperatively
    map_smem = tl.zeros([MAP_NUMEL], dtype=tl.int32)
    offsets = tl.arange(0, BLOCK_SIZE)
    num_map_blocks = tl.cdiv(MAP_NUMEL, BLOCK_SIZE)
    for i in range(num_map_blocks):
        block_start = i * BLOCK_SIZE
        mask = offsets < (MAP_NUMEL - block_start)
        vals = tl.load(expert_map_ptr + block_start + offsets, mask=mask, other=-1)
        tl.store(map_smem + block_start + offsets, vals, mask=mask)

    # Double‑buffered topk_ids counting loop
    num_blocks = tl.cdiv(topk_numel, BLOCK_SIZE)
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.int32)
    base_ptr = topk_ids_ptr + offsets

    cur_ids = tl.zeros((BLOCK_SIZE,), dtype=tl.int32)
    next_ids = tl.zeros((BLOCK_SIZE,), dtype=tl.int32)

    mask0 = offsets < topk_numel
    cur_ids = tl.load(base_ptr, mask=mask0, other=-1)

    for x in range(num_blocks - 1):
        next_start = (x + 1) * BLOCK_SIZE
        mask_next = offsets < (topk_numel - next_start)
        next_ids = tl.load(base_ptr + next_start, mask=mask_next, other=-1)

        # Map lookup from shared memory
        expert_ids = cur_ids
        map_mask = expert_ids >= 0
        mapped = tl.load(map_smem + expert_ids, mask=map_mask, other=-1)
        has_curr = (mapped == curr_expert).to(tl.int32)
        acc += has_curr

        cur_ids, next_ids = next_ids, cur_ids

    # Last block
    last_x = num_blocks - 1
    expert_ids = cur_ids
    map_mask = expert_ids >= 0
    mapped = tl.load(map_smem + expert_ids, mask=map_mask, other=-1)
    has_curr = (mapped == curr_expert).to(tl.int32)
    acc += has_curr

    tl.store(expert_num_tokens_ptr + curr_expert, tl.sum(acc))


@triton.jit
def _count_expert_num_tokens_arithmetic_offset(
    topk_ids_ptr,
    expert_num_tokens_ptr,
    num_experts: tl.constexpr,
    topk_numel: tl.constexpr,
    OFFSET: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Double‑buffered arithmetic kernel – no map memory accesses,
    only global loads for topk_ids are hidden.
    """
    curr_expert = tl.program_id(0)
    if curr_expert >= num_experts:
        return

    offsets = tl.arange(0, BLOCK_SIZE)
    num_blocks = tl.cdiv(topk_numel, BLOCK_SIZE)
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.int32)
    base_ptr = topk_ids_ptr + offsets

    cur_ids = tl.zeros((BLOCK_SIZE,), dtype=tl.int32)
    next_ids = tl.zeros((BLOCK_SIZE,), dtype=tl.int32)

    mask0 = offsets < topk_numel
    cur_ids = tl.load(base_ptr, mask=mask0, other=-1)

    for x in range(num_blocks - 1):
        next_start = (x + 1) * BLOCK_SIZE
        mask_next = offsets < (topk_numel - next_start)
        next_ids = tl.load(base_ptr + next_start, mask=mask_next, other=-1)

        # Arithmetic mapping: local_id = global_id + OFFSET
        expert_ids = cur_ids
        diff = (expert_ids + OFFSET) - curr_expert
        candidate = (diff == 0)
        valid = expert_ids >= 0
        has_curr = (candidate & valid).to(tl.int32)
        acc += has_curr

        cur_ids, next_ids = next_ids, cur_ids

    # Last block
    last_x = num_blocks - 1
    expert_ids = cur_ids
    diff = (expert_ids + OFFSET) - curr_expert
    candidate = (diff == 0)
    valid = expert_ids >= 0
    has_curr = (candidate & valid).to(tl.int32)
    acc += has_curr

    tl.store(expert_num_tokens_ptr + curr_expert, tl.sum(acc))


def count_expert_num_tokens(
    topk_ids: torch.Tensor, num_local_experts: int, expert_map: torch.Tensor | None
) -> torch.Tensor:
    """
    Count the number of tokens assigned to each local expert.
    If expert_map is provided, it maps global expert ids to local expert ids.
    When the mapping is a trivial arithmetic function (identity or constant offset),
    a specialised kernel without memory lookups is launched.
    Otherwise fallback to the cached or global‑lookup kernels.
    Dynamically selects BLOCK_SIZE and num_warps for improved occupancy.
    All kernel variants use double‑buffering on the topk_ids stream.
    """
    assert topk_ids.dtype.is_signed, "The kernel uses -1 to represent invalid topk_ids"
    expert_num_tokens = torch.empty(
        (num_local_experts,), device=topk_ids.device, dtype=torch.int32
    )

    grid = num_local_experts
    topk_numel = topk_ids.numel()
    BLOCK_SIZE = min(topk_numel, 512)
    BLOCK_SIZE = triton.next_power_of_2(BLOCK_SIZE)

    num_warps = 4 if BLOCK_SIZE > 64 else 2

    if expert_map is not None:
        map_numel = expert_map.numel()
        if map_numel > 0:
            # Detect simple arithmetic mapping (identity or constant offset)
            indices = torch.arange(map_numel, device=expert_map.device, dtype=torch.int32)
            diff = expert_map - indices
            if torch.all(diff == diff[0]):          # constant offset
                offset = diff[0].item()
                if offset == 0:
                    # Identity mapping – treat as no map at all
                    _count_expert_num_tokens[(grid,)](
                        topk_ids, expert_num_tokens,
                        num_local_experts, topk_numel,
                        None, HAS_EXPERT_MAP=False, BLOCK_SIZE=BLOCK_SIZE,
                        num_warps=num_warps, num_stages=2)
                    return expert_num_tokens
                else:
                    _count_expert_num_tokens_arithmetic_offset[(grid,)](
                        topk_ids, expert_num_tokens,
                        num_local_experts, topk_numel,
                        OFFSET=offset, BLOCK_SIZE=BLOCK_SIZE,
                        num_warps=num_warps, num_stages=2)
                    return expert_num_tokens

        # Fallback: use SMEM if map fits, otherwise global loads
        MAX_SMEM_MAP_SIZE = 4096
        if 0 < map_numel <= MAX_SMEM_MAP_SIZE:
            _count_expert_num_tokens_smem[(grid,)](
                topk_ids, expert_num_tokens,
                num_local_experts, topk_numel,
                expert_map, MAP_NUMEL=map_numel,
                BLOCK_SIZE=BLOCK_SIZE,
                num_warps=num_warps, num_stages=2)
            return expert_num_tokens

    # No map, or map too large for SMEM / identity already handled
    _count_expert_num_tokens[(grid,)](
        topk_ids, expert_num_tokens,
        num_local_experts, topk_numel,
        expert_map if expert_map is not None else None,
        HAS_EXPERT_MAP=expert_map is not None,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps, num_stages=2)

    return expert_num_tokens