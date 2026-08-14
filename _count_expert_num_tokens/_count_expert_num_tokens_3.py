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
    curr_expert = tl.program_id(0)

    if curr_expert >= num_experts:
        return

    offsets = tl.arange(0, BLOCK_SIZE)
    num_blocks = tl.cdiv(topk_numel, BLOCK_SIZE)
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.int32)

    if num_blocks > 1:
        # Double-buffering: load next tile while processing current tile
        # ----------- Load first tile -----------
        block_start = 0
        mask0 = offsets < topk_numel  # first tile may be partial only if topk_numel < BLOCK_SIZE
        expert_ids0 = tl.load(topk_ids_ptr + offsets + block_start, mask=mask0, other=-1)

        for x in range(1, num_blocks):
            # ----------- Load next tile -----------
            next_block_start = x * BLOCK_SIZE
            mask1 = offsets < (topk_numel - next_block_start)
            expert_ids1 = tl.load(topk_ids_ptr + offsets + next_block_start, mask=mask1, other=-1)

            # ----------- Process current tile (expert_ids0) -----------
            if HAS_EXPERT_MAP:
                map_mask0 = expert_ids0 >= 0
                mapped0 = tl.load(expert_map_ptr + expert_ids0, mask=map_mask0, other=-1)
            else:
                mapped0 = expert_ids0
            has_curr = tl.where(mapped0 == curr_expert, 1, 0)
            acc += has_curr

            # ----------- Swap tiles -----------
            expert_ids0 = expert_ids1

        # ----------- Process last tile (expert_ids0 after loop) -----------
        if HAS_EXPERT_MAP:
            map_mask0 = expert_ids0 >= 0
            mapped0 = tl.load(expert_map_ptr + expert_ids0, mask=map_mask0, other=-1)
        else:
            mapped0 = expert_ids0
        has_curr = tl.where(mapped0 == curr_expert, 1, 0)
        acc += has_curr

    else:
        # Single tile or empty input: simple loop
        for x in range(num_blocks):
            block_start = x * BLOCK_SIZE
            mask = offsets < (topk_numel - block_start)
            expert_ids = tl.load(topk_ids_ptr + offsets + block_start, mask=mask, other=-1)
            if HAS_EXPERT_MAP:
                map_mask = expert_ids >= 0
                expert_ids = tl.load(expert_map_ptr + expert_ids, mask=map_mask, other=-1)
            has_curr = tl.where(expert_ids == curr_expert, 1, 0)
            acc += has_curr

    tl.store(expert_num_tokens_ptr + curr_expert, tl.sum(acc))


def count_expert_num_tokens(
    topk_ids: torch.Tensor, num_local_experts: int, expert_map: torch.Tensor | None
) -> torch.Tensor:
    assert topk_ids.dtype.is_signed, "The kernel uses -1 to represent invalid topk_ids"
    expert_num_tokens = torch.empty(
        (num_local_experts), device=topk_ids.device, dtype=torch.int32
    )

    grid = num_local_experts
    BLOCK_SIZE = min(topk_ids.numel(), 1024)
    BLOCK_SIZE = triton.next_power_of_2(BLOCK_SIZE)

    # Tune launch parameters based on BLOCK_SIZE for low latency on Ascend 910B
    if BLOCK_SIZE <= 16:
        num_warps = 1
        num_stages = 1
    elif BLOCK_SIZE <= 64:
        num_warps = 2
        num_stages = 1
    else:
        num_warps = 4
        num_stages = 2
    num_ctas = 1

    _count_expert_num_tokens[(grid,)](
        topk_ids,
        expert_num_tokens,
        num_local_experts,
        topk_ids.numel(),
        expert_map,
        HAS_EXPERT_MAP=expert_map is not None,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
        num_stages=num_stages,
        num_ctas=num_ctas,
    )

    return expert_num_tokens