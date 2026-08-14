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
    base_ptr = topk_ids_ptr + offsets

    for x in range(num_blocks):
        block_start = x * BLOCK_SIZE
        mask = offsets < (topk_numel - block_start)
        expert_ids = tl.load(base_ptr + block_start, mask=mask, other=-1)

        if HAS_EXPERT_MAP:
            map_mask = expert_ids >= 0
            expert_ids = tl.load(expert_map_ptr + expert_ids, mask=map_mask, other=-1)

        has_curr_expert = tl.where(expert_ids == curr_expert, 1, 0)
        acc = acc + has_curr_expert

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

    _count_expert_num_tokens[(grid,)](
        topk_ids,
        expert_num_tokens,
        num_local_experts,
        topk_ids.numel(),
        expert_map,
        HAS_EXPERT_MAP=expert_map is not None,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return expert_num_tokens