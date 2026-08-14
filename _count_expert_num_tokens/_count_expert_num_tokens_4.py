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
    # One program per (expert, token-block).  This replaces the parent's
    # one-program-per-expert full scan with a more parallel launch geometry.
    curr_expert = tl.program_id(0)
    block_id = tl.program_id(1)

    if curr_expert >= num_experts:
        return

    offsets = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < topk_numel
    expert_ids = tl.load(topk_ids_ptr + offsets, mask=mask, other=-1)

    if HAS_EXPERT_MAP:
        map_mask = expert_ids >= 0
        expert_ids = tl.load(expert_map_ptr + expert_ids, mask=map_mask, other=-1)

    has_curr = tl.where(expert_ids == curr_expert, 1, 0)
    partial = tl.sum(has_curr)

    if topk_numel <= BLOCK_SIZE:
        tl.store(expert_num_tokens_ptr + curr_expert, partial)
    else:
        if partial > 0:
            tl.atomic_add(expert_num_tokens_ptr + curr_expert, partial)


def count_expert_num_tokens(
    topk_ids: torch.Tensor, num_local_experts: int, expert_map: torch.Tensor | None
) -> torch.Tensor:
    assert topk_ids.dtype.is_signed, "The kernel uses -1 to represent invalid topk_ids"

    expert_num_tokens = torch.zeros(
        (num_local_experts), device=topk_ids.device, dtype=torch.int32
    )

    num_tokens = topk_ids.numel()
    if num_local_experts == 0 or num_tokens == 0:
        return expert_num_tokens

    BLOCK_SIZE = min(num_tokens, 1024)
    BLOCK_SIZE = triton.next_power_of_2(BLOCK_SIZE)
    num_blocks = triton.cdiv(num_tokens, BLOCK_SIZE)

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

    grid = (num_local_experts, num_blocks)
    _count_expert_num_tokens[grid](
        topk_ids,
        expert_num_tokens,
        num_local_experts,
        num_tokens,
        expert_map,
        HAS_EXPERT_MAP=expert_map is not None,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
        num_stages=num_stages,
        num_ctas=num_ctas,
    )

    return expert_num_tokens