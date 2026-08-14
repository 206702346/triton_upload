import torch
import triton
import triton.language as tl

@triton.jit
def _count_expert_num_tokens(
    topk_ids_ptr,
    expert_num_tokens_ptr,
    num_experts,
    topk_numel,
    expert_map,
    HAS_EXPERT_MAP: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
):

    expert_id = tl.program_id(0)
    chunk_id = tl.program_id(1)

    if expert_id >= num_experts:
        return

    chunk_start = chunk_id * CHUNK_SIZE
    chunk_end = tl.minimum(chunk_start + CHUNK_SIZE, topk_numel)

    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.int32)

    for block_offset in range(0, CHUNK_SIZE, BLOCK_SIZE):
        current_offset = chunk_start + block_offset

        continue_processing = current_offset < topk_numel
        if continue_processing:

            remaining = tl.minimum(BLOCK_SIZE, topk_numel - current_offset)
            offsets = tl.arange(0, BLOCK_SIZE)
            mask = offsets < remaining

            expert_ids = tl.load(topk_ids_ptr + current_offset + offsets, mask=mask, other=-1)

            if HAS_EXPERT_MAP:

                expert_map_mask = expert_ids >= 0
                expert_map_ptrs = expert_map + expert_ids
                mapped_experts = tl.load(expert_map_ptrs, mask=expert_map_mask, other=-1)
                expert_ids = tl.where(expert_map_mask, mapped_experts, expert_ids)

            has_curr_expert = tl.where(expert_ids == expert_id, 1, 0)
            acc = acc + has_curr_expert

    if chunk_id == 0:
        expert_count = tl.sum(acc)
        tl.store(expert_num_tokens_ptr + expert_id, expert_count)

def count_expert_num_tokens(
    topk_ids: torch.Tensor, num_local_experts: int, expert_map: torch.Tensor | None
) -> torch.Tensor:
    """
    Count the number to tokens assigned to each expert.

    Parameters:
    - topk_ids (torch.Tensor): Tensor mapping each token to its
    list of experts.
    - num_local_experts (int): Number of experts in this rank.
    - expert_map (Optional[torch.Tensor]):  A tensor mapping expert indices
    from the global expert space to the local expert space of the expert
    parallel shard.

    Returns:
    A tensor of size num_local_experts, where tensor[i] holds the number
    of tokens assigned to the ith expert.
    """
    assert topk_ids.dtype.is_signed, "The kernel uses -1 to represent invalid topk_ids"
    expert_num_tokens = torch.zeros(
        (num_local_experts), device=topk_ids.device, dtype=torch.int32
    )

    try:
        import triton.runtime.driver as driver
        npu_props = driver.active.utils.get_device_properties(torch.npu.current_device())
        num_cores = npu_props.get("num_aicore", 24)
    except:
        num_cores = 24

    BLOCK_SIZE = 256
    CHUNK_SIZE = 1024
    
    num_chunks = triton.cdiv(topk_ids.numel(), CHUNK_SIZE)
    grid = (num_local_experts, num_chunks)

    if grid[0] * grid[1] > num_cores * 4:
        CHUNK_SIZE = max(256, CHUNK_SIZE // 2)
        num_chunks = triton.cdiv(topk_ids.numel(), CHUNK_SIZE)
        grid = (num_local_experts, num_chunks)

    _count_expert_num_tokens[grid](
        topk_ids,
        expert_num_tokens,
        num_local_experts,
        topk_ids.numel(),
        expert_map,
        HAS_EXPERT_MAP=expert_map is not None,
        BLOCK_SIZE=BLOCK_SIZE,
        CHUNK_SIZE=CHUNK_SIZE,
    )

    return expert_num_tokens