import torch
import triton
import triton.language as tl

K_THRESHOLD = 128


@triton.jit
def _per_group_transpose(
    data_ptr: torch.Tensor,
    trans_data_ptr: torch.Tensor,
    expert_offsets: torch.Tensor,
    k: int,
    M_ALIGNMENT: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    LOOP_K_INTERNAL: tl.constexpr,
):
    expert_id = tl.program_id(0)
    m_id = tl.program_id(1)
    if not LOOP_K_INTERNAL:
        k_id = tl.program_id(2)

    # Expert bounds (small, reused by every CTA)
    curr_expert_offset = tl.load(expert_offsets + expert_id, cache_modifier=".ca")
    next_expert_offset = tl.load(expert_offsets + expert_id + 1, cache_modifier=".ca")
    num_tokens_of_expert = next_expert_offset - curr_expert_offset

    data_start_ptr = data_ptr + curr_expert_offset * k
    trans_data_start_ptr = trans_data_ptr + curr_expert_offset * k

    num_m_programs = tl.num_programs(1)
    # When looping internally, we walk over all K tiles; otherwise the grid
    # tile index is taken from program_id(2).
    num_k_tiles = tl.cdiv(k, BLOCK_SIZE_K) if LOOP_K_INTERNAL else 1  # dummy

    for start_m in tl.range(0, num_tokens_of_expert, BLOCK_SIZE_M * num_m_programs):
        m_coord = start_m + m_id * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        m_mask = m_coord < num_tokens_of_expert

        if LOOP_K_INTERNAL:
            # Serialise over K tiles to eliminate the K‑program grid dimension
            for k_tile in tl.range(num_k_tiles):
                k_coord = k_tile * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
                k_mask = k_coord < k

                off = m_coord[:, None] * k + k_coord[None, :]
                mask = m_mask[:, None] & k_mask[None, :]
                data = tl.load(data_start_ptr + off, mask=mask)

                # Transpose in registers -> coalesced store
                data_t = tl.trans(data)
                trans_off = k_coord[:, None] * num_tokens_of_expert + m_coord[None, :]
                trans_mask = k_mask[:, None] & m_mask[None, :]
                tl.store(trans_data_start_ptr + trans_off, data_t, mask=trans_mask)
        else:
            # Original path: one K tile per CTA, same semantics
            k_coord = k_id * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
            k_mask = k_coord < k

            off = m_coord[:, None] * k + k_coord[None, :]
            mask = m_mask[:, None] & k_mask[None, :]
            data = tl.load(data_start_ptr + off, mask=mask)

            data_t = tl.trans(data)
            trans_off = k_coord[:, None] * num_tokens_of_expert + m_coord[None, :]
            trans_mask = k_mask[:, None] & m_mask[None, :]
            tl.store(trans_data_start_ptr + trans_off, data_t, mask=trans_mask)


def per_group_transpose(
    a: torch.Tensor,
    expert_offsets: torch.Tensor,
    M_ALIGNMENT: int = 1,
) -> torch.Tensor:
    assert a.dim() == 2
    assert a.is_contiguous(), "`a` is not contiguous"

    m, k = a.size()
    trans_a = torch.empty_like(a)
    num_experts = expert_offsets.size(0) - 1

    # Fixed M‑program count per expert, tuned for occupancy
    num_m_blocks = 4
    BLOCK_SIZE_K = 32

    # Choose the launch strategy based on K
    LOOP_K_INTERNAL = k <= K_THRESHOLD

    if LOOP_K_INTERNAL:
        # 2‑D grid (expert, M programs) – kernel serialises over K internally
        grid = lambda META: (num_experts, num_m_blocks, 1)
    else:
        # 3‑D grid (expert, M programs, K programs) – parallelised over K
        grid = lambda META: (num_experts, num_m_blocks, triton.cdiv(k, BLOCK_SIZE_K))

    _per_group_transpose[grid](
        a,
        trans_a,
        expert_offsets,
        k,
        M_ALIGNMENT,
        BLOCK_SIZE_M=64,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        num_warps=4,
        LOOP_K_INTERNAL=LOOP_K_INTERNAL,
    )
    return trans_a