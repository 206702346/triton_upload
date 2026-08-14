import torch
import triton
import triton.language as tl


@triton.jit
def _per_group_transpose(
    data_ptr: torch.Tensor,
    trans_data_ptr: torch.Tensor,
    expert_offsets: torch.Tensor,
    k: int,
    M_ALIGNMENT: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    expert_id = tl.program_id(0)
    m_id = tl.program_id(1)
    k_id = tl.program_id(2)

    # Expert bounds are small and reused by every CTA working on this expert.
    curr_expert_offset = tl.load(expert_offsets + expert_id, cache_modifier=".ca")
    next_expert_offset = tl.load(expert_offsets + expert_id + 1, cache_modifier=".ca")
    num_tokens_of_expert = next_expert_offset - curr_expert_offset

    data_start_ptr = data_ptr + curr_expert_offset * k
    trans_data_start_ptr = trans_data_ptr + curr_expert_offset * k

    k_coord = k_id * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    k_mask = k_coord < k

    # Fixed number of M-program CTA's per expert. Each CTA walks its share
    # of the M dimension with a strided loop, keeping occupancy independent
    # of the number of tokens in a group.
    num_m_programs = tl.num_programs(1)
    for start_m in tl.range(0, num_tokens_of_expert, BLOCK_SIZE_M * num_m_programs):
        m_coord = start_m + m_id * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        m_mask = m_coord < num_tokens_of_expert

        # Load in the natural row-major layout of `data`: contiguous along K.
        off = m_coord[:, None] * k + k_coord[None, :]
        mask = m_mask[:, None] & k_mask[None, :]
        data = tl.load(data_start_ptr + off, mask=mask)

        # `data` is [BLOCK_SIZE_M, BLOCK_SIZE_K]. The output linear index for
        # the transposed group is `m + k * num_tokens`, i.e. contiguous along M.
        # Transpose in registers so the store is also fully coalesced.
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

    # Fixed M-program count per expert. Combined with register-level
    # transposition this gives coalesced loads and stores for every expert.
    num_m_blocks = 4
    grid = lambda META: (
        num_experts,
        num_m_blocks,
        triton.cdiv(k, META["BLOCK_SIZE_K"]),
    )
    _per_group_transpose[grid](
        a,
        trans_a,
        expert_offsets,
        k,
        M_ALIGNMENT,
        BLOCK_SIZE_M=64,
        BLOCK_SIZE_K=32,
        num_warps=4,
    )
    return trans_a