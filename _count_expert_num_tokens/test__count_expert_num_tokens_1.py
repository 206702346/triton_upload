import torch
import triton
import triton.language as tl
from _count_expert_num_tokens import count_expert_num_tokens

def test_count_expert_num_tokens():

    torch.manual_seed(0)
    device = 'npu'

    topk_ids = torch.tensor([0, 1, 2, 0, 1, 3, 2, 0], dtype=torch.int32, device=device)
    num_local_experts = 4
    expert_map = None

    result_triton = count_expert_num_tokens(topk_ids, num_local_experts, expert_map)

    result_ref = torch.zeros(num_local_experts, dtype=torch.int32, device=device)
    for expert_id in topk_ids:
        if expert_id >= 0:
            result_ref[expert_id] += 1

    assert torch.allclose(result_triton, result_ref), f"Results don't match: Triton={result_triton}, Ref={result_ref}"
    print("Test 1 passed: Basic functionality without expert map")

    expert_map = torch.tensor([1, 0, 3, 2], dtype=torch.int32, device=device)
    result_triton_2 = count_expert_num_tokens(topk_ids, num_local_experts, expert_map)

    result_ref_2 = torch.zeros(num_local_experts, dtype=torch.int32, device=device)
    for expert_id in topk_ids:
        if expert_id >= 0:
            local_expert_id = expert_map[expert_id]
            result_ref_2[local_expert_id] += 1

    assert torch.allclose(result_triton_2, result_ref_2), f"Results don't match with expert map: Triton={result_triton_2}, Ref={result_ref_2}"
    print("Test 2 passed: With expert map")

    large_topk_ids = torch.randint(0, num_local_experts, (1000,), dtype=torch.int32, device=device)
    result_triton_3 = count_expert_num_tokens(large_topk_ids, num_local_experts, None)

    result_ref_3 = torch.zeros(num_local_experts, dtype=torch.int32, device=device)
    for expert_id in large_topk_ids:
        if expert_id >= 0:
            result_ref_3[expert_id] += 1

    assert torch.allclose(result_triton_3, result_ref_3), "Large tensor test failed"
    print("Test 3 passed: Larger random tensor")
    
    print("All tests passed!")

if __name__ == "__main__":
    test_count_expert_num_tokens()
