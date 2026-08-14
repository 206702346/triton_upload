import torch
import triton
import pytest

from _fwd_kernel_ep_gather import ep_gather

def test_ep_gather():

    device = torch.device('npu')

    torch.manual_seed(0)

    num_tokens = 128
    hidden_size = 512
    topk = 2

    input_tensor = torch.randn((num_tokens, hidden_size), device=device)
    recv_topk_ids = torch.randint(0, num_tokens, (num_tokens, topk), device=device)
    recv_topk_weight = torch.randn((num_tokens, topk), device=device)
    input_index = torch.randint(0, num_tokens, (num_tokens, topk), device=device)
    expert_map = None
    output_tensor = torch.zeros((num_tokens, hidden_size), device=device)

    ep_gather(input_tensor, recv_topk_ids, recv_topk_weight, input_index, expert_map, output_tensor)

    reference_output = torch.zeros((num_tokens, hidden_size), device=device)
    for i in range(num_tokens):
        for k in range(topk):
            expert_id = recv_topk_ids[i, k]
            if expert_id >= 0:
                source_token_index = input_index[i, k]
                weight = recv_topk_weight[i, k]
                reference_output[i] += input_tensor[source_token_index] * weight

    assert torch.allclose(output_tensor, reference_output, atol=1e-5), "Output does not match reference"

if __name__ == "__main__":
    test_ep_gather()
    print("Test passed!")
