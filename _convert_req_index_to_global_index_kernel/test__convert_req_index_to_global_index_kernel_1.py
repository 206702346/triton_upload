

import torch
import numpy as np

from _convert_req_index_to_global_index_kernel import triton_convert_req_index_to_global_index

def pytorch_convert_req_index_to_global_index(
    req_id: torch.Tensor,
    block_table: torch.Tensor,
    token_indices: torch.Tensor,
    BLOCK_SIZE: int = 64,
):
    """
    Straightforward implementation without any tiling.
    """
    num_tokens, num_topk = token_indices.shape
    out = torch.full_like(token_indices, -1)

    for t in range(num_tokens):
        r = req_id[t].item()
        for k in range(num_topk):
            pos = token_indices[t, k].item()
            if pos == -1:
                continue
            block_id = pos // BLOCK_SIZE
            if block_id < 0 or block_id >= block_table.shape[1]:
                continue
            base = block_table[r, block_id].item()
            out[t, k] = base * BLOCK_SIZE + (pos % BLOCK_SIZE)
    return out

def test_single_case():
    torch.manual_seed(42)
    device = torch.device("npu")

    NUM_TOKENS = 37
    NUM_REQS   = 5
    MAX_BLOCKS = 11
    BLOCK_SIZE = 64
    NUM_TOPK   = 2048

    req_id = torch.randint(0, NUM_REQS, (NUM_TOKENS,), dtype=torch.int32, device=device)

    block_table = torch.randint(0, 1000, (NUM_REQS, MAX_BLOCKS), dtype=torch.int32, device=device)

    token_indices = torch.randint(-1, MAX_BLOCKS * BLOCK_SIZE + 500,
                                  (NUM_TOKENS, NUM_TOPK),
                                  dtype=torch.int32, device=device)

    out_triton = triton_convert_req_index_to_global_index(
        req_id, block_table, token_indices, BLOCK_SIZE=BLOCK_SIZE, NUM_TOPK_TOKENS=NUM_TOPK)
    out_torch = pytorch_convert_req_index_to_global_index(
        req_id, block_table, token_indices, BLOCK_SIZE=BLOCK_SIZE)

    torch.testing.assert_close(out_triton, out_torch)
    print("✅ Triton and PyTorch results match.")

if __name__ == "__main__":
    test_single_case()

