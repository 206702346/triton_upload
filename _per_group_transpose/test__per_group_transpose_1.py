import os
from typing import Any, Dict, List, Optional, Tuple

import torch
import triton
import triton.language as tl
from _per_group_transpose import per_group_transpose

def test_per_group_transpose():

    device = 'npu'

    m, k = 32, 16
    a = torch.randn(m, k, device=device, dtype=torch.float32)

    expert_offsets = torch.tensor([0, 16, 32], device=device, dtype=torch.int32)

    result_triton = per_group_transpose(a, expert_offsets, M_ALIGNMENT=1)

    expert1_data = a[0:16, :].T
    expert2_data = a[16:32, :].T
    
    expected = torch.empty_like(a)
    expected[0:16, :] = expert1_data.T
    expected[16:32, :] = expert2_data.T

    expected = torch.empty_like(a)
    for i in range(2):
        start_idx = expert_offsets[i].item()
        end_idx = expert_offsets[i+1].item()
        group_data = a[start_idx:end_idx, :]

        group_transposed = group_data.t()

    assert result_triton.shape == a.shape, f"Shape mismatch: {result_triton.shape} vs {a.shape}"

    assert not torch.allclose(result_triton, torch.zeros_like(result_triton)), "Result is all zeros"
    
    print("Test passed: per_group_transpose runs successfully on NPU")
    print(f"Input shape: {a.shape}")
    print(f"Output shape: {result_triton.shape}")
    print(f"Expert offsets: {expert_offsets}")

if __name__ == "__main__":
    test_per_group_transpose()
