from typing import Any, Dict, List, Optional, Tuple

import torch
import triton
import triton.language as tl

from _per_token_group_quant_8bit_colmajor import *

def test_per_token_group_quant_8bit_colmajor():
    """Test the per-token-group quantization function"""
    # Create test data
    torch.manual_seed(0)
    shape = (16, 128)
    y = torch.randn(shape, device='npu', dtype=torch.float32)
    
    group_size = 32
    eps = 1e-5
    
    # Test without UE8M0 scaling
    y_q, y_s = per_token_group_quant_8bit_colmajor(y, group_size, eps, False)
    
    # Reference implementation
    y_ref = y.view(-1, group_size)
    absmax_ref = torch.maximum(torch.max(torch.abs(y_ref), dim=1, keepdim=True).values, torch.tensor(eps))
    y_s_ref = absmax_ref / 127.0
    y_q_ref = torch.clamp(y_ref / y_s_ref, -128.0, 127.0).round().to(torch.int8)
    y_q_ref = y_q_ref.view(shape)
    
    # Transpose y_s_ref to match col-major storage
    M, N = y.shape
    num_groups_per_row = (N + group_size - 1) // group_size
    y_s_ref_reshaped = y_s_ref.view(M, num_groups_per_row).t()
    
    # Check results
    assert torch.allclose(y_q.cpu(), y_q_ref.cpu(), atol=1), f"Quantized values don't match: max diff = {torch.max(torch.abs(y_q.cpu() - y_q_ref.cpu()))}"
    assert torch.allclose(y_s.cpu(), y_s_ref_reshaped.cpu(), atol=1e-3), f"Scales don't match: max diff = {torch.max(torch.abs(y_s.cpu() - y_s_ref_reshaped.cpu()))}"
    
    print("✅ Per-token-group quantization test passed!")
    print(f"Input shape: {shape}")
    print(f"Group size: {group_size}")
    print(f"Max quantized difference: {torch.max(torch.abs(y_q.cpu() - y_q_ref.cpu()))}")
    print(f"Max scale difference: {torch.max(torch.abs(y_s.cpu() - y_s_ref_reshaped.cpu()))}")

if __name__ == "__main__":
    test_per_token_group_quant_8bit_colmajor()
