import torch
from vllm.triton_utils import tl, triton
from typing import Optional

from _silu_mul_fp8_quant_deep_gemm import persistent_masked_m_silu_mul_quant

if __name__ == "__main__":
    device = torch.device("npu")
    torch.npu.set_device(device)
    
    def test_silu_mul_quant():
        """Minimal test for SiLU quantization on NPU"""
        print("Testing persistent_masked_m_silu_mul_quant on NPU...")
        
        # Simple configuration
        E, T, H = 2, 8, 128  # 2 experts, 8 tokens, 128 hidden dim
        
        # Create input: first half is gate, second half is up
        y = torch.randn(E, T, 2 * H, dtype=torch.float16, device=device)
        
        # Create tokens per expert
        tokens_per_expert = torch.tensor([8, 8], dtype=torch.int32, device=device)
        
        # Run quantization
        y_q, y_s = persistent_masked_m_silu_mul_quant(
            y, tokens_per_expert, group_size=128
        )
        
        # Verify output shapes
        assert y_q.shape == (E, T, H), f"y_q shape mismatch: {y_q.shape}"
        assert y_s.shape == (E, T, 1), f"y_s shape mismatch: {y_s.shape}"  # H/128 = 1
        
        # Verify outputs are finite
        assert torch.all(torch.isfinite(y_q)), "y_q contains NaN or Inf"
        assert torch.all(torch.isfinite(y_s)), "y_s contains NaN or Inf"
        
        # Verify scales are positive
        assert torch.all(y_s > 0), "y_s must be positive"
        
        print("✓ SiLU quantization test passed")
    
    test_silu_mul_quant()
    print("✅ NPU quantization kernel test completed successfully!")