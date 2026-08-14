import torch
import triton
import triton.language as tl

from _quantize_k_cache_fast_kernel import _quantize_k_cache_fast

# Test code
if __name__ == "__main__":
    # Create test tensors on NPU
    device = torch.device('npu')
    
    # Test with sample data
    num_tokens = 4
    dim_nope = 512
    dim_rope = 64
    group_size = 128
    
    # Create input tensors
    k_nope = torch.randn(num_tokens, dim_nope, dtype=torch.bfloat16, device=device)
    k_rope = torch.randn(num_tokens, dim_rope, dtype=torch.bfloat16, device=device)
    
    # Run the quantization function
    output = _quantize_k_cache_fast(k_nope, k_rope, group_size)
    
    # Verify output properties
    expected_output_size = dim_nope + (dim_nope // group_size) * 4 + dim_rope
    assert output.shape == (num_tokens, expected_output_size)
    assert output.device.type == 'npu'
    assert output.dtype == torch.bfloat16
    
    # Extract components
    output_nope_q = output[..., :dim_nope]
    output_nope_s = output[..., dim_nope : dim_nope + (dim_nope // group_size) * 4].view(torch.bfloat16)
    output_rope = output[..., dim_nope + (dim_nope // group_size) * 4 :].view(torch.bfloat16)
    
    # Verify rope part matches input
    assert torch.allclose(output_rope, k_rope, atol=1e-3)
    
    print("All tests passed!")
    print(f"Output shape: {output.shape}")
    print(f"Output device: {output.device}")
    print(f"Output dtype: {output.dtype}")