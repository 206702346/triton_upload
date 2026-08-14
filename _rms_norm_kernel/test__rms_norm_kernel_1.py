import torch
import torch_npu
import triton
import triton.language as tl
from _rms_norm_kernel import rms_norm

def test_rms_norm():
    """Test RMS normalization implementation against PyTorch reference."""
    torch.manual_seed(0)
    
    # Test parameters
    batch_size = 32
    seq_len = 128
    hidden_size = 512
    eps = 1e-6
    
    # Create test data on NPU
    input_tensor = torch.randn(batch_size, seq_len, hidden_size, device='npu', dtype=torch.float32)
    weight_tensor = torch.randn(hidden_size, device='npu', dtype=torch.float32)
    
    # Compute using Triton implementation
    for _ in range(15):
        triton_output = rms_norm(input_tensor, weight_tensor, eps)
    
    # Compute reference using PyTorch
    # RMS Norm: y = x / sqrt(mean(x^2) + eps) * weight
    variance = torch.mean(input_tensor ** 2, dim=-1, keepdim=True)
    rms = torch.sqrt(variance + eps)
    torch_output = (input_tensor / rms) * weight_tensor
    
    # Validate correctness
    assert torch.allclose(triton_output, torch_output, atol=1e-5, rtol=1e-4), \
        f"Max difference: {torch.max(torch.abs(triton_output - torch_output))}"
    
    print("✅ RMS Norm test passed!")
    print(f"Input shape: {input_tensor.shape}")
    print(f"Max absolute difference: {torch.max(torch.abs(triton_output - torch_output))}")
    print(f"Mean absolute difference: {torch.mean(torch.abs(triton_output - torch_output))}")

if __name__ == "__main__":
    # Run tests
    test_rms_norm()
    # test_rms_norm_edge_cases()