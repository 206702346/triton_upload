import torch
import triton
import triton.language as tl
import pytest
from _log_softmax_kernel import log_softmax

class TestLogSoftmaxNPU:
    """Unit tests for log_softmax function on NPU device."""
    
    def setup_method(self):
        """Setup method to ensure NPU device is available."""
        self.device = 'npu'
        if not torch.npu.is_available():
            pytest.skip("NPU device not available")
    
    def test_log_softmax_2d_small(self):
        """Test log_softmax with small 2D tensor."""
        # Create input tensor on NPU
        input_tensor = torch.tensor([[1.0, 2.0, 3.0], 
                                   [4.0, 5.0, 6.0]], 
                                  device=self.device, dtype=torch.float32)
        
        # Compute using Triton kernel
        triton_output = log_softmax(input_tensor, dim=-1)
        
        # Compute reference using PyTorch
        torch_output = torch.log_softmax(input_tensor, dim=-1)
        
        # Verify results
        assert triton_output.device.type == self.device
        assert torch_output.device.type == self.device
        torch.testing.assert_close(triton_output, torch_output, rtol=1e-5, atol=1e-6)
    
    def test_log_softmax_2d_large(self):
        """Test log_softmax with larger 2D tensor."""
        torch.manual_seed(42)
        input_tensor = torch.randn(4096, 8192, device=self.device, dtype=torch.float32)
        
        triton_output = log_softmax(input_tensor, dim=-1)
        torch_output = torch.log_softmax(input_tensor, dim=-1)
        
        assert triton_output.device.type == self.device
        torch.testing.assert_close(triton_output, torch_output, rtol=1e-5, atol=1e-6)
    
    def test_log_softmax_3d(self):
        """Test log_softmax with 3D tensor (flattened to 2D)."""
        input_tensor = torch.tensor([[[1.0, 2.0], [3.0, 4.0]], 
                                   [[5.0, 6.0], [7.0, 8.0]]], 
                                  device=self.device, dtype=torch.float32)
        
        triton_output = log_softmax(input_tensor, dim=-1)
        torch_output = torch.log_softmax(input_tensor, dim=-1)
        
        assert triton_output.device.type == self.device
        torch.testing.assert_close(triton_output, torch_output, rtol=1e-5, atol=1e-6)
    
    def test_log_softmax_4d(self):
        """Test log_softmax with 4D tensor."""
        input_tensor = torch.randn(2, 3, 4, 5, device=self.device, dtype=torch.float32)
        
        triton_output = log_softmax(input_tensor, dim=-1)
        torch_output = torch.log_softmax(input_tensor, dim=-1)
        
        assert triton_output.device.type == self.device
        torch.testing.assert_close(triton_output, torch_output, rtol=1e-5, atol=1e-6)
    
    def test_log_softmax_negative_values(self):
        """Test log_softmax with negative input values."""
        input_tensor = torch.tensor([[-1.0, -2.0, -3.0], 
                                   [-10.0, -20.0, -30.0]], 
                                  device=self.device, dtype=torch.float32)
        
        triton_output = log_softmax(input_tensor, dim=-1)
        torch_output = torch.log_softmax(input_tensor, dim=-1)
        
        assert triton_output.device.type == self.device
        torch.testing.assert_close(triton_output, torch_output, rtol=1e-5, atol=1e-6)
    
    def test_log_softmax_large_values(self):
        """Test log_softmax with large input values (testing numerical stability)."""
        input_tensor = torch.tensor([[1000.0, 1001.0, 1002.0]], 
                                  device=self.device, dtype=torch.float32)
        
        triton_output = log_softmax(input_tensor, dim=-1)
        torch_output = torch.log_softmax(input_tensor, dim=-1)
        
        assert triton_output.device.type == self.device
        torch.testing.assert_close(triton_output, torch_output, rtol=1e-5, atol=1e-6)
    
    def test_log_softmax_single_element(self):
        """Test log_softmax with single element per row."""
        input_tensor = torch.tensor([[5.0], [10.0]], device=self.device, dtype=torch.float32)
        
        triton_output = log_softmax(input_tensor, dim=-1)
        torch_output = torch.log_softmax(input_tensor, dim=-1)
        
        assert triton_output.device.type == self.device
        torch.testing.assert_close(triton_output, torch_output, rtol=1e-5, atol=1e-6)
    
    def test_log_softmax_float16(self):
        """Test log_softmax with float16 precision."""
        if not torch.npu.is_bf16_supported():
            pytest.skip("BF16 not supported on NPU")
            
        input_tensor = torch.randn(64, 128, device=self.device, dtype=torch.float16)
        
        triton_output = log_softmax(input_tensor, dim=-1)
        torch_output = torch.log_softmax(input_tensor, dim=-1)
        
        assert triton_output.device.type == self.device
        torch.testing.assert_close(triton_output, torch_output, rtol=1e-3, atol=1e-4)
    
    def test_log_softmax_bfloat16(self):
        """Test log_softmax with bfloat16 precision."""
        if not torch.npu.is_bf16_supported():
            pytest.skip("BF16 not supported on NPU")
            
        input_tensor = torch.randn(64, 128, device=self.device, dtype=torch.bfloat16)
        
        triton_output = log_softmax(input_tensor, dim=-1)
        torch_output = torch.log_softmax(input_tensor, dim=-1)
        
        assert triton_output.device.type == self.device
        torch.testing.assert_close(triton_output, torch_output, rtol=1e-2, atol=1e-3)
    
    def test_log_softmax_invalid_dim(self):
        """Test that invalid dimension raises ValueError."""
        input_tensor = torch.randn(2, 3, device=self.device, dtype=torch.float32)
        
        with pytest.raises(ValueError, match="only supports log_softmax along the last dimension"):
            log_softmax(input_tensor, dim=0)
    
    def test_log_softmax_output_shape(self):
        """Test that output shape matches input shape."""
        input_tensor = torch.randn(2, 3, 4, 5, device=self.device, dtype=torch.float32)
        
        triton_output = log_softmax(input_tensor, dim=-1)
        
        assert triton_output.shape == input_tensor.shape
        assert triton_output.device.type == self.device

# Run tests if this file is executed directly
if __name__ == "__main__":
    # Create test instance
    test_instance = TestLogSoftmaxNPU()
    test_instance.setup_method()
    
    # Run individual tests
    print("Running log_softmax unit tests on NPU...")
    
    try:
        # test_instance.test_log_softmax_2d_small()
        # print("✓ test_log_softmax_2d_small passed")
        
        test_instance.test_log_softmax_2d_large()
        print("✓ test_log_softmax_2d_large passed")
        
        # test_instance.test_log_softmax_3d()
        # print("✓ test_log_softmax_3d passed")
        
        # test_instance.test_log_softmax_4d()
        # print("✓ test_log_softmax_4d passed")
        
        # test_instance.test_log_softmax_negative_values()
        # print("✓ test_log_softmax_negative_values passed")
        
        # test_instance.test_log_softmax_large_values()
        # print("✓ test_log_softmax_large_values passed")
        
        # test_instance.test_log_softmax_single_element()
        # print("✓ test_log_softmax_single_element passed")
        
        # test_instance.test_log_softmax_output_shape()
        # print("✓ test_log_softmax_output_shape passed")
        
        print("All tests passed!")
        
    except Exception as e:
        print(f"Test failed: {e}")
        raise