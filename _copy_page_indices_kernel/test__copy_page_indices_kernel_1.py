import torch
import triton
import pytest
from _copy_page_indices_kernel import _copy_page_indices_kernel

def test_copy_page_indices_kernel():
    """Test the _copy_page_indices_kernel function."""
    # Set device to NPU
    device = 'npu'
    
    # Create test inputs
    num_requests = 4
    block_table_stride = 8
    block_size = 256
    
    # Create block table with random block indices
    block_table = torch.randint(0, 1000, (num_requests, block_table_stride), 
                               device=device, dtype=torch.int32)
    
    # Create cumulative number of blocks per request
    cu_num_blocks = torch.zeros(num_requests + 1, device=device, dtype=torch.int32)
    # Each request has different number of blocks
    num_blocks_per_request = torch.randint(1, block_table_stride, (num_requests,), 
                                         device=device, dtype=torch.int32)
    for i in range(num_requests):
        cu_num_blocks[i + 1] = cu_num_blocks[i] + num_blocks_per_request[i]
    
    total_blocks = cu_num_blocks[-1].item()
    
    # Create output tensor
    page_indices = torch.zeros(total_blocks, device=device, dtype=torch.int32)
    
    # Launch kernel
    grid = (num_requests,)
    _copy_page_indices_kernel[grid](
        page_indices,
        block_table,
        block_table_stride,
        cu_num_blocks,
        BLOCK_SIZE=block_size,
    )
    
    # Create reference implementation using PyTorch
    ref_page_indices = torch.zeros(total_blocks, device=device, dtype=torch.int32)
    ref_offset = 0
    for i in range(num_requests):
        start_idx = cu_num_blocks[i].item()
        end_idx = cu_num_blocks[i + 1].item()
        num_blocks = end_idx - start_idx
        ref_page_indices[ref_offset:ref_offset + num_blocks] = block_table[i, :num_blocks]
        ref_offset += num_blocks
    
    # Verify results
    assert torch.allclose(page_indices, ref_page_indices), \
        "Kernel output does not match reference implementation"
    
    # Verify that all non-zero values are within expected range
    assert torch.all(page_indices >= 0), "Page indices should be non-negative"
    assert torch.max(page_indices) < 1000, "Page indices should be within expected range"
    
    print("test_copy_page_indices_kernel passed!")

if __name__ == "__main__":
    test_copy_page_indices_kernel()
