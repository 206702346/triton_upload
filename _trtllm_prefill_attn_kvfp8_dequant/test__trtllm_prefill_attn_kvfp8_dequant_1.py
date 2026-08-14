import torch
import pytest
import numpy as np
from _trtllm_prefill_attn_kvfp8_dequant import trtllm_prefill_attn_kvfp8_dequant

def test_trtllm_prefill_attn_kvfp8_dequant():
    """Test the FP8 dequantization function for prefilled attention."""
    # Set device to NPU
    device = torch.device('npu')
    
    # Create sample input tensors
    batch_size = 2
    num_pages_per_token = 3
    num_blocks = 10
    head_dim = 64
    num_heads = 8
    block_size = 16
    
    # Create FP8 kv_cache (simulating quantized cache)
    kv_cache_shape = (num_blocks, 2, num_heads, block_size, head_dim)
    kv_cache = torch.randn(kv_cache_shape, dtype=torch.float32, device=device)
    # Convert to FP8 by scaling and converting to uint8 (simulating FP8 storage)
    kv_cache_fp8 = (kv_cache * 100).to(torch.uint8)
    
    # Create block tables for prefills
    block_tables_prefill = torch.randint(1, num_blocks, 
                                        (batch_size, num_pages_per_token), 
                                        dtype=torch.int32, device=device)
    
    # Create quantization scales
    k_scale = torch.tensor([0.01], dtype=torch.float32, device=device)
    v_scale = torch.tensor([0.02], dtype=torch.float32, device=device)
    
    # Test with bfloat16 dequantization
    dequant_dtype = torch.bfloat16
    
    # Call the function
    mock_kv_cache, mock_block_table = trtllm_prefill_attn_kvfp8_dequant(
        kv_cache_fp8, block_tables_prefill, k_scale, v_scale, dequant_dtype
    )
    
    # Verify output shapes
    expected_mock_cache_shape = (batch_size * num_pages_per_token + 1, 2, 
                               num_heads, block_size, head_dim)
    expected_block_table_shape = (batch_size, num_pages_per_token)
    
    assert mock_kv_cache.shape == expected_mock_cache_shape, \
        f"Expected shape {expected_mock_cache_shape}, got {mock_kv_cache.shape}"
    assert mock_block_table.shape == expected_block_table_shape, \
        f"Expected shape {expected_block_table_shape}, got {mock_block_table.shape}"
    
    # Verify data types
    assert mock_kv_cache.dtype == dequant_dtype, \
        f"Expected dtype {dequant_dtype}, got {mock_kv_cache.dtype}"
    assert mock_block_table.dtype == torch.int32, \
        f"Expected dtype torch.int32, got {mock_block_table.dtype}"
    
    # Verify mock block table contains sequential indices starting from 1
    expected_indices = torch.arange(1, batch_size * num_pages_per_token + 1, 
                                   dtype=torch.int32, device=device)
    expected_indices = expected_indices.reshape(batch_size, num_pages_per_token)
    assert torch.equal(mock_block_table, expected_indices), \
        "Mock block table does not contain expected sequential indices"
    
    # Test with float16 dequantization
    dequant_dtype_fp16 = torch.float16
    mock_kv_cache_fp16, mock_block_table_fp16 = trtllm_prefill_attn_kvfp8_dequant(
        kv_cache_fp8, block_tables_prefill, k_scale, v_scale, dequant_dtype_fp16
    )
    
    assert mock_kv_cache_fp16.dtype == dequant_dtype_fp16, \
        f"Expected dtype {dequant_dtype_fp16}, got {mock_kv_cache_fp16.dtype}"
    assert mock_kv_cache_fp16.shape == expected_mock_cache_shape, \
        f"Expected shape {expected_mock_cache_shape}, got {mock_kv_cache_fp16.shape}"

def test_trtllm_prefill_attn_kvfp8_dequant_edge_cases():
    """Test edge cases for the FP8 dequantization function."""
    device = torch.device('npu')

    batch_size = 1
    num_pages_per_token = 2
    num_blocks = 5
    head_dim = 32
    num_heads = 4
    block_size = 8
    
    kv_cache_shape = (num_blocks, 2, num_heads, block_size, head_dim)
    kv_cache = torch.randn(kv_cache_shape, dtype=torch.float32, device=device)
    kv_cache_fp8 = (kv_cache * 100).to(torch.uint8)
    
    block_tables_prefill = torch.randint(1, num_blocks, 
                                        (batch_size, num_pages_per_token), 
                                        dtype=torch.int32, device=device)
    
    k_scale = torch.tensor([0.005], dtype=torch.float32, device=device)
    v_scale = torch.tensor([0.01], dtype=torch.float32, device=device)
    
    mock_kv_cache, mock_block_table = trtllm_prefill_attn_kvfp8_dequant(
        kv_cache_fp8, block_tables_prefill, k_scale, v_scale, torch.bfloat16
    )

    expected_cache_size = batch_size * num_pages_per_token + 1
    assert mock_kv_cache.shape[0] == expected_cache_size, \
        f"Expected cache size {expected_cache_size}, got {mock_kv_cache.shape[0]}"
    
    print("All tests passed!")

if __name__ == "__main__":
    test_trtllm_prefill_attn_kvfp8_dequant()
    test_trtllm_prefill_attn_kvfp8_dequant_edge_cases()
    print("✓ All unit tests completed successfully!")
