import torch
import triton
import triton.language as tl

from _dequantize_k_cache_fast_kernel import _dequantize_k_cache_fast

if __name__ == "__main__":
    # Test the function with sample data
    device = torch.device("npu" if torch.npu.is_available() else "cpu")
    
    # Create sample input data
    num_tokens = 4
    dim_quant = 656
    group_size = 128
    
    # Create quant_k_cache tensor on NPU
    quant_k_cache = torch.randn((num_tokens, dim_quant), dtype=torch.float32, device=device).to(torch.bfloat16)
    
    # Run the Triton kernel
    output_triton = _dequantize_k_cache_fast(quant_k_cache, group_size)
    
    # Reference implementation in PyTorch
    dim_nope = 512
    dim_rope = 64
    num_tiles = dim_nope // group_size
    
    input_nope_q = quant_k_cache[:, :dim_nope]
    input_nope_s = quant_k_cache[:, dim_nope : dim_nope + num_tiles * 4].view(torch.float32)
    input_rope = quant_k_cache[:, dim_nope + num_tiles * 4 : dim_nope + num_tiles * 4 + dim_rope].view(torch.bfloat16)
    
    # Dequantize nope part
    output_ref = torch.empty((num_tokens, dim_nope + dim_rope), dtype=torch.bfloat16, device=device)
    
    # Process nope blocks
    for i in range(num_tiles):
        start_idx = i * group_size
        end_idx = start_idx + group_size
        q_block = input_nope_q[:, start_idx:end_idx].to(torch.float32)
        s_block = input_nope_s[:, i].unsqueeze(1)
        dequant_block = (q_block * s_block).to(torch.bfloat16)
        output_ref[:, start_idx:end_idx] = dequant_block
    
    # Copy rope part
    output_ref[:, dim_nope:] = input_rope[:, :dim_rope]
    
    # Compare results
    torch.testing.assert_close(output_triton, output_ref, rtol=1e-2, atol=1e-2)
    print("Test passed: Triton output matches PyTorch reference implementation.")