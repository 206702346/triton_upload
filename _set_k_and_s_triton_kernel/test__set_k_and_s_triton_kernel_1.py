import torch
import triton
import triton.language as tl

device = 'npu'
from _set_k_and_s_triton_kernel import _set_k_and_s_triton


if __name__ == "__main__":
    # Test parameters
    num_pages = 4
    page_size = 64
    num_tokens_to_write = 3
    index_head_dim = 128
    buf_numel_per_page = page_size * (index_head_dim + 4)  # 128B data + 4B scale

    # Create test tensors on NPU
    buf = torch.zeros((num_pages, buf_numel_per_page), dtype=torch.uint8, device=device)
    loc = torch.tensor([0, 64, 128], dtype=torch.int64, device=device)  # token indices
    index_k = torch.randn((num_tokens_to_write, index_head_dim), dtype=torch.float16, device=device)
    index_k_scale = torch.randn((num_tokens_to_write, 1), dtype=torch.float32, device=device)

    # Call the function
    _set_k_and_s_triton(buf, loc, index_k, index_k_scale, page_size)

    # Verify correctness
    buf_fp16 = buf.view(torch.float16)
    buf_fp32 = buf.view(torch.float32)
    
    # Check first token
    loc_page_index = loc[0] // page_size
    loc_token_offset_in_page = loc[0] % page_size
    out_k_offsets = (
        loc_page_index * buf_numel_per_page
        + loc_token_offset_in_page * index_head_dim
        + torch.arange(0, index_head_dim, device=device)
    )
    out_s_offset = (
        loc_page_index * buf_numel_per_page // 4
        + (page_size * index_head_dim) // 4
        + loc_token_offset_in_page
    )
    
    # Validate k values
    stored_k = buf_fp16.flatten()[out_k_offsets]
    expected_k = index_k[0]
    assert torch.allclose(stored_k, expected_k, atol=1e-3), "K values do not match"
    
    # Validate scale values
    stored_scale = buf_fp32.flatten()[out_s_offset]
    expected_scale = index_k_scale[0]
    assert torch.allclose(stored_scale, expected_scale, atol=1e-3), "Scale values do not match"
    
    print("All tests passed!")