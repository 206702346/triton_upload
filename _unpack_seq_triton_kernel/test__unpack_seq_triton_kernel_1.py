import torch
import triton
import triton.language as tl
from _unpack_seq_triton_kernel import unpack_seq_triton

def test_unpack_seq_triton():

    torch.manual_seed(0)
    device = 'npu'
    
    B, Lmax, D = 32, 4096, 64
    packed_tensor = torch.randn(B, Lmax, D, device=device, dtype=torch.float32)
    lengths = torch.tensor([7, 5, 8], device=device, dtype=torch.int32)

    def unpack_seq_reference(packed_tensor, lengths):
        results = []
        for i, length in enumerate(lengths):
            results.append(packed_tensor[i, :length])
        return torch.cat(results, dim=0)

    result_triton = unpack_seq_triton(packed_tensor, lengths)
    result_reference = unpack_seq_reference(packed_tensor, lengths)
    
    print(f"Triton result shape: {result_triton.shape}")
    print(f"Reference result shape: {result_reference.shape}")
    print(f"Total elements: {lengths.sum().item()}")
    
    assert torch.allclose(result_triton, result_reference, atol=1e-5), "Results don't match!"
    print("✅ Test 1 passed: Basic 3D tensor")

if __name__ == "__main__":
    test_unpack_seq_triton()
