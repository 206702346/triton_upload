import torch
import torch_npu
import triton
import triton.language as tl
from _pack_seq_kernel import pack_seq_triton

def test_pack_seq():

    torch.manual_seed(0)
    device = 'npu'

    N = 4096
    D = 4
    B = 3
    
    x = torch.randn(N, D, device=device, dtype=torch.float32)
    lengths = torch.tensor([3, 4, 3], device=device, dtype=torch.int32)

    result_triton = pack_seq_triton(x, lengths, pad_value=0.0, block_t=32, block_d=32)

    def pack_seq_reference(x, lengths, pad_value=0.0):
        B = lengths.numel()
        Lmax = int(lengths.max().item())
        D = x.shape[1]
        out = torch.full((B, Lmax, D), pad_value, device=x.device, dtype=x.dtype)
        
        start_idx = 0
        for i in range(B):
            length = lengths[i].item()
            out[i, :length, :] = x[start_idx:start_idx + length, :]
            start_idx += length
        
        return out
    
    result_ref = pack_seq_reference(x, lengths, pad_value=0.0)

    assert torch.allclose(result_triton, result_ref, atol=1e-5), "Results don't match!"
    print("Test 1 passed: Basic 2D input")

    print("All tests passed!")

if __name__ == "__main__":
    test_pack_seq()