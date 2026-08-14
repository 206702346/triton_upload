import torch
from _correct_attn_cp_out_kernel_1 import correct_attn_cp_out

device = 'npu'

def torch_correct_attn_cp_out(outputs, lses, lse_idx):
    """Reference implementation in PyTorch"""
    B, H, D = outputs.shape
    N = lses.shape[0]
    
    # Calculate final lse
    lse = lses.clone()
    lse = torch.where((lse != lse) | (lse == float("inf")), -float("inf"), lse)
    lse_max = torch.max(lse, dim=0, keepdim=True).values
    lse_max = torch.where(lse_max == -float("inf"), torch.tensor(0.0, device=device), lse_max)
    lse = lse - lse_max
    lse_exp = torch.exp(lse)
    lse_acc = torch.sum(lse_exp, dim=0)
    lse_final = torch.log(lse_acc) + lse_max
    
    # Correct output
    lse_tmp = lses[lse_idx]
    lse_finally = lse_tmp - lse_final
    lse_finally = torch.where(
        (lse_finally != lse_finally) | (lse_finally == float("inf")),
        -float("inf"),
        lse_finally
    )
    factor = torch.exp(lse_finally)
    new_output = outputs * factor.unsqueeze(-1)
    
    return new_output, lse_final

def test_correct_attn_cp_out():
    """Test the kernel implementation against PyTorch reference"""
    torch.manual_seed(0)

    B, H, D, N = 4, 8, 64, 4
    lse_idx = 2

    outputs = torch.randn(B, H, D, device=device, dtype=torch.float32)
    lses = torch.randn(N, B, H, device=device, dtype=torch.float32)

    lses[0, 0, 0] = float('inf')
    lses[1, 1, 1] = float('nan')

    triton_new_output, triton_vlse = correct_attn_cp_out(outputs, lses, lse_idx)
    torch_new_output, torch_vlse = torch_correct_attn_cp_out(outputs, lses, lse_idx)

    assert torch.allclose(triton_new_output, torch_new_output, atol=1e-5, rtol=1e-3), \
        f"Outputs don't match! Max diff: {torch.max(torch.abs(triton_new_output - torch_new_output))}"
    assert torch.allclose(triton_vlse, torch_vlse, atol=1e-5, rtol=1e-3), \
        f"VLSEs don't match! Max diff: {torch.max(torch.abs(triton_vlse - torch_vlse))}"
    
    print("✅ Triton and PyTorch implementations match!")
    print(f"Output shape: {triton_new_output.shape}")
    print(f"VLSE shape: {triton_vlse.shape}")

    print(f"Sample output values: {triton_new_output[0, 0, :5]}")
    print(f"Sample VLSE values: {triton_vlse[0, :5]}")

if __name__ == "__main__":
    test_correct_attn_cp_out()
