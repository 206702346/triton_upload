import torch
import triton
import triton.language as tl


def to_npu(tensor):
    if tensor is not None and tensor.device.type != 'npu':
        return tensor.to('npu')
    return tensor

from _state_passing_fwd_kernel import _state_passing_fwd

if __name__ == "__main__":
    # ===== 1. 测试配置 =====
    device = torch.device('npu')
    batch, nchunks, nheads, dim = 2, 4, 3, 32
    chunk_size = 8
    seqlen = nchunks * chunk_size
    rtol, atol = 1e-5, 1e-5  # 精度容差

    # ===== 2. 创建输入张量 =====
    states = torch.randn(batch, nchunks, nheads, dim, device=device, dtype=torch.float32)
    dA_cumsum = torch.randn(batch, nheads, nchunks, chunk_size, device=device, dtype=torch.float32)
    initial_states = torch.randn(batch, nheads, dim, device=device, dtype=torch.float32)
    seq_idx = torch.randint(0, 2, (batch, seqlen), device=device)
    chunk_offsets = torch.tensor([0, 2, 4], device=device)

    # ===== 3. 调用Triton内核 =====
    out, final_states = _state_passing_fwd(
        states=states, dA_cumsum=dA_cumsum, initial_states=initial_states,
        seq_idx=seq_idx, chunk_size=chunk_size, is_cont_batched=False,
        chunk_offsets=chunk_offsets
    )

    # ===== 4. PyTorch参考实现 =====
    def pytorch_state_passing_fwd(states, dA_cumsum, initial_states, seq_idx, chunk_size):
    
        batch, nchunks, nheads, dim = states.shape
        out_ref = torch.zeros(batch, nchunks, nheads, dim, device=states.device, dtype=torch.float32)
        final_states_ref = torch.zeros(batch, nheads, dim, device=states.device, dtype=torch.float32)
    
        for b in range(batch):
            for h in range(nheads):
                # Initialize state
                if initial_states is not None:
                    state = initial_states[b, h].clone().float()
                else:
                    state = torch.zeros(dim, device=states.device, dtype=torch.float32)
                
                # Store initial state (before processing any chunk)
                out_ref[b, 0, h] = state
                
                # Process each chunk
                for c in range(nchunks):
                    # Load new states from chunk c
                    new_states = states[b, c, h].float()
                    
                    # Load dA_cs (cumulative sum at end of chunk c)
                    dA_cs = dA_cumsum[b, h, c, -1].float()
                    
                    # Compute scale
                    scale = torch.exp(dA_cs)
                    
                    # State update: matches Triton kernel exactly
                    state = scale * state + new_states
                    
                    # Store output
                    if c < nchunks - 1:
                        # Store intermediate states
                        out_ref[b, c + 1, h] = state
                    else:
                        # Store final state
                        final_states_ref[b, h] = state
        
        return out_ref, final_states_ref

    # ===== 5. 计算参考输出 =====
    out_ref, final_states_ref = pytorch_state_passing_fwd(
        states, dA_cumsum, initial_states, seq_idx, chunk_size
    )
    


    # ===== 6. 精度验证（关键！） =====
    # 6.1 最大绝对误差
    max_diff_out = (out - out_ref).abs().max().item()
    max_diff_final = (final_states - final_states_ref).abs().max().item()
    
    # 6.2 相对误差
    rel_diff_out = ((out - out_ref) / (out_ref.abs() + 1e-9)).abs().max().item()
    rel_diff_final = ((final_states - final_states_ref) / (final_states_ref.abs() + 1e-9)).abs().max().item()
    
    print(f"Max absolute diff - out: {max_diff_out:.2e}, final_states: {max_diff_final:.2e}")
    print(f"Max relative diff - out: {rel_diff_out:.2e}, final_states: {rel_diff_final:.2e}")

    # 6.3 严格断言（必须满足）
    assert out.device.type == 'npu', "输出应在NPU上"
    assert torch.allclose(out, out_ref, rtol=rtol, atol=atol), \
        f"out精度不匹配！最大误差: {max_diff_out:.2e} > {atol}"
    assert torch.allclose(final_states, final_states_ref, rtol=rtol, atol=atol), \
        f"final_states精度不匹配！最大误差: {max_diff_final:.2e} > {atol}"

  