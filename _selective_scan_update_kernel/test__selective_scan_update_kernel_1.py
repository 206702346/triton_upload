import torch
import triton
import triton.language as tl
import torch.nn.functional as F
from packaging import version

from typing import Optional

PAD_SLOT_ID = -1

TRITON3 = version.parse(triton.__version__) >= version.parse("3.0.0")

if TRITON3:

    @triton.jit
    def softplus(dt):
        dt = tl.where(dt <= 20.0, tl.math.log(tl.math.exp(dt) + 1), dt)
        return dt

else:

    @triton.jit
    def softplus(dt):
        dt = tl.where(dt <= 20.0, tl.math.log1p(tl.exp(dt)), dt)
        return dt

from _selective_scan_update_kernel import *



# ========== Reference Implementation ==========

class SelectiveScanUpdateReference(torch.nn.Module):
    """PyTorch reference implementation for Mamba selective scan update"""
    
    def forward(
        self,
        state: torch.Tensor,
        x: torch.Tensor,
        dt: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        D: Optional[torch.Tensor] = None,
        z: Optional[torch.Tensor] = None,
        dt_bias: Optional[torch.Tensor] = None,
        dt_softplus: bool = False,
        state_batch_indices: Optional[torch.Tensor] = None,
        pad_slot_id: int = -1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Reference implementation of selective scan update
        
        Args:
            state: [batch, nheads, dim, dstate] - SSM state
            x: [batch, nheads, dim] - Input
            dt: [batch, nheads, dim] - Delta time
            A: [nheads, dim, dstate] - State transition matrix
            B: [batch, ngroups, dstate] - Input projection
            C: [batch, ngroups, dstate] - Output projection
            D: [nheads, dim] - Skip connection (optional)
            z: [batch, nheads, dim] - Gating (optional)
            dt_bias: [nheads, dim] - Delta time bias (optional)
            dt_softplus: bool - Apply softplus to dt
            state_batch_indices: [batch] - Batch indices for state (optional)
            pad_slot_id: int - ID for padded slots to skip
        
        Returns:
            out: [batch, nheads, dim] - Output
            state: [batch, nheads, dim, dstate] - Updated state
        """
        batch, nheads, dim, dstate = state.shape
        ngroups = B.shape[1]
        nheads_ngroups_ratio = nheads // ngroups
        
        # Check if we should use tie_hdim mode (scalar dt and A)
        tie_hdim = (
            A.stride(-1) == 0 and A.stride(-2) == 0 and 
            dt.stride(-1) == 0 and 
            (dt_bias is None or dt_bias.stride(-1) == 0)
        )
        
        # Clone state for output
        state_updated = state.clone()
        out = torch.zeros_like(x)
        
        # Process each batch
        for b in range(batch):
            # Handle state batch indices
            if state_batch_indices is not None:
                state_batch_idx = state_batch_indices[b].item()
                if state_batch_idx == pad_slot_id:
                    # Skip padded entries
                    continue
                state_b = state_updated[state_batch_idx]
            else:
                state_b = state_updated[b]
            
            # Process each head
            for h in range(nheads):
                group_idx = h // nheads_ngroups_ratio
                
                # Get data for this batch and head
                x_bh = x[b, h, :].float()  # [dim]
                
                if not tie_hdim:
                    # Per-dimension dt and A
                    dt_bh = dt[b, h, :].float()  # [dim]
                    if dt_bias is not None:
                        dt_bh = dt_bh + dt_bias[h, :].float()
                    if dt_softplus:
                        dt_bh = F.softplus(dt_bh)
                    
                    A_h = A[h, :, :].float()  # [dim, dstate]
                    dA = torch.exp(A_h * dt_bh.unsqueeze(-1))  # [dim, dstate]
                else:
                    # Scalar dt and A (tied across dimensions)
                    dt_bh = dt[b, h, 0].float()  # scalar
                    if dt_bias is not None:
                        dt_bh = dt_bh + dt_bias[h, 0].float()
                    if dt_softplus:
                        dt_bh = softplus(dt_bh)
                    
                    A_h = A[h, 0, 0].float()  # scalar
                    dA = torch.exp(A_h * dt_bh)  # scalar
                
                B_b = B[b, group_idx, :].float()  # [dstate]
                C_b = C[b, group_idx, :].float()  # [dstate]
                
                # State update: state = state * dA + dB * x
                if not tie_hdim:
                    dB = B_b.unsqueeze(0) * dt_bh.unsqueeze(-1)  # [dim, dstate]
                    state_b[h] = state_b[h] * dA + dB * x_bh.unsqueeze(-1)
                else:
                    dB = B_b * dt_bh  # [dstate]
                    state_b[h] = state_b[h] * dA + dB.unsqueeze(0) * x_bh.unsqueeze(-1)
                
                # Output: out = sum(state * C, dim=-1)
                out_bh = torch.sum(state_b[h] * C_b.unsqueeze(0), dim=-1)  # [dim]
                
                # Add skip connection if D is provided
                if D is not None:
                    D_h = D[h, :].float()
                    out_bh = out_bh + x_bh * D_h
                
                # Apply gating if z is provided
                if z is not None:
                    z_bh = z[b, h, :].float()
                    out_bh = out_bh * z_bh * torch.sigmoid(z_bh)
                
                out[b, h, :] = out_bh
        
        return out, state_updated


# ========== Precision Tests ==========
def test_selective_scan_all_features():
    """Test with all optional features enabled"""
    batch, nheads, dim, dstate = 2, 4, 64, 16
    ngroups = 2
    
    state = torch.randn(batch, nheads, dim, dstate, device='npu', dtype=torch.float32)
    x = torch.randn(batch, nheads, dim, device='npu', dtype=torch.float32)
    dt = torch.randn(batch, nheads, dim, device='npu', dtype=torch.float32)
    A = torch.randn(nheads, dim, dstate, device='npu', dtype=torch.float32)
    B = torch.randn(batch, ngroups, dstate, device='npu', dtype=torch.float32)
    C = torch.randn(batch, ngroups, dstate, device='npu', dtype=torch.float32)
    D = torch.randn(nheads, dim, device='npu', dtype=torch.float32)
    z = torch.randn(batch, nheads, dim, device='npu', dtype=torch.float32)
    dt_bias = torch.randn(nheads, dim, device='npu', dtype=torch.float32)
    out = torch.zeros(batch, nheads, dim, device='npu', dtype=torch.float32)
    
    state_triton = state.clone()
    state_ref = state.clone()
    
    selective_state_update(
        state_triton, x, dt, A, B, C,
        D=D, z=z, dt_bias=dt_bias,
        dt_softplus=True,
        out=out
    )
    
    reference = SelectiveScanUpdateReference()
    out_ref, state_ref = reference(
        state_ref, x, dt, A, B, C,
        D=D, z=z, dt_bias=dt_bias,
        dt_softplus=True
    )
    
    max_diff_out = (out - out_ref).abs().max().item()
    print(f"All features - Output max diff: {max_diff_out:.6e}")
    
    rtol, atol = 1e-4, 1e-4
    assert torch.allclose(out, out_ref, rtol=rtol, atol=atol), \
        f"Output mismatch with all features: max diff = {max_diff_out}"
    
    print("? Test with all features passed")








# ========== Main Test Runner ==========

if __name__ == "__main__":
    test_selective_scan_all_features()
    print("All tests passed!")