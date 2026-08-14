import math

import torch
import triton
import triton.language as tl
from packaging import version

TRITON3 = version.parse(triton.__version__) >= version.parse("3.0.0")

from _chunk_cumsum_fwd_kernel import _chunk_cumsum_fwd


if __name__ == "__main__":
    # Test the function with sample data
    device = torch.device("npu")
    
    # Create sample input tensors
    batch, seqlen, nheads, chunk_size = 2, 16, 4, 8
    dt = torch.randn(batch, seqlen, nheads, device=device, dtype=torch.float32)
    A = torch.randn(nheads, device=device, dtype=torch.float32)
    dt_bias = torch.randn(nheads, device=device, dtype=torch.float32)
    
    # Call the function
    dA_cumsum, dt_out = _chunk_cumsum_fwd(dt, A, chunk_size, dt_bias=dt_bias, dt_softplus=True, dt_limit=(0.0, 10.0))
    
    # Reference implementation using PyTorch
    def chunk_cumsum_ref(dt, A, chunk_size, dt_bias=None, dt_softplus=False, dt_limit=(0.0, float("inf"))):
        batch, seqlen, nheads = dt.shape
        nchunks = math.ceil(seqlen / chunk_size)
        
        # Pad dt to make it divisible by chunk_size
        padded_seqlen = nchunks * chunk_size
        dt_padded = torch.zeros(batch, padded_seqlen, nheads, device=dt.device, dtype=torch.float32)
        dt_padded[:, :seqlen, :] = dt
        
        # Reshape to chunks
        dt_chunked = dt_padded.view(batch, nchunks, chunk_size, nheads).transpose(2, 3)  # (batch, nchunks, nheads, chunk_size)
        
        # Add bias
        if dt_bias is not None:
            dt_chunked = dt_chunked + dt_bias.view(1, 1, nheads, 1)
        
        # Apply softplus
        if dt_softplus:
            dt_chunked = torch.where(dt_chunked <= 20.0, torch.log1p(torch.exp(dt_chunked)), dt_chunked)
        
        # Clamp
        dt_chunked = torch.clamp(dt_chunked, dt_limit[0], dt_limit[1])
        
        # Compute dA = dt * A
        A_expanded = A.view(1, 1, nheads, 1)
        dA = dt_chunked * A_expanded
        
        # Compute cumulative sum
        dA_cumsum_ref = torch.cumsum(dA, dim=-1)
        
        return dA_cumsum_ref.transpose(1, 2), dt_chunked.transpose(1, 2)  # Transpose to match output format
    
    # Compute reference result
    dA_cumsum_ref, dt_out_ref = chunk_cumsum_ref(dt, A, chunk_size, dt_bias=dt_bias, dt_softplus=True, dt_limit=(0.0, 10.0))
    
    # Compare results
    torch.testing.assert_close(dA_cumsum, dA_cumsum_ref, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(dt_out, dt_out_ref, atol=1e-4, rtol=1e-4)
    
    print("All tests passed!")