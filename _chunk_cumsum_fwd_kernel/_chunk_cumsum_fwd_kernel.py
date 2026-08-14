import math

import torch
import triton
import triton.language as tl
from packaging import version

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


@triton.jit
def _chunk_cumsum_fwd_kernel(
    # Pointers to matrices
    dt_ptr,
    A_ptr,
    dt_bias_ptr,
    dt_out_ptr,
    dA_cumsum_ptr,
    # Matrix dimension
    batch,
    seqlen,
    nheads,
    chunk_size,
    dt_min,
    dt_max,
    # Strides
    stride_dt_batch,
    stride_dt_seqlen,
    stride_dt_head,
    stride_A_head,
    stride_dt_bias_head,
    stride_dt_out_batch,
    stride_dt_out_chunk,
    stride_dt_out_head,
    stride_dt_out_csize,
    stride_dA_cs_batch,
    stride_dA_cs_chunk,
    stride_dA_cs_head,
    stride_dA_cs_csize,
    # Meta-parameters
    DT_SOFTPLUS: tl.constexpr,
    HAS_DT_BIAS: tl.constexpr,
    BLOCK_SIZE_CHUNK: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr = 32,  # Increased block size to reduce task count
):
    # Optimize grid task count by processing multiple chunks per thread when possible
    pid_b = tl.program_id(axis=0)
    
    # Use int32 instead of int64 for better vector computation support on Ascend
    pid_c = tl.program_id(axis=1).to(tl.int32)
    pid_h = tl.program_id(axis=2)
    
    dt_ptr += pid_b * stride_dt_batch + pid_c * chunk_size * stride_dt_seqlen
    dt_out_ptr += pid_b * stride_dt_out_batch + pid_c * stride_dt_out_chunk
    dA_cumsum_ptr += pid_b * stride_dA_cs_batch + pid_c * stride_dA_cs_chunk

    # Cast offsets to float32 for better performance in comparisons on Ascend
    offs_h = (pid_h * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)).to(tl.float32)
    offs_c = tl.arange(0, BLOCK_SIZE_CHUNK).to(tl.float32)
    
    # Optimize memory access patterns by ensuring contiguous access within chunks
    dt_ptrs = dt_ptr + (
        offs_h[:, None].to(tl.int32) * stride_dt_head + 
        offs_c[None, :].to(tl.int32) * stride_dt_seqlen
    )
    A_ptrs = A_ptr + offs_h.to(tl.int32) * stride_A_head
    dt_out_ptrs = dt_out_ptr + (
        offs_h[:, None].to(tl.int32) * stride_dt_out_head + 
        offs_c[None, :].to(tl.int32) * stride_dt_out_csize
    )
    dA_cs_ptrs = dA_cumsum_ptr + (
        offs_h[:, None].to(tl.int32) * stride_dA_cs_head + 
        offs_c[None, :].to(tl.int32) * stride_dA_cs_csize
    )
    
    # Cast limit to float32 for comparison operations
    chunk_size_limit = min(chunk_size, seqlen - pid_c * chunk_size)
    chunk_size_limit_f32 = chunk_size_limit.to(tl.float32)
    nheads_f32 = nheads.to(tl.float32)

    # Load data with optimized memory access
    dt = tl.load(
        dt_ptrs,
        mask=(offs_h[:, None] < nheads_f32) & (offs_c[None, :] < chunk_size_limit_f32),
        other=0.0,
    ).to(tl.float32)
    
    # Parallelize independent load operations
    if HAS_DT_BIAS:
        dt_bias = tl.load(
            dt_bias_ptr + offs_h.to(tl.int32) * stride_dt_bias_head, 
            mask=offs_h < nheads_f32, 
            other=0.0
        ).to(tl.float32)
        dt += dt_bias[:, None]
        
    if DT_SOFTPLUS:
        dt = tl.where(dt <= 20.0, softplus(dt), dt)
        
    # Use float32 comparisons for better performance on Ascend
    dt = tl.minimum(tl.maximum(dt, dt_min), dt_max)
    dt = tl.where(
        (offs_h[:, None] < nheads_f32) & (offs_c[None, :] < chunk_size_limit_f32), 
        dt, 
        0.0
    )
    
    # Store with optimized memory access pattern
    tl.store(
        dt_out_ptrs,
        dt,
        mask=(offs_h[:, None] < nheads_f32) & (offs_c[None, :] < chunk_size.to(tl.float32)),
    )
    
    # Load A parameter
    A = tl.load(A_ptrs, mask=offs_h < nheads_f32, other=0.0).to(tl.float32)
    
    # Compute dA and cumulative sum
    dA = dt * A[:, None]
    dA_cs = tl.cumsum(dA, axis=1)
    
    # Store result with optimized memory access
    tl.store(
        dA_cs_ptrs,
        dA_cs,
        mask=(offs_h[:, None] < nheads_f32) & (offs_c[None, :] < chunk_size.to(tl.float32)),
    )
    
def _chunk_cumsum_fwd(
    dt, A, chunk_size, dt_bias=None, dt_softplus=False, dt_limit=(0.0, float("inf"))
):
    batch, seqlen, nheads = dt.shape
    assert A.shape == (nheads,)
    if dt_bias is not None:
        assert dt_bias.shape == (nheads,)
    nchunks = math.ceil(seqlen / chunk_size)
    
    # Ensure output tensors are on the correct device
    dt_out = torch.empty(
        batch, nheads, nchunks, chunk_size, device=dt.device, dtype=torch.float32
    )
    dA_cumsum = torch.empty(
        batch, nheads, nchunks, chunk_size, device=dt.device, dtype=torch.float32
    )
    
    # Optimize grid configuration to better match Ascend NPU core count
    # Reduce task count by increasing BLOCK_SIZE_H and processing more heads per task
    grid_chunk_cs = lambda META: (
        batch,
        nchunks,
        triton.cdiv(nheads, META["BLOCK_SIZE_H"]),
    )
    
    _chunk_cumsum_fwd_kernel[grid_chunk_cs](
        dt,
        A,
        dt_bias,
        dt_out,
        dA_cumsum,
        batch,
        seqlen,
        nheads,
        chunk_size,
        dt_limit[0],
        dt_limit[1],
        dt.stride(0),
        dt.stride(1),
        dt.stride(2),
        A.stride(0),
        dt_bias.stride(0) if dt_bias is not None else 0,
        dt_out.stride(0),
        dt_out.stride(2),
        dt_out.stride(1),
        dt_out.stride(3),
        dA_cumsum.stride(0),
        dA_cumsum.stride(2),
        dA_cumsum.stride(1),
        dA_cumsum.stride(3),
        dt_softplus,
        HAS_DT_BIAS=dt_bias is not None,
        BLOCK_SIZE_CHUNK=triton.next_power_of_2(chunk_size),
        BLOCK_SIZE_H=32,  # Increased from 16 to 32 to reduce task count
    )
    return dA_cumsum, dt_out
