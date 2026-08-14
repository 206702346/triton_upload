import torch
import triton
import triton.language as tl
import math

def _next_power_of_two(n: int) -> int:
    """Return the next power of two >= n."""
    return 1 << (n - 1).bit_length()

def _compute_block_size_and_warps(dim: int):
    """Heuristic to choose BLOCK_SIZE and num_warps based on dim."""
    # New heuristic: use fixed block sizes with consistent warps
    if dim <= 64:
        block_size = 64
        num_warps = 4
    elif dim <= 128:
        block_size = 128
        num_warps = 4
    else:
        block_size = 128
        num_warps = 8
    return block_size, num_warps

@triton.jit
def _state_passing_fwd_kernel(
    states_ptr,
    out_ptr,
    final_states_ptr,
    dA_cs_ptr,
    initstates_ptr,
    seq_idx_ptr,
    chunk_offsets_ptr,
    chunk_meta_num,
    dim,
    nchunks,
    seqlen,
    chunk_size,
    stride_states_batch,
    stride_states_chunk,
    stride_states_head,
    stride_states_dim,
    stride_out_batch,
    stride_out_chunk,
    stride_out_head,
    stride_out_dim,
    stride_final_states_batch,
    stride_final_states_head,
    stride_final_states_dim,
    stride_dA_cs_batch,
    stride_dA_cs_chunk,
    stride_dA_cs_head,
    stride_dA_cs_csize,
    stride_initstates_batch,
    stride_initstates_head,
    stride_initstates_dim,
    stride_seq_idx_batch,
    stride_seq_idx_seqlen,
    HAS_INITSTATES: tl.constexpr,
    HAS_SEQ_IDX: tl.constexpr,
    IS_CONT_BATCHED: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid_b = tl.program_id(axis=1)
    pid_h = tl.program_id(axis=2)
    pid_m = tl.program_id(axis=0)

    states_ptr = states_ptr + pid_b * stride_states_batch + pid_h * stride_states_head
    dA_cs_ptr = dA_cs_ptr + (
        pid_b * stride_dA_cs_batch
        + pid_h * stride_dA_cs_head
        + (chunk_size - 1) * stride_dA_cs_csize
    )
    out_ptr = out_ptr + pid_b * stride_out_batch + pid_h * stride_out_head
    final_states_ptr = final_states_ptr + (
        pid_b * stride_final_states_batch + pid_h * stride_final_states_head
    )
    
    if HAS_INITSTATES:
        initstates_ptr = initstates_ptr + pid_h * stride_initstates_head
        if not IS_CONT_BATCHED:
            initstates_ptr = initstates_ptr + pid_b * stride_initstates_batch

    if HAS_SEQ_IDX:
        seq_idx_ptr = seq_idx_ptr + pid_b * stride_seq_idx_batch

    offs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs_m < dim
    
    states_ptrs = states_ptr + offs_m * stride_states_dim
    out_ptrs = out_ptr + offs_m * stride_out_dim
    final_states_ptrs = final_states_ptr + offs_m * stride_final_states_dim

    if not HAS_INITSTATES:
        states = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    else:
        initstates_temp_ptr = initstates_ptr + offs_m * stride_initstates_dim
        states = tl.load(initstates_temp_ptr, mask=mask, other=0.0).to(tl.float32)

    tl.store(out_ptrs, states, mask=mask)
    out_ptrs += stride_out_chunk

    prev_seq_idx_chunk_end = tl.zeros((), dtype=tl.int32)
    logical_chunk_idx = tl.zeros((), dtype=tl.int32)

    for c in range(nchunks):
        new_states = tl.load(states_ptrs, mask=mask, other=0.0).to(tl.float32)
        dA_cs = tl.load(dA_cs_ptr).to(tl.float32)
        scale_mask = tl.full((), True, dtype=tl.int1)
        
        if HAS_SEQ_IDX:
            seq_end_idx = min((c + 1) * chunk_size, seqlen) - 1
            seq_start_idx = min(c * chunk_size, seqlen)
            
            seq_idx_chunk_end = tl.load(
                seq_idx_ptr + seq_end_idx * stride_seq_idx_seqlen
            ).to(tl.int32)
            
            if HAS_INITSTATES:
                if IS_CONT_BATCHED:
                    seq_changed = (prev_seq_idx_chunk_end != seq_idx_chunk_end)
                    
                    if seq_changed:
                        initstates_temp_ptr = initstates_ptr + seq_idx_chunk_end * stride_initstates_batch
                        states = tl.load(initstates_temp_ptr, mask=mask, other=0.0).to(tl.float32)

                        seq_idx_chunk_start = tl.load(
                            seq_idx_ptr + seq_start_idx * stride_seq_idx_seqlen
                        ).to(tl.int32)
                        logical_chunk_idx = logical_chunk_idx + (seq_idx_chunk_end - seq_idx_chunk_start)

                        if logical_chunk_idx < chunk_meta_num:
                            c_off = tl.load(chunk_offsets_ptr + logical_chunk_idx, other=0)
                            if c_off > 0:
                                boundary_idx = c_off - 1
                                if boundary_idx >= 0:
                                    dA_cs_boundary = tl.load(
                                        dA_cs_ptr - (chunk_size - 1) * stride_dA_cs_csize
                                        + boundary_idx * stride_dA_cs_csize,
                                        other=0.0
                                    ).to(tl.float32)
                                    dA_cs = dA_cs - dA_cs_boundary
                    
                    logical_chunk_idx = logical_chunk_idx + 1
            else:
                scale_mask = (seq_idx_chunk_end == prev_seq_idx_chunk_end)
            
            prev_seq_idx_chunk_end = seq_idx_chunk_end

        # Fused multiply-add using tl.math.fma, computing exp inside the where
        # This combines the scale and addition into a single fused operation when scale_mask is true.
        states = tl.where(
            scale_mask,
            tl.math.fma(tl.exp(dA_cs), states, new_states),
            new_states
        )

        if c < nchunks - 1:
            tl.store(out_ptrs, states, mask=mask)
        else:
            tl.store(final_states_ptrs, states, mask=mask)

        states_ptrs += stride_states_chunk
        dA_cs_ptr += stride_dA_cs_chunk
        out_ptrs += stride_out_chunk

def _state_passing_fwd(
    states,
    dA_cumsum,
    initial_states=None,
    seq_idx=None,
    chunk_size=None,
    out_dtype=None,
    is_cont_batched=False,
    chunk_offsets=None,
):
    batch, nchunks, nheads, dim = states.shape
    if chunk_size is None:
        chunk_size = dA_cumsum.shape[-1]
    else:
        assert chunk_size == dA_cumsum.shape[-1]
    assert dA_cumsum.shape == (batch, nheads, nchunks, chunk_size)
    if initial_states is not None:
        if is_cont_batched:
            assert seq_idx is not None, "seq_idx must be provided for continuous batching"
            assert chunk_offsets is not None, "chunk_offsets must be provided for continuous batching"
        else:
            assert initial_states.shape == (batch, nheads, dim)

    if seq_idx is not None:
        seqlen = seq_idx.shape[-1]
        assert seq_idx.shape == (batch, seqlen)
    out_dtype = states.dtype if out_dtype is None else out_dtype
    out = torch.empty(
        (batch, nchunks, nheads, dim), device=states.device, dtype=out_dtype
    )
    final_states = torch.empty(
        (batch, nheads, dim), device=states.device, dtype=torch.float32
    )

    # Compute adaptive block size and num_warps using new heuristic
    BLOCK_SIZE, num_warps = _compute_block_size_and_warps(dim)
    grid = lambda _: (triton.cdiv(dim, BLOCK_SIZE), batch, nheads)

    with torch.npu.device(states.device.index):
        _state_passing_fwd_kernel[grid](
            states,
            out,
            final_states,
            dA_cumsum,
            initial_states,
            seq_idx,
            chunk_offsets,
            len(chunk_offsets) if chunk_offsets is not None else 0,
            dim,
            nchunks,
            seqlen if seq_idx is not None else 0,
            chunk_size,
            states.stride(0),
            states.stride(1),
            states.stride(2),
            states.stride(3),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            out.stride(3),
            final_states.stride(0),
            final_states.stride(1),
            final_states.stride(2),
            dA_cumsum.stride(0),
            dA_cumsum.stride(2),
            dA_cumsum.stride(1),
            dA_cumsum.stride(3),
            *(
                (
                    initial_states.stride(0),
                    initial_states.stride(1),
                    initial_states.stride(2),
                )
                if initial_states is not None
                else (0, 0, 0)
            ),
            *(
                (seq_idx.stride(0), seq_idx.stride(1))
                if seq_idx is not None
                else (0, 0)
            ),
            HAS_INITSTATES=initial_states is not None,
            HAS_SEQ_IDX=seq_idx is not None,
            IS_CONT_BATCHED=is_cont_batched,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
        )
    return out, final_states