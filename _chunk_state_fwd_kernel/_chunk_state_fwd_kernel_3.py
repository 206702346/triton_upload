# Adapted from: https://github.com/vllm-project/vllm/tree/main/vllm/model_executor/layers/mamba/ops/ssd_chunk_state.py

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Copyright (c) 2024, Tri Dao, Albert Gu.
# Adapted from https://github.com/state-spaces/mamba/blob/v2.2.4/mamba_ssm/ops/triton/ssd_chunk_state.py

# ruff: noqa: E501

import math

import torch
import triton
import triton.language as tl


@triton.jit
def softplus(dt):
    dt = tl.where(dt <= 20.0, tl.math.log(tl.math.exp(dt) + 1), dt)
    return dt


@triton.jit
def _chunk_state_fwd_kernel(
    # Pointers to matrices
    x_ptr,
    b_ptr,
    states_ptr,
    scale_ptr,                    # precomputed scale tensor (exp(dA_cs_last - dA_cs_k) * dt_k)
    seq_idx_ptr,
    # Matrix dimensions
    hdim,
    dstate,
    chunk_size,
    batch,
    seqlen,
    nheads_ngroups_ratio,
    # Strides
    stride_x_batch,
    stride_x_seqlen,
    stride_x_head,
    stride_x_hdim,
    stride_b_batch,
    stride_b_seqlen,
    stride_b_head,
    stride_b_dstate,
    stride_states_batch,
    stride_states_chunk,
    stride_states_head,
    stride_states_hdim,
    stride_states_dstate,
    stride_scale_batch,
    stride_scale_chunk,
    stride_scale_head,
    stride_scale_csize,
    stride_seq_idx_batch,
    stride_seq_idx_seqlen,
    # Meta-parameters
    HAS_SEQ_IDX: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr = 16,
    BLOCK_SIZE_N: tl.constexpr = 16,
    BLOCK_SIZE_K: tl.constexpr = 16,
    USE_MANUAL_REDUCTION: tl.constexpr = False,  # for small dstate
):
    # program id mapping: (batch, chunk, head)
    pid_bc = tl.program_id(axis=1).to(tl.int64)
    pid_c = pid_bc // batch
    pid_b = pid_bc - pid_c * batch
    pid_h = tl.program_id(axis=2)

    # tile indices within the (headdim, dstate) output matrix
    num_pid_n = tl.cdiv(dstate, BLOCK_SIZE_N)
    pid_m = tl.program_id(axis=0) // num_pid_n
    pid_n = tl.program_id(axis=0) % num_pid_n

    # chunk size limit for the last incomplete chunk
    chunk_size_limit = min(chunk_size, seqlen - pid_c * chunk_size)

    # Base pointers for the current (batch, chunk, head)
    x_base = (
        x_ptr
        + pid_b * stride_x_batch
        + pid_c * chunk_size * stride_x_seqlen
        + pid_h * stride_x_head
    )
    b_base = (
        b_ptr
        + pid_b * stride_b_batch
        + pid_c * chunk_size * stride_b_seqlen
        + (pid_h // nheads_ngroups_ratio) * stride_b_head
    )
    states_base = (
        states_ptr
        + pid_b * stride_states_batch
        + pid_c * stride_states_chunk
        + pid_h * stride_states_head
    )
    scale_base = (
        scale_ptr
        + pid_b * stride_scale_batch
        + pid_c * stride_scale_chunk
        + pid_h * stride_scale_head
    )

    # --- block pointers for the inputs ---
    x_block_ptr = tl.make_block_ptr(
        base=x_base,
        shape=(chunk_size_limit, hdim),
        strides=(stride_x_seqlen, stride_x_hdim),
        offsets=(0, pid_m * BLOCK_SIZE_M),
        block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_M),
        order=(1, 0),
    )
    b_block_ptr = tl.make_block_ptr(
        base=b_base,
        shape=(chunk_size_limit, dstate),
        strides=(stride_b_seqlen, stride_b_dstate),
        offsets=(0, pid_n * BLOCK_SIZE_N),
        block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_N),
        order=(1, 0),
    )
    scale_block_ptr = tl.make_block_ptr(
        base=scale_base,
        shape=(chunk_size_limit,),
        strides=(stride_scale_csize,),
        offsets=(0,),
        block_shape=(BLOCK_SIZE_K,),
        order=(0,),
    )
    if HAS_SEQ_IDX:
        seq_idx_block_ptr = tl.make_block_ptr(
            base=seq_idx_ptr
            + pid_b * stride_seq_idx_batch
            + pid_c * chunk_size * stride_seq_idx_seqlen,
            shape=(chunk_size_limit,),
            strides=(stride_seq_idx_seqlen,),
            offsets=(0,),
            block_shape=(BLOCK_SIZE_K,),
            order=(0,),
        )
        seq_idx_last = tl.load(
            seq_idx_ptr
            + pid_b * stride_seq_idx_batch
            + (pid_c * chunk_size + chunk_size_limit - 1) * stride_seq_idx_seqlen
        )

    # output accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # time loop over the chunk
    for k in range(0, chunk_size_limit, BLOCK_SIZE_K):
        # load tiles with padding
        x = tl.load(x_block_ptr, boundary_check=(0, 1), padding_option="zero")
        b = tl.load(b_block_ptr, boundary_check=(0, 1), padding_option="zero").to(
            tl.float32
        )
        scale = tl.load(scale_block_ptr, boundary_check=(0,), padding_option="zero").to(
            tl.float32
        )

        # apply seq_idx masking if needed
        if HAS_SEQ_IDX:
            seq_idx_k = tl.load(
                seq_idx_block_ptr, boundary_check=(0,), padding_option="zero"
            )
            scale = tl.where(seq_idx_k == seq_idx_last, scale, 0.0)

        b *= scale[:, None]               # (BLOCK_SIZE_K, BLOCK_SIZE_N)

        if USE_MANUAL_REDUCTION:
            # For very small dstate, avoid tl.dot and use a manual reduction
            # (per-warp sequential multiply + reduce) to reduce tensor core overhead.
            # x_t shape: (BLOCK_SIZE_M, BLOCK_SIZE_K), b shape: (BLOCK_SIZE_K, BLOCK_SIZE_N)
            x_t = tl.trans(x)
            # Broadcast multiply and sum over the K dimension.
            # (M, K, 1) * (1, K, N) -> sum over K -> (M, N)
            prod = x_t[:, :, None] * b[None, :, :]   # (M, K, N)
            acc += tl.sum(prod, axis=1)               # (M, N)
        else:
            b = b.to(x_ptr.dtype.element_ty)          # match precision for tl.dot
            x_t = tl.trans(x)
            acc += tl.dot(x_t, b)

        # advance block pointers along the time dimension
        x_block_ptr = tl.advance(x_block_ptr, (BLOCK_SIZE_K, 0))
        b_block_ptr = tl.advance(b_block_ptr, (BLOCK_SIZE_K, 0))
        scale_block_ptr = tl.advance(scale_block_ptr, (BLOCK_SIZE_K,))
        if HAS_SEQ_IDX:
            seq_idx_block_ptr = tl.advance(seq_idx_block_ptr, (BLOCK_SIZE_K,))

    # store the output tile -> coalesced store into the states tensor
    states_block_ptr = tl.make_block_ptr(
        base=states_base,
        shape=(hdim, dstate),
        strides=(stride_states_hdim, stride_states_dstate),
        offsets=(pid_m * BLOCK_SIZE_M, pid_n * BLOCK_SIZE_N),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N),
        order=(1, 0),
    )
    states_out = acc.to(states_ptr.dtype.element_ty)
    tl.store(states_block_ptr, states_out, boundary_check=(0, 1))


def _chunk_state_fwd(
    B, x, dt, dA_cumsum, seq_idx=None, states=None, states_in_fp32=True
):
    batch, seqlen, nheads, headdim = x.shape
    _, _, nchunks, chunk_size = dt.shape
    _, _, ngroups, dstate = B.shape
    assert nheads % ngroups == 0
    assert B.shape == (batch, seqlen, ngroups, dstate)
    assert dt.shape == (batch, nheads, nchunks, chunk_size)
    assert dA_cumsum.shape == dt.shape
    if seq_idx is not None:
        assert seq_idx.shape == (batch, seqlen)
    if states is not None:
        assert states.shape == (batch, nchunks, nheads, headdim, dstate)
    else:
        states_dtype = torch.float32 if states_in_fp32 else B.dtype
        states = torch.empty(
            (batch, nchunks, nheads, headdim, dstate),
            device=B.device,
            dtype=states_dtype,
        )

    # Precompute the scaling factor for every (batch, head, chunk, time‑step)
    # scale = exp(dA_cumsum_last - dA_cumsum) * dt
    # This avoids recomputing exp and multiply inside the main kernel for each tile.
    dA_cumsum_last = dA_cumsum[..., -1:]                          # keep last dim singleton
    dA_cumsum_last_expanded = dA_cumsum_last.expand_as(dA_cumsum) # broadcast along chunk_size
    scale_precomputed = torch.exp(dA_cumsum_last_expanded - dA_cumsum) * dt

    # Use the same dtype as dt / dA_cumsum for the precomputed scale to avoid extra casts.
    scale_precomputed = scale_precomputed.to(dt.dtype)

    # Decide kernel variant based on dstate size.
    # For very small dstate (≤16) we use a manual reduction to avoid tensor core overhead.
    is_small_dstate = (dstate <= 16)
    if is_small_dstate:
        BLOCK_SIZE_N_val = dstate  # cover the entire dstate in one tile
        USE_MANUAL_REDUCTION = True
    else:
        BLOCK_SIZE_N_val = 16
        USE_MANUAL_REDUCTION = False

    # Grid depends on selected BLOCK_SIZE_N
    grid = lambda META: (
        triton.cdiv(headdim, META["BLOCK_SIZE_M"])
        * triton.cdiv(dstate, META["BLOCK_SIZE_N"]),
        batch * nchunks,
        nheads,
    )

    _chunk_state_fwd_kernel[grid](
        x,
        B,
        states,
        scale_precomputed,
        seq_idx if seq_idx is not None else torch.empty(0, device=x.device),
        headdim,
        dstate,
        chunk_size,
        batch,
        seqlen,
        nheads // ngroups,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        x.stride(3),
        B.stride(0),
        B.stride(1),
        B.stride(2),
        B.stride(-1),
        states.stride(0),
        states.stride(1),
        states.stride(2),
        states.stride(3),
        states.stride(4),
        scale_precomputed.stride(0),   # batch stride
        scale_precomputed.stride(2),   # chunk stride (dim=2)
        scale_precomputed.stride(1),   # head stride (dim=1)
        scale_precomputed.stride(3),   # csize stride (dim=3)
        seq_idx.stride(0) if seq_idx is not None else 0,
        seq_idx.stride(1) if seq_idx is not None else 0,
        HAS_SEQ_IDX=seq_idx is not None,
        BLOCK_SIZE_N=BLOCK_SIZE_N_val,
        USE_MANUAL_REDUCTION=USE_MANUAL_REDUCTION,
    )
    return states