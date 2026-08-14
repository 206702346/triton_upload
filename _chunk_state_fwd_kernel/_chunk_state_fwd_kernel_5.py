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
    x_ptr,
    b_ptr,
    states_ptr,
    scale_ptr,
    seq_idx_ptr,
    hdim,
    dstate,
    chunk_size,
    batch,
    seqlen,
    nheads_ngroups_ratio,
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
    HAS_SEQ_IDX: tl.constexpr = False,
    BLOCK_SIZE_M: tl.constexpr = 16,
    BLOCK_SIZE_N: tl.constexpr = 16,
    BLOCK_SIZE_K: tl.constexpr = 16,
    USE_MANUAL_REDUCTION: tl.constexpr = False,
):
    pid_bc = tl.program_id(axis=1).to(tl.int64)
    pid_c = pid_bc // batch
    pid_b = pid_bc - pid_c * batch
    pid_h = tl.program_id(axis=2)

    num_pid_n = tl.cdiv(dstate, BLOCK_SIZE_N)
    pid_m = tl.program_id(axis=0) // num_pid_n
    pid_n = tl.program_id(axis=0) % num_pid_n

    chunk_size_limit = min(chunk_size, seqlen - pid_c * chunk_size)

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

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, chunk_size_limit, BLOCK_SIZE_K):
        x = tl.load(x_block_ptr, boundary_check=(0, 1), padding_option="zero")
        b = tl.load(b_block_ptr, boundary_check=(0, 1), padding_option="zero").to(
            tl.float32
        )
        scale = tl.load(scale_block_ptr, boundary_check=(0,), padding_option="zero").to(
            tl.float32
        )

        if HAS_SEQ_IDX:
            seq_idx_k = tl.load(
                seq_idx_block_ptr, boundary_check=(0,), padding_option="zero"
            )
            scale = tl.where(seq_idx_k == seq_idx_last, scale, 0.0)

        if USE_MANUAL_REDUCTION:
            b *= scale[:, None]
            x_t = tl.trans(x).to(tl.float32)
            prod = x_t[:, :, None] * b[None, :, :]
            acc += tl.sum(prod, axis=1)
        else:
            # Fused multiply‑scale and cast to input dtype
            b = (scale[:, None] * b).to(x_ptr.dtype.element_ty)
            x_t = tl.trans(x)
            acc += tl.dot(x_t, b)

        x_block_ptr = tl.advance(x_block_ptr, (BLOCK_SIZE_K, 0))
        b_block_ptr = tl.advance(b_block_ptr, (BLOCK_SIZE_K, 0))
        scale_block_ptr = tl.advance(scale_block_ptr, (BLOCK_SIZE_K,))
        if HAS_SEQ_IDX:
            seq_idx_block_ptr = tl.advance(seq_idx_block_ptr, (BLOCK_SIZE_K,))

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


@triton.jit
def _chunk_state_fwd_kernel_ksplit_atomic(
    x_ptr,
    b_ptr,
    states_ptr,
    scale_ptr,
    seq_idx_ptr,
    hdim,
    dstate,
    chunk_size,
    batch,
    seqlen,
    nheads_ngroups_ratio,
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
    num_ksplits,
    HAS_SEQ_IDX: tl.constexpr = False,
    BLOCK_SIZE_M: tl.constexpr = 16,
    BLOCK_SIZE_N: tl.constexpr = 16,
    BLOCK_SIZE_K: tl.constexpr = 16,
    USE_MANUAL_REDUCTION: tl.constexpr = False,
):
    pid_bc = tl.program_id(axis=1).to(tl.int64)
    pid_c = pid_bc // batch
    pid_b = pid_bc - pid_c * batch
    pid_h = tl.program_id(axis=2)

    num_pid_m = tl.cdiv(hdim, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(dstate, BLOCK_SIZE_N)
    num_tiles_per_chunk = num_pid_m * num_pid_n

    pid_flat = tl.program_id(axis=0).to(tl.int64)
    pid_k = pid_flat // num_tiles_per_chunk
    pid_tile = pid_flat % num_tiles_per_chunk

    pid_m = pid_tile // num_pid_n
    pid_n = pid_tile % num_pid_n

    chunk_size_limit = min(chunk_size, seqlen - pid_c * chunk_size)

    # determine K-slice boundaries
    k_slice_size = tl.cdiv(chunk_size_limit, num_ksplits)
    k_start = pid_k * k_slice_size
    k_end = tl.minimum(k_start + k_slice_size, chunk_size_limit)

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

    x_block_ptr = tl.make_block_ptr(
        base=x_base,
        shape=(chunk_size_limit, hdim),
        strides=(stride_x_seqlen, stride_x_hdim),
        offsets=(k_start, pid_m * BLOCK_SIZE_M),
        block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_M),
        order=(1, 0),
    )
    b_block_ptr = tl.make_block_ptr(
        base=b_base,
        shape=(chunk_size_limit, dstate),
        strides=(stride_b_seqlen, stride_b_dstate),
        offsets=(k_start, pid_n * BLOCK_SIZE_N),
        block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_N),
        order=(1, 0),
    )
    scale_block_ptr = tl.make_block_ptr(
        base=scale_base,
        shape=(chunk_size_limit,),
        strides=(stride_scale_csize,),
        offsets=(k_start,),
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
            offsets=(k_start,),
            block_shape=(BLOCK_SIZE_K,),
            order=(0,),
        )
        seq_idx_last = tl.load(
            seq_idx_ptr
            + pid_b * stride_seq_idx_batch
            + (pid_c * chunk_size + chunk_size_limit - 1) * stride_seq_idx_seqlen
        )

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    k_slice_len = k_end - k_start
    num_k_loops = tl.cdiv(k_slice_len, BLOCK_SIZE_K)  # static iteration count

    for _ in range(num_k_loops):
        x = tl.load(x_block_ptr, boundary_check=(0, 1), padding_option="zero")
        b = tl.load(b_block_ptr, boundary_check=(0, 1), padding_option="zero").to(
            tl.float32
        )
        scale = tl.load(scale_block_ptr, boundary_check=(0,), padding_option="zero").to(
            tl.float32
        )

        if HAS_SEQ_IDX:
            seq_idx_k = tl.load(
                seq_idx_block_ptr, boundary_check=(0,), padding_option="zero"
            )
            scale = tl.where(seq_idx_k == seq_idx_last, scale, 0.0)

        if USE_MANUAL_REDUCTION:
            b *= scale[:, None]
            x_t = tl.trans(x).to(tl.float32)
            prod = x_t[:, :, None] * b[None, :, :]
            acc += tl.sum(prod, axis=1)
        else:
            # Fused multiply‑scale and cast to input dtype
            b = (scale[:, None] * b).to(x_ptr.dtype.element_ty)
            x_t = tl.trans(x)
            acc += tl.dot(x_t, b)

        x_block_ptr = tl.advance(x_block_ptr, (BLOCK_SIZE_K, 0))
        b_block_ptr = tl.advance(b_block_ptr, (BLOCK_SIZE_K, 0))
        scale_block_ptr = tl.advance(scale_block_ptr, (BLOCK_SIZE_K,))
        if HAS_SEQ_IDX:
            seq_idx_block_ptr = tl.advance(seq_idx_block_ptr, (BLOCK_SIZE_K,))

    # atomic-add partial sum to states (must be pre-zeroed)
    acc_val = acc.to(states_ptr.dtype.element_ty)
    for i in tl.static_range(BLOCK_SIZE_M):
        for j in tl.static_range(BLOCK_SIZE_N):
            m_off = pid_m * BLOCK_SIZE_M + i
            n_off = pid_n * BLOCK_SIZE_N + j
            if m_off < hdim and n_off < dstate:
                ptr = states_base + m_off * stride_states_hdim + n_off * stride_states_dstate
                tl.atomic_add(ptr, acc_val[i, j])


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

    # Heuristic: use K-split atomic kernel only for large chunk_size
    KSPLIT_THRESHOLD = 128
    BLOCK_K_ATOMIC = 64
    MAX_KSPLIT = 8

    if chunk_size >= KSPLIT_THRESHOLD:
        # atomic split path
        num_ksplits = max(1, triton.cdiv(chunk_size, BLOCK_K_ATOMIC))
        num_ksplits = min(num_ksplits, MAX_KSPLIT)

        # allocate/zero states
        if states is not None:
            states.zero_()
        else:
            states_dtype = torch.float32 if states_in_fp32 else B.dtype
            states = torch.zeros(
                (batch, nchunks, nheads, headdim, dstate),
                device=B.device,
                dtype=states_dtype,
            )
    else:
        # original path: no atomic, no need to zero
        if states is not None:
            pass  # will be overwritten completely
        else:
            states_dtype = torch.float32 if states_in_fp32 else B.dtype
            states = torch.empty(
                (batch, nchunks, nheads, headdim, dstate),
                device=B.device,
                dtype=states_dtype,
            )

    # common precomputation and reordering
    dA_cumsum_last = dA_cumsum[..., -1:]
    scale_precomputed = torch.exp(dA_cumsum_last - dA_cumsum) * dt
    scale_precomputed = scale_precomputed.to(dt.dtype)

    x_reorder = x.permute(0, 2, 1, 3).reshape(
        batch, nheads, nchunks, chunk_size, headdim
    ).contiguous()
    B_reorder = B.permute(0, 2, 1, 3).reshape(
        batch, ngroups, nchunks, chunk_size, dstate
    ).contiguous()
    if seq_idx is not None:
        seq_idx_reorder = seq_idx.reshape(
            batch, nchunks, chunk_size
        ).contiguous()
    else:
        seq_idx_reorder = None

    stride_x_batch_new = x_reorder.stride(0)
    stride_x_head_new = x_reorder.stride(1)
    stride_x_seqlen_new = x_reorder.stride(3)
    stride_x_hdim_new = x_reorder.stride(4)

    stride_b_batch_new = B_reorder.stride(0)
    stride_b_head_new = B_reorder.stride(1)
    stride_b_seqlen_new = B_reorder.stride(3)
    stride_b_dstate_new = B_reorder.stride(4)

    if seq_idx_reorder is not None:
        stride_seq_idx_batch_new = seq_idx_reorder.stride(0)
        stride_seq_idx_seqlen_new = seq_idx_reorder.stride(2)
    else:
        stride_seq_idx_batch_new = 0
        stride_seq_idx_seqlen_new = 0

    # choose block sizes (same heuristic as parent)
    if dstate < 16:
        USE_MANUAL_REDUCTION = True
        BLOCK_SIZE_M_val = 16
        BLOCK_SIZE_N_val = triton.next_power_of_2(dstate)
    else:
        USE_MANUAL_REDUCTION = False
        if headdim >= 64:
            BLOCK_SIZE_M_val = 64
        elif headdim >= 32:
            BLOCK_SIZE_M_val = 32
        else:
            BLOCK_SIZE_M_val = 16
        BLOCK_SIZE_N_val = 32 if dstate > 16 else 16

    if USE_MANUAL_REDUCTION:
        BLOCK_SIZE_K_val = 16
    elif x.dtype == torch.float32:
        BLOCK_SIZE_K_val = 16
    elif chunk_size >= 32:
        BLOCK_SIZE_K_val = 32
    else:
        BLOCK_SIZE_K_val = 16

    num_warps_val = 8 if BLOCK_SIZE_M_val * BLOCK_SIZE_N_val >= 1024 else 4

    if chunk_size >= KSPLIT_THRESHOLD:
        # launch atomic split kernel
        num_tiles = triton.cdiv(headdim, BLOCK_SIZE_M_val) * triton.cdiv(
            dstate, BLOCK_SIZE_N_val
        )
        grid = lambda META: (
            num_ksplits * num_tiles,
            batch * nchunks,
            nheads,
        )
        _chunk_state_fwd_kernel_ksplit_atomic[grid](
            x_reorder,
            B_reorder,
            states,
            scale_precomputed,
            seq_idx_reorder if seq_idx_reorder is not None else torch.empty(0, device=x.device),
            headdim,
            dstate,
            chunk_size,
            batch,
            seqlen,
            nheads // ngroups,
            stride_x_batch_new,
            stride_x_seqlen_new,
            stride_x_head_new,
            stride_x_hdim_new,
            stride_b_batch_new,
            stride_b_seqlen_new,
            stride_b_head_new,
            stride_b_dstate_new,
            states.stride(0),
            states.stride(1),
            states.stride(2),
            states.stride(3),
            states.stride(4),
            scale_precomputed.stride(0),
            scale_precomputed.stride(2),
            scale_precomputed.stride(1),
            scale_precomputed.stride(3),
            stride_seq_idx_batch_new,
            stride_seq_idx_seqlen_new,
            num_ksplits,
            HAS_SEQ_IDX=seq_idx_reorder is not None,
            BLOCK_SIZE_M=BLOCK_SIZE_M_val,
            BLOCK_SIZE_N=BLOCK_SIZE_N_val,
            BLOCK_SIZE_K=BLOCK_SIZE_K_val,
            USE_MANUAL_REDUCTION=USE_MANUAL_REDUCTION,
            num_warps=num_warps_val,
        )
    else:
        # original kernel (non‑atomic)
        grid = lambda META: (
            triton.cdiv(headdim, META["BLOCK_SIZE_M"])
            * triton.cdiv(dstate, META["BLOCK_SIZE_N"]),
            batch * nchunks,
            nheads,
        )
        _chunk_state_fwd_kernel[grid](
            x_reorder,
            B_reorder,
            states,
            scale_precomputed,
            seq_idx_reorder if seq_idx_reorder is not None else torch.empty(0, device=x.device),
            headdim,
            dstate,
            chunk_size,
            batch,
            seqlen,
            nheads // ngroups,
            stride_x_batch_new,
            stride_x_seqlen_new,
            stride_x_head_new,
            stride_x_hdim_new,
            stride_b_batch_new,
            stride_b_seqlen_new,
            stride_b_head_new,
            stride_b_dstate_new,
            states.stride(0),
            states.stride(1),
            states.stride(2),
            states.stride(3),
            states.stride(4),
            scale_precomputed.stride(0),
            scale_precomputed.stride(2),
            scale_precomputed.stride(1),
            scale_precomputed.stride(3),
            stride_seq_idx_batch_new,
            stride_seq_idx_seqlen_new,
            HAS_SEQ_IDX=seq_idx_reorder is not None,
            BLOCK_SIZE_M=BLOCK_SIZE_M_val,
            BLOCK_SIZE_N=BLOCK_SIZE_N_val,
            BLOCK_SIZE_K=BLOCK_SIZE_K_val,
            USE_MANUAL_REDUCTION=USE_MANUAL_REDUCTION,
            num_warps=num_warps_val,
        )
    return states