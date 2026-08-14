import torch
import triton
import triton.language as tl

@triton.jit
def _unpack_seq_triton_kernel(
    packed_ptr,
    out_ptr,
    cum_lengths_ptr,
    B: tl.constexpr,
    Lmax: tl.constexpr,
    D: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_batch = tl.program_id(0)
    pid_tblock = tl.program_id(1)
    pid_dblock = tl.program_id(2)

    if pid_batch >= B:
        return

    t_start = pid_tblock * BLOCK_T
    d_start = pid_dblock * BLOCK_D

    # Skip blocks beyond the actual tensor extent
    if t_start >= Lmax:
        return

    cum_len_start = tl.load(cum_lengths_ptr + pid_batch)
    cum_len_end = tl.load(cum_lengths_ptr + pid_batch + 1)
    seq_len = cum_len_end - cum_len_start

    if seq_len == 0:
        return

    # No output elements in this block if the block starts at or after seq_len
    if t_start >= seq_len:
        return

    t_idx = t_start + tl.arange(0, BLOCK_T)
    d_idx = d_start + tl.arange(0, BLOCK_D)

    t_mask = t_idx < seq_len
    d_mask = d_idx < D
    mask = t_mask[:, None] & d_mask[None, :]

    packed_offset = (pid_batch * Lmax * D +
                     t_idx[:, None] * D +
                     d_idx[None, :])

    out_row = cum_len_start + t_idx
    out_offset = (out_row[:, None] * D + d_idx[None, :])

    packed_vals = tl.load(packed_ptr + packed_offset, mask=mask)
    tl.store(out_ptr + out_offset, packed_vals, mask=mask)


def unpack_seq_triton(
    packed_tensor: torch.Tensor,
    lengths: torch.Tensor,
    block_t: int = 32,
    block_d: int = 64,
) -> torch.Tensor:
    """
    Unpack a packed decode query tensor back to the original format.
    Optimized Triton implementation for Ascend NPU.

    This version uses a 2‑D launch grid over (batch, t_blocks, d_blocks) to
    increase parallelism and occupancy compared to the original nested‑loop
    approach.

    Args:
        packed_tensor: [B, Lmax, ...] - packed tensor from pack_seq_triton
        lengths: [B] - sequence lengths for each batch
        block_t: block size for time dimension (recommended: 32)
        block_d: block size for feature dimension (recommended: 64)

    Returns:
        unpacked_tensor: [N, ...] where N = sum(lengths)
    """

    original_shape = packed_tensor.shape
    if len(original_shape) > 3:
        B, Lmax = original_shape[:2]
        packed_reshaped = packed_tensor.reshape(B, Lmax, -1)
        D = packed_reshaped.shape[2]
    else:
        B, Lmax, D = packed_tensor.shape
        packed_reshaped = packed_tensor

    N = int(lengths.sum().item())

    out = torch.empty((N, D), device=packed_tensor.device, dtype=packed_tensor.dtype)

    cum_lengths = torch.zeros(B + 1, dtype=torch.int32, device=packed_tensor.device)
    torch.cumsum(lengths.int(), dim=0, out=cum_lengths[1:])

    t_blocks = (Lmax + block_t - 1) // block_t
    d_blocks = (D + block_d - 1) // block_d
    grid = (B, t_blocks, d_blocks)

    _unpack_seq_triton_kernel[grid](
        packed_reshaped,
        out,
        cum_lengths,
        B,
        Lmax,
        D,
        BLOCK_T=block_t,
        BLOCK_D=block_d,
        num_warps=4,
        num_stages=2,
    )

    if len(original_shape) > 3:
        output_shape = (N,) + original_shape[2:]
        out = out.reshape(output_shape)

    return out