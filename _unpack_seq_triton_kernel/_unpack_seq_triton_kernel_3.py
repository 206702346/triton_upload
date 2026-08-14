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
    pid = tl.program_id(0)
    if pid >= B:
        return

    cum_len_start = tl.load(cum_lengths_ptr + pid)
    cum_len_end = tl.load(cum_lengths_ptr + pid + 1)
    seq_len = cum_len_end - cum_len_start
    if seq_len == 0:
        return

    # packed tensor is transposed to [Lmax, B, D] before kernel launch
    # offset = t * B * D + b * D + d
    for t_start in range(0, seq_len, BLOCK_T):
        t_idx = t_start + tl.arange(0, BLOCK_T)
        t_mask = t_idx < seq_len
        for d_start in range(0, D, BLOCK_D):
            d_idx = d_start + tl.arange(0, BLOCK_D)
            d_mask = d_idx < D
            mask = t_mask[:, None] & d_mask[None, :]

            # packed offset using transposed layout: [Lmax, B, D]
            packed_offset = (t_idx[:, None] * B * D +
                             pid * D +
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

    Strategy: transpose packed tensor to [Lmax, B, D] before kernel launch
    to improve cache locality for the time-step dimension.

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

    # Transpose packed tensor to [Lmax, B, D] for better locality
    packed_transposed = packed_reshaped.permute(1, 0, 2).contiguous()

    N = int(lengths.sum().item())
    out = torch.empty((N, D), device=packed_tensor.device, dtype=packed_tensor.dtype)

    cum_lengths = torch.zeros(B + 1, dtype=torch.int32, device=packed_tensor.device)
    torch.cumsum(lengths.int(), dim=0, out=cum_lengths[1:])

    grid = (B,)
    _unpack_seq_triton_kernel[grid](
        packed_transposed,
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