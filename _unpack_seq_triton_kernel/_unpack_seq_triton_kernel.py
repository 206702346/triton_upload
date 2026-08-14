import torch
import triton
import triton.language as tl

@triton.jit
def _unpack_seq_triton_kernel(
    packed_ptr,
    out_ptr,
    lengths_ptr,
    B: tl.constexpr,
    Lmax: tl.constexpr,
    D: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_t = tl.program_id(1)
    pid_d = tl.program_id(2)
    off_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    off_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)

    in_start = 0
    for i in range(pid_b):
        in_start += tl.load(lengths_ptr + i)
    seq_len = tl.load(lengths_ptr + pid_b)

    t_mask = off_t < Lmax
    valid_row = (off_t < seq_len) & t_mask

    out_row = in_start + off_t

    packed_row_ptr = packed_ptr + (pid_b * Lmax + off_t)[:, None] * D + off_d[None, :]

    out_row_ptr = out_ptr + out_row[:, None] * D + off_d[None, :]

    d_mask = off_d[None, :] < D
    packed_vals = tl.load(packed_row_ptr, mask=valid_row[:, None] & d_mask)
    tl.store(out_row_ptr, packed_vals, mask=valid_row[:, None] & d_mask)

def unpack_seq_triton(
    packed_tensor: torch.Tensor,
    lengths: torch.Tensor,
    block_t: int = 64,
    block_d: int = 64,
) -> torch.Tensor:
    """
    Unpack a packed decode query tensor back to the original format.
    Efficient Triton implementation.

    Args:
        packed_tensor: [B, Lmax, ...] - packed tensor from pack_seq_triton
        lengths: [B] - sequence lengths for each batch
        block_t: block size for time dimension
        block_d: block size for feature dimension

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

    grid = (B, triton.cdiv(Lmax, block_t), triton.cdiv(D, block_d))
    _unpack_seq_triton_kernel[grid](
        packed_reshaped,
        out,
        lengths.int(),
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
