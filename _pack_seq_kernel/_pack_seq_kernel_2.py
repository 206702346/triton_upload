import torch
import triton
import triton.language as tl

@triton.jit
def _pack_seq_kernel(
    x_ptr,
    out_ptr,
    starts_ptr,
    lengths_ptr,
    N: tl.constexpr,
    D: tl.constexpr,
    Lmax: tl.constexpr,
    PAD_VALUE: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    total_t_blocks = tl.cdiv(Lmax, BLOCK_T)
    batch_idx = pid // total_t_blocks
    t_block_idx = pid % total_t_blocks

    off_t = t_block_idx * BLOCK_T + tl.arange(0, BLOCK_T)
    t_mask = off_t < Lmax

    seq_len = tl.load(lengths_ptr + batch_idx)
    valid_row = (off_t < seq_len) & t_mask
    in_start = tl.load(starts_ptr + batch_idx)
    in_row = in_start + off_t

    # Loop over the feature dimension to reduce grid launch overhead
    for d_start in range(0, D, BLOCK_D):
        off_d = d_start + tl.arange(0, BLOCK_D)
        d_mask = off_d < D
        load_mask = valid_row[:, None] & d_mask[None, :]

        x_row_ptr = x_ptr + in_row[:, None] * D + off_d[None, :]
        out_row_ptr = out_ptr + (batch_idx * Lmax + off_t)[:, None] * D + off_d[None, :]

        x_vals = tl.load(x_row_ptr, mask=load_mask, other=PAD_VALUE)
        store_mask = t_mask[:, None] & d_mask[None, :]
        tl.store(out_row_ptr, x_vals, mask=store_mask)


def pack_seq_triton(
    x: torch.Tensor,
    lengths: torch.Tensor,
    pad_value: float = -float("inf"),
    block_t: int = 64,
    block_d: int = 64,
    use_precomputed_starts: bool = True,
) -> torch.Tensor:
    """
    Pack sequences of different lengths into a batched tensor.

    Args:
        x: [N, ...] - input tensor where N is total number of tokens
        lengths: [B] - sequence lengths for each batch
        pad_value: value to use for padding
        block_t: block size for time dimension (ignored, tuned internally)
        block_d: block size for feature dimension (ignored, tuned internally)
        use_precomputed_starts: whether to compute start indices outside kernel

    Returns:
        packed: [B, Lmax, ...] - packed tensor
    """
    original_shape = x.shape
    if len(original_shape) > 2:
        N = original_shape[0]
        x_reshaped = x.reshape(N, -1)
        D = x_reshaped.shape[1]
    else:
        N, D = x.shape
        x_reshaped = x

    B = lengths.numel()
    Lmax = int(lengths.max().item())

    out = torch.empty((B, Lmax, D), device=x.device, dtype=x.dtype)

    if use_precomputed_starts:
        starts = torch.zeros_like(lengths)
        if B > 1:
            starts[1:] = torch.cumsum(lengths[:-1], dim=0)
        starts = starts.int()
    else:
        starts = torch.zeros_like(lengths, dtype=torch.int32)

    # Tuned block sizes: reduce grid dimensions while keeping good occupancy
    tuned_block_t = min(256, Lmax)
    # Ensure block_t is at least 32 for reasonable thread usage
    if tuned_block_t < 32:
        tuned_block_t = Lmax  # fallback to full length for tiny Lmax
    tuned_block_d = min(256, D)
    if tuned_block_d < 32:
        tuned_block_d = D

    num_t_blocks = triton.cdiv(Lmax, tuned_block_t)
    grid = (B * num_t_blocks, 1, 1)  # 1D grid to reduce launch overhead

    _pack_seq_kernel[grid](
        x_reshaped,
        out,
        starts.int(),
        lengths.int(),
        N,
        D,
        Lmax,
        PAD_VALUE=float(pad_value),
        BLOCK_T=tuned_block_t,
        BLOCK_D=tuned_block_d,
        num_warps=8,
        num_stages=3,  # software pipelining over D loop
    )

    if len(original_shape) > 2:
        output_shape = (B, Lmax) + original_shape[1:]
        out = out.reshape(output_shape)

    return out