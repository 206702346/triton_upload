import torch
import triton
import triton.language as tl


@triton.jit
def _rms_norm_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
    input_row_stride,
    output_row_stride,
    n_rows,
    n_cols,
    eps,
    BLOCK_SIZE: tl.constexpr,
    ROWS_PER_BLOCK: tl.constexpr,
):
    """
    Multi‑row RMS normalization with fused single‑pass for rows fitting in one block.
    Uses warp‑level reduction via triton's sum (tree reduction) and explicit masking
    to meet precision and mask invariants.  EPS is placed inside the sqrt, after mean_sq.
    """
    pid = tl.program_id(0)
    row_start = pid * ROWS_PER_BLOCK
    row_ids = row_start + tl.arange(0, ROWS_PER_BLOCK)
    row_mask = row_ids < n_rows

    if n_cols <= BLOCK_SIZE:
        # ---- Fused single‑pass path ----
        col_offs = tl.arange(0, BLOCK_SIZE)
        col_mask = col_offs < n_cols
        tile = tl.load(
            input_ptr + row_ids[:, None] * input_row_stride + col_offs[None, :],
            mask=row_mask[:, None] & col_mask[None, :],
            other=0.0,
        )
        tile_f32 = tile.to(tl.float32)

        # Warp‑level sum of squares (triton reduces across threads in hardware)
        sq = tile_f32 * tile_f32
        # Use explicit where to respect column mask – safer for any accumulation edge cases
        sq = tl.where(col_mask[None, :], sq, 0.0)
        sum_sq = tl.sum(sq, axis=1)

        # EPS inside sqrt, after mean
        mean_sq = sum_sq / n_cols
        inv_rms = tl.math.rsqrt(mean_sq + eps)

        # Weight broadcast and combine
        weight = tl.load(weight_ptr + col_offs, mask=col_mask, other=1.0)
        weight_f32 = weight.to(tl.float32)
        out_f32 = tile_f32 * inv_rms[:, None] * weight_f32[None, :]
        out = out_f32.to(tile.dtype)

        tl.store(
            output_ptr + row_ids[:, None] * output_row_stride + col_offs[None, :],
            out,
            mask=row_mask[:, None] & col_mask[None, :],
        )
    else:
        # ---- Two‑pass fallback for large rows ----
        sum_sq = tl.zeros((ROWS_PER_BLOCK,), dtype=tl.float32)

        # First pass: accumulate squares over column blocks
        for col_start in range(0, n_cols, BLOCK_SIZE):
            col_offs = col_start + tl.arange(0, BLOCK_SIZE)
            col_mask = col_offs < n_cols

            tile = tl.load(
                input_ptr + row_ids[:, None] * input_row_stride + col_offs[None, :],
                mask=row_mask[:, None] & col_mask[None, :],
                other=0.0,
            )
            tile_f32 = tile.to(tl.float32)
            sq = tile_f32 * tile_f32
            sq = tl.where(col_mask[None, :], sq, 0.0)
            sum_sq += tl.sum(sq, axis=1)

        mean_sq = sum_sq / n_cols
        inv_rms = tl.math.rsqrt(mean_sq + eps)

        # Second pass: apply norm and weight
        for col_start in range(0, n_cols, BLOCK_SIZE):
            col_offs = col_start + tl.arange(0, BLOCK_SIZE)
            col_mask = col_offs < n_cols

            tile = tl.load(
                input_ptr + row_ids[:, None] * input_row_stride + col_offs[None, :],
                mask=row_mask[:, None] & col_mask[None, :],
                other=0.0,
            )
            weight = tl.load(weight_ptr + col_offs, mask=col_mask, other=1.0)

            tile_f32 = tile.to(tl.float32)
            weight_f32 = weight.to(tl.float32)

            out_f32 = tile_f32 * inv_rms[:, None] * weight_f32[None, :]
            out = out_f32.to(tile.dtype)

            tl.store(
                output_ptr + row_ids[:, None] * output_row_stride + col_offs[None, :],
                out,
                mask=row_mask[:, None] & col_mask[None, :],
            )


def rms_norm(
    input: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    """Apply RMSNorm along the last dimension of `input` with learnable `weight`."""
    assert weight.dim() == 1, "Weight must be 1‑dimensional"
    assert input.shape[-1] == weight.shape[0], (
        f"Input last dimension ({input.shape[-1]}) must match "
        f"weight dimension ({weight.shape[0]})"
    )

    original_shape = input.shape
    input_2d = input.reshape(-1, input.shape[-1]).contiguous()
    weight = weight.contiguous()

    n_rows, n_cols = input_2d.shape
    if n_rows == 0:
        return input_2d.reshape(original_shape)

    output = torch.empty_like(input_2d)

    # Choose BLOCK_SIZE: smallest power of two ≥ n_cols, clamped to [64, 4096]
    blk = 1
    while blk < n_cols:
        blk *= 2
    BLOCK_SIZE = min(max(blk, 64), 4096)

    # Dynamic ROWS_PER_BLOCK to maximise occupancy and weight reuse
    if n_cols <= 256:
        ROWS_PER_BLOCK = min(8, triton.cdiv(n_rows, triton.cdiv(n_rows, 8)))
    elif n_cols <= 512:
        ROWS_PER_BLOCK = 6
    else:
        ROWS_PER_BLOCK = 4

    grid = (triton.cdiv(n_rows, ROWS_PER_BLOCK),)

    _rms_norm_kernel[grid](
        input_2d,
        weight,
        output,
        input_2d.stride(0),
        output.stride(0),
        n_rows,
        n_cols,
        eps,
        BLOCK_SIZE=BLOCK_SIZE,
        ROWS_PER_BLOCK=ROWS_PER_BLOCK,
        num_warps=16,
        num_stages=2,
    )

    return output.reshape(original_shape)


def rms_norm_batch_invariant(
    input: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    """
    Batch‑invariant RMS normalization: identical semantics and performance
    regardless of the leading batch dimensions.
    """
    return rms_norm(input, weight, eps=eps)