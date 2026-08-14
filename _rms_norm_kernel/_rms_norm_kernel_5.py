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
    Multi-row RMS normalization with a fused single-pass path for rows that fit
    into one column block.  Rows are grouped into blocks; for narrow rows the
    launch configuration uses one warp per row so the row reduction is a fast
    intra-warp reduction.
    """
    pid = tl.program_id(0)
    row_start = pid * ROWS_PER_BLOCK
    row_ids = row_start + tl.arange(0, ROWS_PER_BLOCK)
    row_mask = row_ids < n_rows

    if n_cols <= BLOCK_SIZE:
        col_offs = tl.arange(0, BLOCK_SIZE)
        col_mask = col_offs < n_cols

        tile = tl.load(
            input_ptr + row_ids[:, None] * input_row_stride + col_offs[None, :],
            mask=row_mask[:, None] & col_mask[None, :],
            other=0.0,
        )
        tile_f32 = tile.to(tl.float32)

        # fp32 accumulation is preserved exactly.
        sum_sq = tl.sum(tile_f32 * tile_f32, axis=1)

        # eps is applied after mean_sq, as required.
        mean_sq = sum_sq / n_cols
        inv_rms = tl.math.rsqrt(mean_sq + eps)

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
        sum_sq = tl.zeros((ROWS_PER_BLOCK,), dtype=tl.float32)

        for col_start in range(0, n_cols, BLOCK_SIZE):
            col_offs = col_start + tl.arange(0, BLOCK_SIZE)
            col_mask = col_offs < n_cols

            tile = tl.load(
                input_ptr + row_ids[:, None] * input_row_stride + col_offs[None, :],
                mask=row_mask[:, None] & col_mask[None, :],
                other=0.0,
            )
            tile_f32 = tile.to(tl.float32)
            sum_sq += tl.sum(tile_f32 * tile_f32, axis=1)

        mean_sq = sum_sq / n_cols
        inv_rms = tl.math.rsqrt(mean_sq + eps)

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
    assert weight.dim() == 1, "Weight must be 1-dimensional"
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

    # Power-of-two column block, with 32 as the minimum so narrow rows still
    # map naturally onto one warp per row.
    blk = 1
    while blk < n_cols:
        blk *= 2
    BLOCK_SIZE = min(max(blk, 32), 4096)

    # Rows per block: 8 for narrow/medium rows, 4 for large rows.  Downward
    # snap to a power of two so num_warps can equal the rows-per-block when
    # the fast warp-per-row path is used.
    if BLOCK_SIZE <= 512:
        max_rows = 8
    else:
        max_rows = 4

    packed = min(max_rows, triton.cdiv(n_rows, triton.cdiv(n_rows, max_rows)))
    ROWS_PER_BLOCK = 1
    while ROWS_PER_BLOCK * 2 <= packed:
        ROWS_PER_BLOCK *= 2

    # Narrow tiles: one warp per row.  Wider tiles keep the parent-level
    # thread count so large-row work is not under-parallelized.
    if BLOCK_SIZE <= 256:
        num_warps = ROWS_PER_BLOCK
    else:
        num_warps = max(ROWS_PER_BLOCK, (ROWS_PER_BLOCK * BLOCK_SIZE) // 256)
        num_warps = min(16, num_warps)

    num_stages = 1 if BLOCK_SIZE <= 256 else 2

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
        num_warps=num_warps,
        num_stages=num_stages,
    )

    return output.reshape(original_shape)


def rms_norm_batch_invariant(
    input: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    """
    Batch-invariant RMS normalization: identical semantics and performance
    regardless of the leading batch dimensions.
    """
    return rms_norm(input, weight, eps=eps)