import torch
import torch_npu
import triton
import triton.language as tl


@triton.jit
def _log_softmax_kernel(
    input_ptr,
    output_ptr,
    input_row_stride,
    output_row_stride,
    n_cols: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Online log_softmax with stable FP32 accumulation.
    Each program handles one row, tiling along columns.
    Uses two passes: the first computes the online max and log-sum-exp,
    the second writes the final log-softmax outputs.
    """
    row_idx = tl.program_id(0).to(tl.int64)
    row_start_ptr = input_ptr + row_idx * input_row_stride
    output_row_start_ptr = output_ptr + row_idx * output_row_stride

    # Accumulators in float32 for numerical stability
    m_i = tl.full((), -float("inf"), dtype=tl.float32)
    l_i = tl.zeros((), dtype=tl.float32)

    # First pass: online max and sum
    for col_offset in range(0, n_cols, BLOCK_SIZE):
        col_idx = col_offset + tl.arange(0, BLOCK_SIZE)
        mask = col_idx < n_cols

        # Load, cast to float32, out-of-bounds elements become -inf
        vals = tl.load(row_start_ptr + col_idx, mask=mask, other=-float("inf")).to(tl.float32)

        block_max = tl.max(vals, axis=0)
        m_new = tl.maximum(m_i, block_max)

        # Re-scale running sum with the new max
        alpha = tl.exp(m_i - m_new)
        l_i = l_i * alpha + tl.sum(tl.exp(vals - m_new), axis=0)
        m_i = m_new

    log_sum_exp = tl.log(l_i) + m_i

    # Second pass: write final results
    for col_offset in range(0, n_cols, BLOCK_SIZE):
        col_idx = col_offset + tl.arange(0, BLOCK_SIZE)
        mask = col_idx < n_cols

        vals = tl.load(row_start_ptr + col_idx, mask=mask, other=-float("inf")).to(tl.float32)
        output = vals - log_sum_exp
        tl.store(output_row_start_ptr + col_idx, output, mask=mask)


def log_softmax(input: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Compute log_softmax along the last dimension using a Triton kernel.
    Handles arbitrary last-dimension lengths without padding,
    uses FP32 accumulation for numerical stability.
    """
    if dim != -1 and dim != input.ndim - 1:
        raise ValueError(
            "This implementation only supports log_softmax along the last dimension"
        )

    if input.device.type != 'npu':
        input = input.to('npu')

    original_shape = input.shape
    input_2d = input.reshape(-1, input.shape[-1])
    input_2d = input_2d.contiguous()

    n_rows, n_cols = input_2d.shape
    if n_cols == 0:
        raise ValueError("Input tensor cannot have empty last dimension")

    # Tune BLOCK_SIZE for long rows (8192+ columns)
    if n_cols >= 8192:
        BLOCK_SIZE = 8192
    elif n_cols >= 4096:
        BLOCK_SIZE = 4096
    elif n_cols >= 2048:
        BLOCK_SIZE = 2048
    else:
        # For shorter rows use a smaller power-of-two tile
        BLOCK_SIZE = min(1024, triton.next_power_of_2(n_cols))

    output = torch.empty_like(input_2d, device='npu')
    grid = (n_rows,)
    _log_softmax_kernel[grid](
        input_2d,
        output,
        input_2d.stride(0),
        output.stride(0),
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return output.reshape(original_shape)