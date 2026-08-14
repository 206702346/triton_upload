import torch
import triton
import triton.language as tl


@triton.jit
def _set_k_and_s_triton_kernel(
    buf_fp8_ptr,
    buf_fp32_ptr,
    loc_ptr,
    index_k_ptr,
    index_k_scale_ptr,
    index_k_ptr_stride_0,
    PAGE_SIZE: tl.constexpr,
    BUF_NUMEL_PER_PAGE: tl.constexpr,
    NUM_K_ELEMS_PER_TOKEN: tl.constexpr,
    S_OFFSET_NBYTES_IN_PAGE: tl.constexpr,
):
    """
    Each program writes one token into its destination page and offset.
    Bitwise shift / mask are used for fast address decomposition
    (PAGE_SIZE == 64, which is a power of two).
    """
    token_id = tl.program_id(0)

    loc = tl.load(loc_ptr + token_id)

    in_k_offsets = token_id * index_k_ptr_stride_0 + tl.arange(0, NUM_K_ELEMS_PER_TOKEN)
    # no mask needed because both loads are whole vectors and aligned
    k = tl.load(index_k_ptr + in_k_offsets, eviction_policy="evict_last")
    k_scale = tl.load(index_k_scale_ptr + token_id, eviction_policy="evict_last")

    # PAGE_SIZE is guaranteed to be 64 -> // 64 = >> 6, % 64 = & 63
    loc_page_index = loc >> 6             # loc // PAGE_SIZE
    loc_token_offset_in_page = loc & 63   # loc % PAGE_SIZE

    out_k_offsets = (
        loc_page_index * BUF_NUMEL_PER_PAGE
        + loc_token_offset_in_page * NUM_K_ELEMS_PER_TOKEN
        + tl.arange(0, NUM_K_ELEMS_PER_TOKEN)
    )

    # s offset is after all the k data in the page (fp32 scale stored interleaved)
    out_s_offset = (
        loc_page_index * BUF_NUMEL_PER_PAGE // 4
        + S_OFFSET_NBYTES_IN_PAGE // 4
        + loc_token_offset_in_page
    )

    tl.store(buf_fp8_ptr + out_k_offsets, k)
    tl.store(buf_fp32_ptr + out_s_offset, k_scale)


def _set_k_and_s_triton(
    buf: torch.Tensor,
    loc: torch.Tensor,
    index_k: torch.Tensor,
    index_k_scale: torch.Tensor,
    page_size: int,
):
    """
    Scatter tokens into the page buffer.

    :param buf: (num_pages, page_size 64 * (128B data + 4B scale)), uint8
    :param loc: (num_tokens_to_write,), int64, destination token index
    :param index_k: (num_tokens_to_write, 128 elem), fp16 (holding fp8 values)
    :param index_k_scale: (num_tokens_to_write,) or (num_tokens_to_write,1), fp32
    :param page_size: must be 64 (power of two asserted)
    """
    num_pages, buf_numel_per_page = buf.shape
    (num_tokens_to_write,) = loc.shape
    num_tokens_to_write_, index_head_dim = index_k.shape

    # Handle both 1D (num_tokens,) and 2D (num_tokens, 1) shapes for index_k_scale
    if index_k_scale.ndim == 1:
        num_tokens_to_write__ = index_k_scale.shape[0]
        scale_dim = 1
    elif index_k_scale.ndim == 2:
        num_tokens_to_write__, scale_dim = index_k_scale.shape
    else:
        raise ValueError(
            f"index_k_scale must be 1D or 2D, got shape {index_k_scale.shape}"
        )

    assert buf_numel_per_page == page_size * (128 + 4)   # 64 * (128 + 4) = 8448
    assert num_tokens_to_write == num_tokens_to_write_ == num_tokens_to_write__
    assert index_head_dim == 128
    assert scale_dim == 1
    assert page_size == 64                                # power‑of‑two guarantee

    assert buf.dtype == torch.uint8
    assert loc.dtype == torch.int64
    assert index_k.dtype == torch.float16
    assert index_k_scale.dtype == torch.float32

    assert buf.is_contiguous()
    assert loc.is_contiguous()
    assert index_k.is_contiguous()
    assert index_k_scale.is_contiguous()

    # View the buffer in appropriate dtypes for the kernel's pointers
    buf_fp16 = buf.view(torch.float16)      # used as buf_fp8_ptr for fp8 data
    buf_fp32 = buf.view(torch.float32)      # used as buf_fp32_ptr for scale

    _set_k_and_s_triton_kernel[(num_tokens_to_write,)](
        buf_fp16,
        buf_fp32,
        loc,
        index_k,
        index_k_scale,
        index_k.stride(0),
        PAGE_SIZE=page_size,
        BUF_NUMEL_PER_PAGE=buf_numel_per_page,
        NUM_K_ELEMS_PER_TOKEN=index_head_dim,
        S_OFFSET_NBYTES_IN_PAGE=page_size * index_head_dim,
    )