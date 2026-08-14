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
    Sorted scatter kernel – each program writes one token.
    The host sorts the inputs by target page index (not full loc)
    to group tokens belonging to the same page, enabling coalesced
    writes within each page.
    Uses bitwise shift and mask for fast address calculation.
    """
    token_id = tl.program_id(0)

    loc = tl.load(loc_ptr + token_id)

    in_k_offsets = token_id * index_k_ptr_stride_0 + tl.arange(0, NUM_K_ELEMS_PER_TOKEN)
    k = tl.load(index_k_ptr + in_k_offsets, eviction_policy="evict_last")
    k_scale = tl.load(index_k_scale_ptr + token_id, eviction_policy="evict_last")

    # Bitwise page index and token offset (PAGE_SIZE = 64 is a power of two)
    loc_page_index = loc >> 6             # loc // PAGE_SIZE
    loc_token_offset_in_page = loc & 63   # loc % PAGE_SIZE

    out_k_offsets = (
        loc_page_index * BUF_NUMEL_PER_PAGE
        + loc_token_offset_in_page * NUM_K_ELEMS_PER_TOKEN
        + tl.arange(0, NUM_K_ELEMS_PER_TOKEN)
    )

    # s offset is after all the k data in the page (fp32 scale stored interleaved)
    out_s_offset = (
        loc_page_index * (BUF_NUMEL_PER_PAGE // 4)   # page start in fp32 view
        + (S_OFFSET_NBYTES_IN_PAGE // 4)              # offset to scale region
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
    Semantic crossover offspring: combines the sorted scatter framework of
    Parent A (which sorted by full loc) with the insight that only page-level
    grouping is required for coalescing.  Here we sort tokens by their target
    **page index** (not the full destination index) to reduce sorting overhead
    while still achieving coalesced writes within each page.

    The kernel itself uses bitwise shift/mask arithmetic for address
    calculation (originally from Parent B).

    :param buf: (num_pages, page_size 64 * (128B k + 4B s per token)), uint8
    :param loc: (num_tokens_to_write,), int64, destination token index
    :param index_k: (num_tokens_to_write, 128 elements), fp16 (holding fp8 values)
    :param index_k_scale: (num_tokens_to_write,) or (num_tokens_to_write,1), fp32
    :param page_size: must be 64
    """
    num_pages, buf_numel_per_page = buf.shape
    (num_tokens_to_write,) = loc.shape
    num_tokens_to_write_, index_head_dim = index_k.shape

    # Handle both 1D (num_tokens,) and 2D (num_tokens,1) for index_k_scale
    if index_k_scale.ndim == 1:
        num_tokens_to_write__ = index_k_scale.shape[0]
        scale_dim = 1
    elif index_k_scale.ndim == 2:
        num_tokens_to_write__, scale_dim = index_k_scale.shape
    else:
        raise ValueError(
            f"index_k_scale must be 1D or 2D, got shape {index_k_scale.shape}"
        )

    assert buf_numel_per_page == page_size * (128 + 4)  # 64 * (128 + 4) = 8448
    assert num_tokens_to_write == num_tokens_to_write_ == num_tokens_to_write__
    assert index_head_dim == 128
    assert scale_dim == 1
    assert page_size == 64

    assert buf.dtype == torch.uint8
    assert loc.dtype == torch.int64
    assert index_k.dtype == torch.float16
    assert index_k_scale.dtype == torch.float32

    assert buf.is_contiguous()
    assert loc.is_contiguous()
    assert index_k.is_contiguous()
    assert index_k_scale.is_contiguous()

    # ------------------------------------------------------------------
    # Crossover idea: sort by page index (high 6 bits of loc) only,
    # not by the full loc.  This groups all tokens belonging to the same
    # page together, yielding coalesced within‑page writes while avoiding
    # the overhead of a full global sort.
    # ------------------------------------------------------------------
    sorted_perm = torch.argsort(loc >> 6, stable=True)

    # Reorder inputs according to the sorted page-index order.
    sorted_loc = loc[sorted_perm]
    sorted_index_k = index_k[sorted_perm]
    sorted_index_k_scale = index_k_scale[sorted_perm]

    # ------------------------------------------------------------------
    # Launch kernel with tokens sorted by page index.
    # ------------------------------------------------------------------
    buf_fp16 = buf.view(torch.float16)
    buf_fp32 = buf.view(torch.float32)

    _set_k_and_s_triton_kernel[(num_tokens_to_write,)](
        buf_fp16,
        buf_fp32,
        sorted_loc,
        sorted_index_k,
        sorted_index_k_scale,
        sorted_index_k.stride(0),  # index_k_ptr_stride_0
        PAGE_SIZE=page_size,
        BUF_NUMEL_PER_PAGE=buf_numel_per_page,
        NUM_K_ELEMS_PER_TOKEN=index_head_dim,
        S_OFFSET_NBYTES_IN_PAGE=page_size * index_head_dim,
        num_warps=2,
        num_stages=2,
    )