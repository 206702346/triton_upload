import torch
import triton
import triton.language as tl

PAD_SLOT_ID = -1

FP8_DTYPE = torch.float8_e4m3fn
FP4_DTYPE = torch.uint8


@triton.jit
def _trtllm_prefill_attn_kvfp8_dequant(
    kv_cache_ptr,
    block_tables_prefill_ptr,
    block_table_stride,
    mock_kv_cache_ptr,
    k_scale_ptr,
    v_scale_ptr,
    K_CACHE_STRIDE: tl.constexpr,
    KV_CACHE_STRIDE: tl.constexpr,
):
    linear_idx = tl.program_id(0).to(tl.int32)
    batch_idx = linear_idx // block_table_stride
    page_idx = linear_idx % block_table_stride

    orig_page_num = tl.load(
        block_tables_prefill_ptr + batch_idx * block_table_stride + page_idx,
        eviction_policy='evict_first'
    )

    dequant_dtype = mock_kv_cache_ptr.dtype.element_ty
    mock_base_offset = (linear_idx + 1) * KV_CACHE_STRIDE

    k_offsets = tl.arange(0, K_CACHE_STRIDE)

    # Invalid slots are zero-filled in the mock cache.
    if orig_page_num <= 0:
        tl.store(
            mock_kv_cache_ptr + mock_base_offset + k_offsets,
            tl.zeros((K_CACHE_STRIDE,), dequant_dtype),
            eviction_policy='evict_first'
        )
        tl.store(
            mock_kv_cache_ptr + mock_base_offset + K_CACHE_STRIDE + k_offsets,
            tl.zeros((K_CACHE_STRIDE,), dequant_dtype),
            eviction_policy='evict_first'
        )
        return

    k_scale_val = tl.load(k_scale_ptr, eviction_policy='evict_first').to(tl.float32)
    v_scale_val = tl.load(v_scale_ptr, eviction_policy='evict_first').to(tl.float32)

    # Direct, contiguous vectorised loads. These loads are always in-bounds for
    # valid block-table entries.
    kv_cache_offset = orig_page_num * KV_CACHE_STRIDE

    fp8_vals_k = tl.load(
        kv_cache_ptr + kv_cache_offset + k_offsets,
        eviction_policy='evict_first'
    )
    deq_k = (fp8_vals_k.to(tl.float32) * k_scale_val).to(dequant_dtype)
    tl.store(
        mock_kv_cache_ptr + mock_base_offset + k_offsets,
        deq_k,
        eviction_policy='evict_first'
    )

    fp8_vals_v = tl.load(
        kv_cache_ptr + kv_cache_offset + K_CACHE_STRIDE + k_offsets,
        eviction_policy='evict_first'
    )
    deq_v = (fp8_vals_v.to(tl.float32) * v_scale_val).to(dequant_dtype)
    tl.store(
        mock_kv_cache_ptr + mock_base_offset + K_CACHE_STRIDE + k_offsets,
        deq_v,
        eviction_policy='evict_first'
    )


def trtllm_prefill_attn_kvfp8_dequant(
    kv_cache: torch.Tensor,
    block_tables_prefill: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    dequant_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert kv_cache.device.type == 'npu', "kv_cache must be on NPU"
    assert block_tables_prefill.device.type == 'npu', "block_tables_prefill must be on NPU"
    assert k_scale.device.type == 'npu', "k_scale must be on NPU"
    assert v_scale.device.type == 'npu', "v_scale must be on NPU"

    batch_size, num_of_page_per_token = block_tables_prefill.shape
    s = kv_cache.shape
    assert s[1] == 2
    assert dequant_dtype in (torch.bfloat16, torch.float16)

    k_cache_stride = s[2] * s[3] * s[4]
    kv_cache_stride = k_cache_stride * s[1]

    total_slots = batch_size * num_of_page_per_token
    new_s = (total_slots + 1, s[1], s[2], s[3], s[4])

    mock_block_table = torch.arange(
        start=1,
        end=total_slots + 1,
        dtype=torch.int32,
        device='npu',
    ).reshape(batch_size, num_of_page_per_token)

    if total_slots == 0:
        mock_kv_cache = torch.zeros(new_s, dtype=dequant_dtype, device='npu')
        return mock_kv_cache, mock_block_table

    # Avoid the parent's full zero initialization: the kernel writes every
    # logical slot, so only the unused row 0 needs an explicit zero.
    mock_kv_cache = torch.empty(new_s, dtype=dequant_dtype, device='npu')
    mock_kv_cache[0:1].zero_()

    grid = (total_slots,)

    # Use fewer warps for small cache strides, improving occupancy; keep 4
    # warps for larger vectors.
    num_warps = 1
    if k_cache_stride > 1024:
        num_warps = 2
    if k_cache_stride > 4096:
        num_warps = 4

    _trtllm_prefill_attn_kvfp8_dequant[grid](
        kv_cache,
        block_tables_prefill,
        num_of_page_per_token,
        mock_kv_cache,
        k_scale,
        v_scale,
        k_cache_stride,
        kv_cache_stride,
        num_warps=num_warps,
        num_stages=4,
    )

    return mock_kv_cache, mock_block_table