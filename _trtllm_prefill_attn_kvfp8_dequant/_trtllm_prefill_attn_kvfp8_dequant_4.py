import torch
import triton
import triton.language as tl
import time

PAD_SLOT_ID = -1

FP8_DTYPE = torch.float8_e4m3fn
FP4_DTYPE = torch.uint8


def _synchronize_device():
    """Synchronize the current device, supporting NPU and CUDA backends."""
    try:
        if torch.npu.is_available():
            torch.npu.synchronize()
        else:
            torch.cuda.synchronize()
    except Exception:
        torch.cuda.synchronize()


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
    batch_idx = tl.program_id(0).to(tl.int32)
    mock_block_table_idx = tl.program_id(1).to(tl.int32)
    orig_page_num = tl.load(
        block_tables_prefill_ptr + batch_idx * block_table_stride + mock_block_table_idx
    )
    if orig_page_num <= 0:
        return
    dequant_dtype = mock_kv_cache_ptr.dtype.element_ty

    # Dequantize K
    k_scale_val = tl.load(k_scale_ptr)
    offset = orig_page_num * KV_CACHE_STRIDE + tl.arange(0, K_CACHE_STRIDE)
    fp8_vals = tl.load(kv_cache_ptr + offset)
    dequantized_vals = fp8_vals.to(tl.float32) * k_scale_val
    mock_cache_offset = (
        (batch_idx * block_table_stride + mock_block_table_idx + 1) * KV_CACHE_STRIDE
        + tl.arange(0, K_CACHE_STRIDE)
    )
    dequantized_vals = dequantized_vals.to(dequant_dtype)
    tl.store(mock_kv_cache_ptr + mock_cache_offset, dequantized_vals)

    # Dequantize V
    v_scale_val = tl.load(v_scale_ptr)
    offset = (
        orig_page_num * KV_CACHE_STRIDE + K_CACHE_STRIDE + tl.arange(0, K_CACHE_STRIDE)
    )
    fp8_vals = tl.load(kv_cache_ptr + offset)
    dequantized_vals = fp8_vals.to(tl.float32) * v_scale_val
    mock_cache_offset = (
        (batch_idx * block_table_stride + mock_block_table_idx + 1) * KV_CACHE_STRIDE
        + K_CACHE_STRIDE
        + tl.arange(0, K_CACHE_STRIDE)
    )
    dequantized_vals = dequantized_vals.to(dequant_dtype)
    tl.store(mock_kv_cache_ptr + mock_cache_offset, dequantized_vals)


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
    new_s = (batch_size * num_of_page_per_token + 1, s[1], s[2], s[3], s[4])

    mock_block_table = torch.arange(
        start=1,
        end=batch_size * num_of_page_per_token + 1,
        dtype=torch.int32,
        device='npu',
    ).reshape(batch_size, num_of_page_per_token)

    # Auto-tuning of num_warps and num_stages
    if not hasattr(trtllm_prefill_attn_kvfp8_dequant, '_best_config_cache'):
        trtllm_prefill_attn_kvfp8_dequant._best_config_cache = {}

    cache_key = (batch_size, num_of_page_per_token)
    if cache_key not in trtllm_prefill_attn_kvfp8_dequant._best_config_cache:
        warp_candidates = [2, 4, 8]
        stage_candidates = [0, 1, 2, 3]
        best_config = (4, 2)
        best_time = float('inf')

        tmp_mock = torch.empty(new_s, dtype=dequant_dtype, device='npu')

        for w in warp_candidates:
            for st in stage_candidates:
                try:
                    _synchronize_device()
                    start = time.perf_counter()
                    _trtllm_prefill_attn_kvfp8_dequant[(batch_size, num_of_page_per_token)](
                        kv_cache,
                        block_tables_prefill,
                        num_of_page_per_token,
                        tmp_mock,
                        k_scale,
                        v_scale,
                        k_cache_stride,
                        kv_cache_stride,
                        num_warps=w,
                        num_stages=st,
                    )
                    _synchronize_device()
                    elapsed = time.perf_counter() - start
                    if elapsed < best_time:
                        best_time = elapsed
                        best_config = (w, st)
                except Exception:
                    continue

        trtllm_prefill_attn_kvfp8_dequant._best_config_cache[cache_key] = best_config

    best_warps, best_stages = trtllm_prefill_attn_kvfp8_dequant._best_config_cache[cache_key]

    mock_kv_cache = torch.empty(new_s, dtype=dequant_dtype, device='npu')
    _trtllm_prefill_attn_kvfp8_dequant[(batch_size, num_of_page_per_token)](
        kv_cache,
        block_tables_prefill,
        num_of_page_per_token,
        mock_kv_cache,
        k_scale,
        v_scale,
        k_cache_stride,
        kv_cache_stride,
        num_warps=best_warps,
        num_stages=best_stages,
    )

    return mock_kv_cache, mock_block_table