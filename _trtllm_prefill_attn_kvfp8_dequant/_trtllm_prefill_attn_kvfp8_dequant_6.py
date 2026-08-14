# SPDX-License-Identifier: Apache-2.0
# Interface derived from the competition dataset / vLLM attention code.
# Chunked Ascend implementation generated for this project on 2026-08-02.
"""Chunked and bounds-safe KV-cache dequantization for Triton-Ascend."""

import logging

import torch
import triton
import triton.language as tl


PAD_SLOT_ID = -1
logger = logging.getLogger(__name__)
FLASHINFER_WORKSPACE_BUFFER_SIZE_BATCH_INVARIANT = 2048 * 1024 * 1024
FP8_DTYPE = torch.float8_e4m3fn
FP4_DTYPE = torch.uint8
trtllm_gen_workspace_buffer = None


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
    BLOCK_SIZE: tl.constexpr,
):
    batch_idx = tl.program_id(0).to(tl.int64)
    table_idx = tl.program_id(1).to(tl.int64)
    chunk_idx = tl.program_id(2).to(tl.int64)

    orig_page = tl.load(
        block_tables_prefill_ptr + batch_idx * block_table_stride + table_idx
    ).to(tl.int64)

    offsets = chunk_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    in_page = offsets < KV_CACHE_STRIDE
    valid_page = orig_page > 0
    safe_page = tl.where(valid_page, orig_page, 0)

    # Treat K and V as one contiguous page.  This exposes enough independent
    # work to the NPU and avoids materialising an entire page in one program's
    # UB.  Masked invalid pages are deterministically zero-filled.
    raw = tl.load(
        kv_cache_ptr + safe_page * KV_CACHE_STRIDE + offsets,
        mask=in_page & valid_page,
        other=0.0,
    ).to(tl.float32)
    k_scale = tl.load(k_scale_ptr).to(tl.float32)
    v_scale = tl.load(v_scale_ptr).to(tl.float32)
    scale = tl.where(offsets < K_CACHE_STRIDE, k_scale, v_scale)
    values = raw * scale

    dst_page = batch_idx * block_table_stride + table_idx + 1
    tl.store(
        mock_kv_cache_ptr + dst_page * KV_CACHE_STRIDE + offsets,
        values,
        mask=in_page,
    )


def trtllm_prefill_attn_kvfp8_dequant(
    kv_cache: torch.Tensor,
    block_tables_prefill: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    dequant_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert kv_cache.device.type == "npu"
    assert block_tables_prefill.device.type == "npu"
    assert k_scale.device.type == "npu"
    assert v_scale.device.type == "npu"
    assert block_tables_prefill.ndim == 2
    assert kv_cache.ndim == 5 and kv_cache.shape[1] == 2
    assert dequant_dtype in (torch.bfloat16, torch.float16)

    batch_size, pages_per_request = block_tables_prefill.shape
    shape = kv_cache.shape
    k_cache_stride = shape[2] * shape[3] * shape[4]
    kv_cache_stride = 2 * k_cache_stride

    mock_shape = (
        batch_size * pages_per_request + 1,
        shape[1],
        shape[2],
        shape[3],
        shape[4],
    )
    mock_kv_cache = torch.empty(
        mock_shape, dtype=dequant_dtype, device=kv_cache.device
    )
    # Page zero is reserved and deterministic.  It is inexpensive to clear
    # only that page and avoids exposing uninitialised data to downstream code.
    mock_kv_cache[0].zero_()

    mock_block_table = torch.arange(
        1,
        batch_size * pages_per_request + 1,
        dtype=torch.int32,
        device=kv_cache.device,
    ).reshape(batch_size, pages_per_request)

    block_size = 1024
    grid = (
        batch_size,
        pages_per_request,
        triton.cdiv(kv_cache_stride, block_size),
    )
    _trtllm_prefill_attn_kvfp8_dequant[grid](
        kv_cache,
        block_tables_prefill,
        pages_per_request,
        mock_kv_cache,
        k_scale,
        v_scale,
        K_CACHE_STRIDE=k_cache_stride,
        KV_CACHE_STRIDE=kv_cache_stride,
        BLOCK_SIZE=block_size,
    )
    return mock_kv_cache, mock_block_table
