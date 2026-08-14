import torch
import triton
import triton.language as tl


def _dequantize_k_cache_fast(quant_k_cache, group_size: int = 128):
    num_tokens, dim_quant = quant_k_cache.shape

    dim_nope = 512
    dim_rope = 64
    num_tiles = dim_nope // group_size
    assert dim_quant == 656

    output = torch.empty(
        (num_tokens, dim_nope + dim_rope),
        dtype=torch.bfloat16,
        device=quant_k_cache.device,
    )

    assert dim_nope % group_size == 0

    input_nope_q = quant_k_cache[:, :dim_nope]
    scales_compact = quant_k_cache[:, dim_nope : dim_nope + num_tiles * 4].view(torch.float32)
    input_rope = quant_k_cache[
        :, dim_nope + num_tiles * 4 : dim_nope + num_tiles * 4 + dim_rope
    ].view(torch.bfloat16)

    # Pre‑expand per‑block scales into full per‑element array outside the kernel
    input_nope_s = torch.repeat_interleave(scales_compact, group_size, dim=1).contiguous()

    # One program per token, highly parallelized with wide warps and deep pipelining
    _dequantize_k_cache_fast_kernel[(num_tokens,)](
        output,
        input_nope_q,
        input_nope_s,
        input_rope,
        output.stride(0),
        input_nope_q.stride(0),
        input_nope_s.stride(0),
        input_rope.stride(0),
        NUM_NOPE_BLOCKS=num_tiles,
        GROUP_SIZE=group_size,
        DIM_NOPE=dim_nope,
        DIM_ROPE=dim_rope,
        num_warps=16,     # 512 elements / 32 = 16 warps
        num_stages=4,
    )

    return output


@triton.jit
def _dequantize_k_cache_fast_kernel(
    output_ptr,
    input_nope_q_ptr,
    input_nope_s_ptr,
    input_rope_ptr,
    output_stride_0: int,
    input_nope_q_stride_0: int,
    input_nope_s_stride_0: int,
    input_rope_stride_0: int,
    NUM_NOPE_BLOCKS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    DIM_NOPE: tl.constexpr,
    DIM_ROPE: tl.constexpr,
):
    token_id = tl.program_id(0)

    # --- Dequantize the whole NOPE part in one wide vector operation ---
    offs_nope = tl.arange(0, DIM_NOPE)
    mask_nope = offs_nope < DIM_NOPE

    ptr_q = input_nope_q_ptr + token_id * input_nope_q_stride_0 + offs_nope
    y_q = tl.load(ptr_q, mask=mask_nope, other=0.0).to(tl.float32)

    # Scales have been pre‑expanded to the full length (float32)
    ptr_s = input_nope_s_ptr + token_id * input_nope_s_stride_0 + offs_nope
    y_s = tl.load(ptr_s, mask=mask_nope, other=0.0)

    y = (y_q * y_s).to(output_ptr.dtype.element_ty)

    dst_nope = output_ptr + token_id * output_stride_0 + offs_nope
    tl.store(dst_nope, y, mask=mask_nope)

    # --- Copy the ROPE part (raw bf16, no dequant) ---
    offs_rope = tl.arange(0, DIM_ROPE)
    mask_rope = offs_rope < DIM_ROPE

    src_ptr = input_rope_ptr + token_id * input_rope_stride_0 + offs_rope
    dst_ptr = output_ptr + token_id * output_stride_0 + DIM_NOPE + offs_rope

    data = tl.load(src_ptr, mask=mask_rope)
    tl.store(dst_ptr, data, mask=mask_rope)