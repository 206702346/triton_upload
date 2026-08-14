import torch
import triton
import triton.language as tl

@triton.jit
def _count_expert_num_tokens(
    topk_ids_ptr,
    expert_num_tokens_ptr,
    num_experts: tl.constexpr,      # 编译时常量，减少运行时标量计算
    topk_numel: tl.constexpr,       # 编译时常量
    expert_map_ptr,
    HAS_EXPERT_MAP: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    curr_expert = tl.program_id(0)
    
    # 提前退出无效 expert，避免空转
    if curr_expert >= num_experts:
        return
    
    # 向量化偏移，一次性生成
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # 预计算循环次数（标量外提，编译时常量折叠）
    # 若 topk_numel 是编译时常量，tl.cdiv 可在编译期计算
    num_blocks = tl.cdiv(topk_numel, BLOCK_SIZE)
    
    # 初始化累加器
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.int32)
    
    # 预计算基址指针，循环内只做常量偏移（减少 scaler）
    base_ptr = topk_ids_ptr + offsets
    
    # 主循环：向量化加载 + 计算
    for x in range(num_blocks):
        # 计算当前块掩码（向量化，无标量分支）
        # 使用 tl.minimum 避免标量比较
        block_start = x * BLOCK_SIZE
        mask = offsets < (topk_numel - block_start)
        
        # 加载 expert_ids（MTE2）
        # other=-1 确保越界数据不参与后续计算
        expert_ids = tl.load(base_ptr + block_start, mask=mask, other=-1)
        
        # expert_map 查找（条件编译，无运行时分支开销）
        if HAS_EXPERT_MAP:
            # 向量化索引，避免逐元素标量操作
            map_mask = expert_ids >= 0
            expert_ids = tl.load(expert_map_ptr + expert_ids, mask=map_mask, other=-1)
        
        # 向量化比较和累加（Vector 核心计算）
        # tl.where 是向量指令，无标量分支
        has_curr_expert = tl.where(expert_ids == curr_expert, 1, 0)
        acc = acc + has_curr_expert
    
    # 向量化规约：tl.sum 应编译为树形归约，减少 scaler 参与
    # 存储结果（MTE3）
    tl.store(expert_num_tokens_ptr + curr_expert, tl.sum(acc))

def count_expert_num_tokens(
    topk_ids: torch.Tensor, num_local_experts: int, expert_map: torch.Tensor | None
) -> torch.Tensor:
    """
    Count the number to tokens assigned to each expert.

    Parameters:
    - topk_ids (torch.Tensor): Tensor mapping each token to its
    list of experts.
    - num_local_experts (int): Number of experts in this rank.
    - expert_map (Optional[torch.Tensor]):  A tensor mapping expert indices
    from the global expert space to the local expert space of the expert
    parallel shard.

    Returns:
    A tensor of size num_local_experts, where tensor[i] holds the number
    of tokens assigned to the ith expert.
    """
    assert topk_ids.dtype.is_signed, "The kernel uses -1 to represent invalid topk_ids"
    expert_num_tokens = torch.empty(
        (num_local_experts), device=topk_ids.device, dtype=torch.int32
    )

    grid = num_local_experts
    BLOCK_SIZE = min(topk_ids.numel(), 1024)
    BLOCK_SIZE = triton.next_power_of_2(BLOCK_SIZE)

    _count_expert_num_tokens[(grid,)](
        topk_ids,
        expert_num_tokens,
        num_local_experts,
        topk_ids.numel(),
        expert_map,
        HAS_EXPERT_MAP=expert_map is not None,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return expert_num_tokens