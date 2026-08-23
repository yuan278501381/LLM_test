# Copyright (c) 2026 Yy1 (yuan278501381) | MIT License
"""
nn_core.paged_kv_cache - PagedAttention 显存分页与前缀共享管理引擎

依据 Woosuk Kwon et al. (2023) "Efficient Memory Management for Large Language Model Serving with PagedAttention" (vLLM)：
- 将连续的 KV 序列切分为固定大小的物理块 (Physical Blocks)
- 利用虚拟页表 (Block Table) 消除外部内存碎片并将显存浪费率从 ~70% 降至 <4%
- 支持多请求并发共享前缀 (Prefix Caching) 与写时复制 (Copy-on-Write / CoW)
- 提供完整的显存利用率对比与碎片率定量遥测
"""

from typing import Any

import numpy as np

from nn_core.observability import get_logger

logger = get_logger("nn_core.paged_kv_cache")


class PhysicalBlock:
    """物理显存块，存放固定数量 Token 的 Key 与 Value 张量"""

    def __init__(self, block_id: int, block_size: int, num_kv_heads: int, head_dim: int) -> None:
        self.block_id = block_id
        self.block_size = block_size
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim

        # 预分配物理块内存: shape (num_kv_heads, block_size, head_dim)
        self.k = np.zeros((num_kv_heads, block_size, head_dim), dtype=np.float64)
        self.v = np.zeros((num_kv_heads, block_size, head_dim), dtype=np.float64)

        # 当前块内已填充的有效 token 数
        self.num_filled: int = 0
        # 引用计数 (用于 Prefix Caching 共享块与 Copy-on-Write)
        self.ref_count: int = 0

    def is_full(self) -> bool:
        return self.num_filled >= self.block_size

    def reset(self) -> None:
        self.k.fill(0.0)
        self.v.fill(0.0)
        self.num_filled = 0
        self.ref_count = 0


class PagedKVCache:
    """
    PagedAttention 统一显存池与虚拟页表管理器。
    """

    def __init__(
        self,
        total_blocks: int = 32,
        block_size: int = 4,
        num_kv_heads: int = 4,
        head_dim: int = 16,
    ) -> None:
        self.total_blocks = total_blocks
        self.block_size = block_size
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim

        # 物理显存池 (Physical Memory Pool)
        self.physical_blocks: list[PhysicalBlock] = [
            PhysicalBlock(i, block_size, num_kv_heads, head_dim) for i in range(total_blocks)
        ]
        # 空闲块队列
        self.free_block_ids: list[int] = list(range(total_blocks))

        # 虚拟页表 (Block Tables): request_id -> list of block_ids
        self.block_tables: dict[str, list[int]] = {}
        # 请求序列长度记录: request_id -> total_tokens
        self.seq_lengths: dict[str, int] = {}

        logger.info(
            "PagedKVCache 初始化: total_blocks=%d, block_size=%d, heads=%d, dim=%d",
            total_blocks,
            block_size,
            num_kv_heads,
            head_dim,
        )

    def allocate_sequence(
        self, request_id: str, prompt_tokens_k: np.ndarray, prompt_tokens_v: np.ndarray
    ) -> list[int]:
        """
        为一个新请求分配物理块并写入初始 Prompt KV 数据。
        prompt_tokens_k: shape (num_kv_heads, prompt_len, head_dim)
        prompt_tokens_v: shape (num_kv_heads, prompt_len, head_dim)
        """
        if request_id in self.block_tables:
            raise ValueError(f"请求 ID [{request_id}] 已存在，请使用唯一 ID 或先释放")

        prompt_len = prompt_tokens_k.shape[1]
        needed_blocks = int(np.ceil(prompt_len / self.block_size))

        if len(self.free_block_ids) < needed_blocks:
            raise RuntimeError(
                f"物理显存不足 (OOM): 需要 {needed_blocks} 个空闲块，剩余 {len(self.free_block_ids)} 个"
            )

        assigned_block_ids: list[int] = []
        for i in range(needed_blocks):
            bid = self.free_block_ids.pop(0)
            block = self.physical_blocks[bid]
            block.reset()
            block.ref_count = 1

            start_t = i * self.block_size
            end_t = min(start_t + self.block_size, prompt_len)
            chunk_len = end_t - start_t

            block.k[:, :chunk_len, :] = prompt_tokens_k[:, start_t:end_t, :]
            block.v[:, :chunk_len, :] = prompt_tokens_v[:, start_t:end_t, :]
            block.num_filled = chunk_len

            assigned_block_ids.append(bid)

        self.block_tables[request_id] = assigned_block_ids
        self.seq_lengths[request_id] = prompt_len
        return assigned_block_ids

    def append_token(
        self, request_id: str, k_token: np.ndarray, v_token: np.ndarray
    ) -> tuple[int, bool]:
        """
        为指定请求追加单个生成的 Token KV。
        若当前最后一个物理块已满，自动申请新物理块；
        若最后一个物理块被多请求共享 (ref_count > 1)，触发写时复制 (Copy-on-Write / CoW)。

        k_token: (num_kv_heads, head_dim)
        v_token: (num_kv_heads, head_dim)

        Returns:
            (target_block_id, triggered_cow)
        """
        if request_id not in self.block_tables:
            raise KeyError(f"未找到请求 ID [{request_id}]")

        table = self.block_tables[request_id]
        if not table:
            if not self.free_block_ids:
                raise RuntimeError("显存池已满，无法分配新块")
            bid = self.free_block_ids.pop(0)
            block = self.physical_blocks[bid]
            block.reset()
            block.ref_count = 1
            table.append(bid)

        last_bid = table[-1]
        last_block = self.physical_blocks[last_bid]
        triggered_cow = False

        # 若当前块已满，分配新物理块并追加至页表
        if last_block.is_full():
            if not self.free_block_ids:
                raise RuntimeError("显存不足，无法追加新 Token 块")
            new_bid = self.free_block_ids.pop(0)
            new_block = self.physical_blocks[new_bid]
            new_block.reset()
            new_block.ref_count = 1
            table.append(new_bid)
            last_block = new_block
            last_bid = new_bid
        elif last_block.ref_count > 1:
            # 若当前块未满但被多请求共享，执行写时复制 (CoW)
            if not self.free_block_ids:
                raise RuntimeError("显存不足，无法执行 Copy-on-Write 分裂新块")
            new_bid = self.free_block_ids.pop(0)
            new_block = self.physical_blocks[new_bid]
            new_block.reset()
            new_block.k[:] = last_block.k[:]
            new_block.v[:] = last_block.v[:]
            new_block.num_filled = last_block.num_filled
            new_block.ref_count = 1

            # 原共享块引用计数减 1
            last_block.ref_count -= 1
            # 当前请求页表更新为私有新块
            table[-1] = new_bid
            last_block = new_block
            last_bid = new_bid
            triggered_cow = True

        # 写入 token
        idx = last_block.num_filled
        last_block.k[:, idx, :] = k_token
        last_block.v[:, idx, :] = v_token
        last_block.num_filled += 1
        self.seq_lengths[request_id] += 1

        return last_bid, triggered_cow

    def fork_sequence_prefix(self, source_request_id: str, new_request_id: str) -> list[int]:
        """
        前缀共享 (Prefix Caching)：派生一个新请求并共享源请求的全部当前物理页，无需重复复制显存。
        """
        if source_request_id not in self.block_tables:
            raise KeyError(f"源请求 ID [{source_request_id}] 不存在")
        if new_request_id in self.block_tables:
            raise ValueError(f"新请求 ID [{new_request_id}] 已存在")

        src_table = self.block_tables[source_request_id]
        new_table: list[int] = []

        for bid in src_table:
            block = self.physical_blocks[bid]
            block.ref_count += 1
            new_table.append(bid)

        self.block_tables[new_request_id] = new_table
        self.seq_lengths[new_request_id] = self.seq_lengths[source_request_id]
        return new_table

    def free_sequence(self, request_id: str) -> None:
        """释放指定请求占用的虚拟页表，自动对引用计数归零的物理块归还空闲池"""
        if request_id not in self.block_tables:
            return

        table = self.block_tables.pop(request_id)
        self.seq_lengths.pop(request_id, None)

        for bid in table:
            block = self.physical_blocks[bid]
            block.ref_count -= 1
            if block.ref_count <= 0:
                block.reset()
                self.free_block_ids.append(bid)

    def read_full_kv(self, request_id: str) -> tuple[np.ndarray, np.ndarray]:
        """
        根据页表虚拟逻辑地址重组出完整的 Key 和 Value 张量。
        shape: (num_kv_heads, total_seq_len, head_dim)
        """
        if request_id not in self.block_tables:
            raise KeyError(f"未找到请求 ID [{request_id}]")

        table = self.block_tables[request_id]
        seq_len = self.seq_lengths[request_id]

        full_k = np.zeros((self.num_kv_heads, seq_len, self.head_dim), dtype=np.float64)
        full_v = np.zeros((self.num_kv_heads, seq_len, self.head_dim), dtype=np.float64)

        cur_pos = 0
        for bid in table:
            block = self.physical_blocks[bid]
            count = min(block.num_filled, seq_len - cur_pos)
            if count <= 0:
                break
            full_k[:, cur_pos : cur_pos + count, :] = block.k[:, :count, :]
            full_v[:, cur_pos : cur_pos + count, :] = block.v[:, :count, :]
            cur_pos += count

        return full_k, full_v

    def get_memory_stats(self) -> dict[str, Any]:
        """
        计算显存利用率、碎片率与共享前缀节省收益。
        """
        total_slots = self.total_blocks * self.block_size
        used_blocks = [b for b in self.physical_blocks if b.ref_count > 0]
        num_used_blocks = len(used_blocks)

        total_logical_tokens = sum(self.seq_lengths.values())
        physical_stored_tokens = sum(b.num_filled for b in used_blocks)

        # 内部碎片：已分配物理块中未填满的尾部槽位
        internal_frag_slots = sum(b.block_size - b.num_filled for b in used_blocks)
        internal_frag_rate = (
            float(internal_frag_slots / max(1, num_used_blocks * self.block_size))
            if num_used_blocks > 0
            else 0.0
        )

        # 前缀共享节省 Token 数
        shared_saved_tokens = max(0, total_logical_tokens - physical_stored_tokens)

        # 对比传统静态连续预分配 (假设每个请求预先按最大长度 max_len=64 预分配):
        simulated_contiguous_slots = len(self.block_tables) * 64
        traditional_waste_rate = (
            float(
                (simulated_contiguous_slots - total_logical_tokens)
                / max(1, simulated_contiguous_slots)
            )
            if simulated_contiguous_slots > 0
            else 0.0
        )

        return {
            "total_blocks": self.total_blocks,
            "free_blocks": len(self.free_block_ids),
            "used_blocks": num_used_blocks,
            "block_size": self.block_size,
            "total_capacity_tokens": total_slots,
            "active_requests": len(self.block_tables),
            "total_logical_tokens": total_logical_tokens,
            "physical_stored_tokens": physical_stored_tokens,
            "internal_fragmentation_rate": internal_frag_rate,
            "shared_saved_tokens": shared_saved_tokens,
            "traditional_prealloc_waste_rate": traditional_waste_rate,
            "paged_memory_utilization": float(physical_stored_tokens / max(1, total_slots)),
        }
