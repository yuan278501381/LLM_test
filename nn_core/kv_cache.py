# Copyright (c) 2026 Yy1 (yuan278501381) | MIT License
"""
nn_core.kv_cache - KV-Cache 推理显存与吞吐优化容器

实现自回归大语言模型推理加速的核心机制 (KV-Cache)：
- 避免过去生成序列 Key 和 Value 的冗余重复矩阵乘法
- 每步生成仅计算当前单个 Token 的 Q/K/V，将计算复杂度从 O(N^2) 降至 O(1) 每步
- 显存占用动态计量与理论计算量节省追踪
"""

import logging
import uuid
import numpy as np

logger = logging.getLogger(__name__)


class KVCache:
    """
    单层 / 多层 KV 缓存容器。
    """

    def __init__(
        self,
        num_layers: int,
        max_batch_size: int = 1,
        max_seq_len: int = 128,
        num_kv_heads: int = 4,
        head_dim: int = 16,
    ) -> None:
        self.num_layers = num_layers
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim

        # 当前已缓存的 Token 数量
        self.current_seq_len = 0

        # 每层的缓存存储: list of dict {'k': ndarray, 'v': ndarray}
        # shape: (max_batch_size, num_kv_heads, max_seq_len, head_dim)
        self.k_cache: list[np.ndarray] = [
            np.zeros((max_batch_size, num_kv_heads, max_seq_len, head_dim), dtype=np.float64)
            for _ in range(num_layers)
        ]
        self.v_cache: list[np.ndarray] = [
            np.zeros((max_batch_size, num_kv_heads, max_seq_len, head_dim), dtype=np.float64)
            for _ in range(num_layers)
        ]

        # 统计指标追踪
        self.total_flops_with_cache: int = 0
        self.total_flops_without_cache: int = 0

        tid = uuid.uuid4().hex[:8]
        logger.info(
            "[%s] KVCache 已初始化: layers=%d, max_len=%d, heads=%d, dim=%d",
            tid, num_layers, max_seq_len, num_kv_heads, head_dim
        )

    def update(
        self,
        layer_idx: int,
        new_k: np.ndarray,
        new_v: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        向指定层追加新的 Key 和 Value，并返回截至当前步的完整历史 KV。
        
        Args:
            layer_idx: 层索引
            new_k: 形状 (batch_size, num_kv_heads, new_seq_len, head_dim)
            new_v: 形状 (batch_size, num_kv_heads, new_seq_len, head_dim)
            
        Returns:
            full_k: 形状 (batch_size, num_kv_heads, total_seq_len, head_dim)
            full_v: 形状 (batch_size, num_kv_heads, total_seq_len, head_dim)
        """
        batch_size, num_kv_heads, new_len, head_dim = new_k.shape
        start_pos = self.current_seq_len
        end_pos = start_pos + new_len

        if end_pos > self.max_seq_len:
            raise ValueError(f"序列长度超过最大缓存容量 ({end_pos} > {self.max_seq_len})")

        # 就地填入缓存
        self.k_cache[layer_idx][:batch_size, :, start_pos:end_pos, :] = new_k
        self.v_cache[layer_idx][:batch_size, :, start_pos:end_pos, :] = new_v

        # 如果是最后一层，推进全局序列长度计数
        if layer_idx == self.num_layers - 1:
            self.current_seq_len = end_pos

        # 返回有效历史区间
        full_k = self.k_cache[layer_idx][:batch_size, :, :end_pos, :]
        full_v = self.v_cache[layer_idx][:batch_size, :, :end_pos, :]
        return full_k, full_v

    def reset(self) -> None:
        """清空缓存，准备下一次会话"""
        self.current_seq_len = 0
        for i in range(self.num_layers):
            self.k_cache[i].fill(0)
            self.v_cache[i].fill(0)

    def get_memory_stats(self) -> dict[str, float]:
        """
        返回当前 KV 缓存的物理显存占用与开销指标 (以 KB 和 MB 为单位)。
        """
        # 每个 float64 占 8 字节 (或 float32 占 4 字节)
        bytes_per_elem = 8
        total_elements = 2 * self.num_layers * self.max_batch_size * self.num_kv_heads * self.current_seq_len * self.head_dim
        allocated_elements = 2 * self.num_layers * self.max_batch_size * self.num_kv_heads * self.max_seq_len * self.head_dim

        used_bytes = total_elements * bytes_per_elem
        allocated_bytes = allocated_elements * bytes_per_elem

        return {
            "used_kb": used_bytes / 1024.0,
            "allocated_kb": allocated_bytes / 1024.0,
            "current_tokens": float(self.current_seq_len),
            "max_tokens": float(self.max_seq_len),
            "utilization_percent": (self.current_seq_len / max(1, self.max_seq_len)) * 100.0,
        }
