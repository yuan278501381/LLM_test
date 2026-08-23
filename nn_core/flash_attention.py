# Copyright (c) 2026 Yy1 (yuan278501381) | MIT License
"""
nn_core.flash_attention - FlashAttention-2 核心分块与 Online Softmax 算法引擎

依据 Tri Dao (2023) "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning"：
- 利用 SRAM 有限高速缓存进行分块循环 (Block-wise Tiling)
- 利用 Online Softmax 动态维护局部最大值 m 与行指数和 l，避免 O(N^2) 注意力矩阵全量落地 HBM
- 实现与标准缩放点积注意力 (Scaled Dot-Product Attention) 的数学严格等价
- 提供 HBM 读写字节数、SRAM 占用量与计算复杂度的全链路追踪
"""

from typing import Any

import numpy as np

from nn_core.observability import get_logger

logger = get_logger("nn_core.flash_attention")


def flash_attention_2_forward(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    block_size_r: int = 16,
    block_size_c: int = 16,
    is_causal: bool = False,
    scale: float | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    r"""
    FlashAttention-2 前向计算核心函数 (纯 NumPy 白盒实现)。

    算法流程 (以单头为例，支持多维 Batch/Head 广播)：
    1. 将 Q 划分为 Tr = ceil(N / Br) 个行块 Qi (大小 Br x d)；
    2. 将 K, V 划分为 Tc = ceil(N / Bc) 个列块 Kj, Vj (大小 Bc x d)；
    3. 外层遍历 Qi，从 HBM 加载至 SRAM，初始化局部统计量 mi = -inf, li = 0, Oi = 0；
    4. 内层遍历 Kj, Vj，计算局部块内分数 Sij = (Qi @ Kj.T) * scale；
    5. 执行 Online Softmax 动态更新：
       $$m_{new} = \max(m_i, \text{rowmax}(S_{ij}))$$
       $$\alpha = \exp(m_i - m_{new})$$
       $$\tilde{P} = \exp(S_{ij} - m_{new})$$
       $$\ell_{new} = \alpha \cdot \ell_i + \text{rowsum}(\tilde{P})$$
       $$O_i = \alpha \cdot O_i + \tilde{P} @ V_j$$
    6. 最终归一化 $O_i = O_i / \ell_i$ 并写回 HBM。

    Args:
        q: Query 张量，形状 (..., seq_len_q, d_k)
        k: Key 张量，形状 (..., seq_len_k, d_k)
        v: Value 张量，形状 (..., seq_len_k, d_v)
        block_size_r: Query 划分行块大小 Br (默认 16)
        block_size_c: Key/Value 划分列块大小 Bc (默认 16)
        is_causal: 是否应用因果自回归掩码
        scale: 缩放因子，默认为 1 / sqrt(d_k)

    Returns:
        output: 注意力加权输出，形状与标准注意力完全一致
        telemetry: 包含算法分步追踪、SRAM 占用量、HBM IO 读写量对比的字典
    """
    q_arr = np.asarray(q, dtype=np.float64)
    k_arr = np.asarray(k, dtype=np.float64)
    v_arr = np.asarray(v, dtype=np.float64)

    if q_arr.ndim < 2 or k_arr.ndim < 2 or v_arr.ndim < 2:
        raise ValueError("q, k, v 至少需要序列维与特征维")
    if q_arr.shape[-1] != k_arr.shape[-1]:
        raise ValueError(f"q 与 k 的特征维度不匹配 ({q_arr.shape[-1]} != {k_arr.shape[-1]})")
    if k_arr.shape[-2] != v_arr.shape[-2]:
        raise ValueError(f"k 与 v 的序列长度不匹配 ({k_arr.shape[-2]} != {v_arr.shape[-2]})")

    orig_shape_q = q_arr.shape
    d_k = q_arr.shape[-1]
    d_v = v_arr.shape[-1]
    seq_len_q = q_arr.shape[-2]
    seq_len_k = k_arr.shape[-2]

    if scale is None:
        scale = 1.0 / np.sqrt(d_k)

    # 展平批次维度便于逐头/逐批次处理: (Batch, N, d)
    q_flat = q_arr.reshape(-1, seq_len_q, d_k)
    k_flat = k_arr.reshape(-1, seq_len_k, d_k)
    v_flat = v_arr.reshape(-1, seq_len_k, d_v)
    batch_size = q_flat.shape[0]

    br = max(1, min(block_size_r, seq_len_q))
    bc = max(1, min(block_size_c, seq_len_k))

    tr = int(np.ceil(seq_len_q / br))
    tc = int(np.ceil(seq_len_k / bc))

    out_flat = np.zeros((batch_size, seq_len_q, d_v), dtype=np.float64)

    # 统计信息追踪
    steps_trace: list[dict[str, Any]] = []
    total_flops = 0
    hbm_read_bytes = 0
    hbm_write_bytes = 0

    # 元素字节大小 (float64 为 8 字节，float16 为 2 字节，教学统一按 float32 4 字节计算 IO 模型)
    dtype_bytes = 4

    for b in range(batch_size):
        q_b = q_flat[b]
        k_b = k_flat[b]
        v_b = v_flat[b]

        for i in range(tr):
            r_start = i * br
            r_end = min(r_start + br, seq_len_q)
            q_i = q_b[r_start:r_end, :]  # (Br_actual, d_k)
            br_actual = r_end - r_start

            # 从 HBM 读取 Qi
            hbm_read_bytes += br_actual * d_k * dtype_bytes

            # 初始化当前行块的统计量
            m_i = np.full((br_actual,), -np.inf, dtype=np.float64)
            l_i = np.zeros((br_actual,), dtype=np.float64)
            o_i = np.zeros((br_actual, d_v), dtype=np.float64)

            # 若因果掩码生效，j 只需要遍历到与当前行块有重叠的列块即可
            max_j = tc
            if is_causal:
                max_j = min(tc, int(np.ceil(r_end / bc)))

            for j in range(max_j):
                c_start = j * bc
                c_end = min(c_start + bc, seq_len_k)
                k_j = k_b[c_start:c_end, :]  # (Bc_actual, d_k)
                v_j = v_b[c_start:c_end, :]  # (Bc_actual, d_v)
                bc_actual = c_end - c_start

                # 从 HBM 读取 Kj, Vj
                hbm_read_bytes += (bc_actual * d_k + bc_actual * d_v) * dtype_bytes

                # 1. 块内点积: S_ij = Q_i @ K_j.T * scale (在 SRAM 中完成)
                s_ij = np.dot(q_i, k_j.T) * scale  # (Br_actual, Bc_actual)
                total_flops += 2 * br_actual * bc_actual * d_k

                # 2. 因果掩码处理
                if is_causal:
                    q_indices = np.arange(r_start, r_end)[:, None]
                    k_indices = np.arange(c_start, c_end)[None, :]
                    causal_mask = q_indices < k_indices
                    s_ij = np.where(causal_mask, -np.inf, s_ij)

                # 3. Online Softmax 更新
                # 计算当前块内各行最大值
                m_ij = np.max(s_ij, axis=-1)  # (Br_actual,)
                # 防止全 -inf 产生 NaN
                m_ij = np.where(np.isneginf(m_ij), -1e9, m_ij)

                # 更新全局行最大值
                m_new = np.maximum(m_i, m_ij)
                m_new = np.where(np.isneginf(m_new), 0.0, m_new)

                # 尺度对齐系数 alpha = exp(m_i - m_new)
                alpha = np.exp(np.where(np.isneginf(m_i), 0.0, m_i - m_new))
                alpha = np.where(np.isneginf(m_i), 0.0, alpha)

                # 当前块的新权重 p_tilde = exp(S_ij - m_new)
                p_tilde = np.exp(s_ij - m_new[:, None])
                p_tilde = np.where(np.isnan(p_tilde), 0.0, p_tilde)

                # 累加行指数和 l_new = alpha * l_i + rowsum(p_tilde)
                l_new = alpha * l_i + np.sum(p_tilde, axis=-1)

                # 累加加权值 O_i = alpha * O_i + p_tilde @ V_j
                pv = np.dot(p_tilde, v_j)
                total_flops += 2 * br_actual * bc_actual * d_v
                o_i = alpha[:, None] * o_i + pv

                m_i = m_new
                l_i = l_new

                if b == 0 and len(steps_trace) < 20:
                    steps_trace.append(
                        {
                            "step": len(steps_trace) + 1,
                            "query_block": i,
                            "kv_block": j,
                            "q_range": (r_start, r_end),
                            "kv_range": (c_start, c_end),
                            "sram_peak_bytes": (
                                (br_actual * d_k + 2 * bc_actual * d_k + br_actual * bc_actual)
                                * dtype_bytes
                            ),
                            "m_sample": float(np.mean(m_i)),
                            "l_sample": float(np.mean(l_i)),
                        }
                    )

            # 4. 最终归一化: O_i = O_i / l_i
            l_i_safe = np.where(l_i == 0, 1.0, l_i)
            o_final = o_i / l_i_safe[:, None]

            # 写回 HBM
            out_flat[b, r_start:r_end, :] = o_final
            hbm_write_bytes += br_actual * d_v * dtype_bytes

    out = out_flat.reshape((*orig_shape_q[:-1], d_v))

    # 理论标准 Attention 显存 IO 计算 (标准 Attention 必须将全量 N x N 矩阵写回 HBM 再读回):
    # Standard IO: Read Q, K (2*N*d), Write S (N^2), Read S, V (N^2 + N*d), Write O (N*d)
    standard_hbm_io_bytes = (
        batch_size
        * (
            2 * seq_len_q * d_k
            + seq_len_k * d_v
            + seq_len_q * seq_len_k  # Write S
            + seq_len_q * seq_len_k  # Read S (Softmax)
            + seq_len_k * d_v  # Read V
            + seq_len_q * d_v  # Write O
        )
        * dtype_bytes
    )
    flash_hbm_io_bytes = hbm_read_bytes + hbm_write_bytes

    telemetry = {
        "block_size_r": br,
        "block_size_c": bc,
        "num_q_blocks": tr,
        "num_kv_blocks": tc,
        "total_flops": total_flops,
        "flash_hbm_io_bytes": flash_hbm_io_bytes,
        "standard_hbm_io_bytes": standard_hbm_io_bytes,
        "io_reduction_ratio": (
            float(standard_hbm_io_bytes / max(1, flash_hbm_io_bytes))
            if flash_hbm_io_bytes > 0
            else 1.0
        ),
        "steps_trace": steps_trace,
        "sram_footprint_bytes": (br * d_k + 2 * bc * d_k + br * bc) * dtype_bytes,
    }

    logger.debug(
        "FlashAttention-2 完成: seq_len=(%d,%d), IO节省=%.2fx",
        seq_len_q,
        seq_len_k,
        telemetry["io_reduction_ratio"],
    )
    return out, telemetry
