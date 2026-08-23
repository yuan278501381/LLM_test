# Copyright (c) 2026 Yy1 (yuan278501381) | MIT License
"""
nn_core.moe - 混合专家模型 (Mixture-of-Experts / MoE) 与稀疏门控路由引擎

包含：
- `ExpertFFN` (独立专家前馈神经网络)
- `TopKGating` (稀疏门控路由器，支持 Top-1 / Top-2 动态选择与重新归一化)
- `MoELayer` (端到端 MoE 专家分发与加权融合层)
- 负载均衡辅助损失 (Load Balancing Auxiliary Loss) 与路由坍塌 (Routing Collapse) 消融诊断
- DeepSeek-V3 式动态无辅助损失偏置平衡机制 (Auxiliary-Loss-Free Bias Adjustment)
"""

from typing import Any

import numpy as np

from nn_core.activations import ReLU
from nn_core.observability import get_logger
from nn_core.swiglu import SwiGLU

logger = get_logger("nn_core.moe")


class ExpertFFN:
    """单个专家前馈神经网络 (Feed-Forward Network)"""

    def __init__(
        self, d_model: int, d_ff: int, activation_type: str = "swiglu", seed: int | None = None
    ) -> None:
        self.d_model = d_model
        self.d_ff = d_ff
        self.activation_type = activation_type
        rng = np.random.RandomState(seed)

        if activation_type == "swiglu":
            self.act = SwiGLU(d_model=d_model, d_ff=d_ff)
        else:
            self.w1 = rng.randn(d_model, d_ff) * np.sqrt(2.0 / d_model)
            self.b1 = np.zeros(d_ff)
            self.w2 = rng.randn(d_ff, d_model) * np.sqrt(2.0 / d_ff)
            self.b2 = np.zeros(d_model)
            self.relu = ReLU()

    def forward(self, x: np.ndarray) -> np.ndarray:
        """前向传播: x shape (N, d_model) -> output shape (N, d_model)"""
        if self.activation_type == "swiglu":
            return self.act.forward(x)
        h = self.relu.forward(np.dot(x, self.w1) + self.b1)
        return np.dot(h, self.w2) + self.b2


class TopKGating:
    """
    稀疏门控路由器 (Sparse Gating Router)。
    依据 Shazeer et al. (2017) 与 Switch Transformers：
    1. 计算每个 Token 对各专家的初始亲和度 Logits: H = x @ W_g + bias
    2. 计算全专家 Softmax 概率用于辅助损失监督: P = Softmax(H)
    3. 选择 Top-K 最高分专家索引与分数，并对 Top-K 进行局部分数归一化
    """

    def __init__(
        self,
        d_model: int,
        num_experts: int,
        top_k: int = 2,
        aux_loss_weight: float = 0.01,
        seed: int | None = None,
    ) -> None:
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        self.aux_loss_weight = aux_loss_weight

        rng = np.random.RandomState(seed)
        self.w_gating = rng.randn(d_model, num_experts) * np.sqrt(2.0 / d_model)
        self.expert_bias = np.zeros(num_experts, dtype=np.float64)

    def forward(
        self, x: np.ndarray, apply_bias: bool = True
    ) -> tuple[np.ndarray, np.ndarray, float, dict[str, Any]]:
        """
        前向路由选择。

        Args:
            x: 输入 Token 特征，形状 (num_tokens, d_model)
            apply_bias: 是否叠加动态专家偏置

        Returns:
            topk_weights: 选中专家的重归一化权重，形状 (num_tokens, top_k)
            topk_indices: 选中专家的索引，形状 (num_tokens, top_k)
            aux_loss: 负载均衡辅助损失标量
            gating_stats: 包含各专家处理比例与负载均衡度量指标
        """
        num_tokens = x.shape[0]
        # 1. 原始门控得分
        raw_logits = np.dot(x, self.w_gating)  # (N, num_experts)
        if apply_bias:
            raw_logits += self.expert_bias

        # 2. 全专家 Softmax 概率分布 (用于计算 auxiliary loss)
        shifted_logits = raw_logits - np.max(raw_logits, axis=-1, keepdims=True)
        exp_logits = np.exp(shifted_logits)
        full_probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)  # (N, num_experts)

        # 3. Top-K 选择
        topk_indices = np.argpartition(-raw_logits, kth=self.top_k - 1, axis=-1)[:, : self.top_k]
        # 按得分从高到低精确排序
        row_indices = np.arange(num_tokens)[:, None]
        topk_logits = raw_logits[row_indices, topk_indices]
        sort_order = np.argsort(-topk_logits, axis=-1)
        topk_indices = np.take_along_axis(topk_indices, sort_order, axis=-1)
        sorted_topk_logits = np.take_along_axis(topk_logits, sort_order, axis=-1)

        # 4. 对选中的 Top-K 权重重新 Softmax 归一化 (确保加权和为 1.0)
        shifted_topk = sorted_topk_logits - np.max(sorted_topk_logits, axis=-1, keepdims=True)
        exp_topk = np.exp(shifted_topk)
        topk_weights = exp_topk / np.sum(exp_topk, axis=-1, keepdims=True)  # (N, top_k)

        # 5. 负载均衡辅助损失 (Switch Transformer 标准辅助损失公式):
        # L_aux = alpha * E * sum_{e=1}^E (f_e * P_e)
        # f_e: 派发到专家 e 的 token 频次比例
        # P_e: 模型预测分配给专家 e 的平均门控概率
        expert_mask = np.zeros((num_tokens, self.num_experts), dtype=np.float64)
        np.put_along_axis(expert_mask, topk_indices, 1.0, axis=-1)

        f_e = np.mean(expert_mask, axis=0)  # (num_experts,)
        p_e = np.mean(full_probs, axis=0)  # (num_experts,)

        aux_loss = float(self.aux_loss_weight * self.num_experts * np.sum(f_e * p_e))

        # 统计指标与负载均衡度量 (基尼系数 / 熵)
        prob_entropy = -float(np.sum(p_e * np.log(np.maximum(p_e, 1e-12))))
        max_utilization = float(np.max(f_e))
        min_utilization = float(np.min(f_e))

        gating_stats = {
            "expert_dispatch_fractions": f_e.tolist(),
            "expert_mean_probs": p_e.tolist(),
            "entropy": prob_entropy,
            "max_load_imbalance_ratio": float(max_utilization / max(1e-6, min_utilization)),
            "is_collapsed": bool(max_utilization > 0.75 and self.num_experts >= 4),
        }

        return topk_weights, topk_indices, aux_loss, gating_stats

    def update_dynamic_bias(
        self, target_fraction: float = 0.25, gamma: float = 0.05, f_e: np.ndarray | None = None
    ) -> None:
        """
        DeepSeek-V3 风格的无辅助损失动态偏置微调：
        根据当前专家的实际处理频次与目标均分负载的差异，动态微调 bias 抑制过热专家并激发冷门专家。
        """
        if f_e is not None:
            # 偏置更新: bias += gamma * (target - actual)
            delta_bias = gamma * (target_fraction - f_e)
            self.expert_bias += delta_bias


class MoELayer:
    """
    混合专家模型层 (MoE Layer)。
    将输入张量路由给多个独立的 ExpertFFN 并按门控权重融合输出。
    """

    def __init__(
        self,
        d_model: int = 32,
        d_ff: int = 64,
        num_experts: int = 4,
        top_k: int = 2,
        aux_loss_weight: float = 0.01,
        activation_type: str = "swiglu",
        seed: int = 42,
    ) -> None:
        self.d_model = d_model
        self.d_ff = d_ff
        self.num_experts = num_experts
        self.top_k = top_k
        self.aux_loss_weight = aux_loss_weight

        # 门控路由器
        self.router = TopKGating(
            d_model=d_model,
            num_experts=num_experts,
            top_k=top_k,
            aux_loss_weight=aux_loss_weight,
            seed=seed,
        )

        # 实例化各专家
        self.experts = [
            ExpertFFN(
                d_model=d_model, d_ff=d_ff, activation_type=activation_type, seed=seed + i * 10
            )
            for i in range(num_experts)
        ]

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, float, dict[str, Any]]:
        """
        MoE 前向传播。

        Args:
            x: 输入特征，形状 (Batch, Seq_Len, d_model) 或 (N, d_model)

        Returns:
            output: 加权合成特征，形状与 x 完全一致
            aux_loss: 辅助损失
            telemetry: 详细路由指标
        """
        orig_shape = x.shape
        x_2d = x.reshape(-1, self.d_model)
        num_tokens = x_2d.shape[0]

        topk_weights, topk_indices, aux_loss, gating_stats = self.router.forward(x_2d)

        output_2d = np.zeros_like(x_2d, dtype=np.float64)

        # 按专家分发 Token 批次进行前向计算并累加
        expert_token_counts = [0] * self.num_experts
        for e_idx, expert in enumerate(self.experts):
            # 查找当前专家被选中的 (token_idx, rank_k) 坐标
            matched_mask = topk_indices == e_idx
            matched_tokens_idx, matched_rank_k = np.where(matched_mask)

            if len(matched_tokens_idx) == 0:
                continue

            expert_token_counts[e_idx] = len(matched_tokens_idx)
            expert_in = x_2d[matched_tokens_idx]
            expert_out = expert.forward(expert_in)

            weights = topk_weights[matched_tokens_idx, matched_rank_k][:, None]
            output_2d[matched_tokens_idx] += weights * expert_out

        output = output_2d.reshape(orig_shape)

        telemetry = {
            "num_experts": self.num_experts,
            "top_k": self.top_k,
            "total_tokens": num_tokens,
            "expert_token_counts": expert_token_counts,
            "gating_stats": gating_stats,
            "aux_loss": aux_loss,
            "active_parameter_ratio": float(self.top_k / self.num_experts),
        }

        return output, aux_loss, telemetry
