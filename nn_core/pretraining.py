# Copyright (c) 2026 Yy1 (yuan278501381) | MIT License
"""
nn_core.pretraining - 预训练范式全景计算引擎 (MLM / CLM / 对比学习 / MAE)

包含：
- `MaskedLanguageModel` (MLM / BERT 式双向掩码完形填空预训练与真实梯度更新)
- `CausalLanguageModel` (CLM / GPT 式单向自回归因果接龙预训练与真实梯度更新)
- `ContrastiveLearning` (SimCLR / CLIP 式正负样本对 NT-Xent 对比学习)
- `MaskedAutoEncoder` (MAE 视觉高比例图块掩码自编码重建)
- `PretrainingComparator` (预训练范式向下游任务迁移能力画像模拟器)
"""

import logging
from typing import Optional, Tuple
import numpy as np

from nn_core.attention import causal_mask
from nn_core.embeddings import Embedding, PositionalEncoding
from nn_core.transformer import TransformerBlock

logger = logging.getLogger("nn_core.pretraining")


class MaskedLanguageModel:
    """
    BERT 式双向掩码语言模型 (Masked Language Model / MLM)。

    预训练目标：
        随机遮蔽 ~15% 的输入词汇，迫使模型利用左右双向全量上下文推断被遮蔽词：
        $$L_{MLM} = -\\sum_{i \\in \\text{Masked}} \\log p(w_i | w_{\\setminus i})$$
    """

    def __init__(
        self,
        vocab_size: int = 100,
        d_model: int = 32,
        num_heads: int = 2,
        num_layers: int = 1,
        mask_token_id: int = 0,
        mask_ratio: float = 0.15,
    ) -> None:
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.mask_token_id = mask_token_id
        self.mask_ratio = mask_ratio

        self.embed = Embedding(vocab_size=vocab_size, d_model=d_model)
        self.pos = PositionalEncoding(max_len=64, d_model=d_model)
        self.blocks = [
            TransformerBlock(d_model=d_model, num_heads=num_heads, d_ff=d_model * 2)
            for _ in range(num_layers)
        ]
        # 预测头 (Head)
        self.head_w = np.random.randn(d_model, vocab_size) * np.sqrt(2.0 / d_model)
        self.head_b = np.zeros(vocab_size)

    def create_mlm_batch(self, token_ids: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        根据标准 BERT 策略创建掩码样本：
        - 80% 替换为 [MASK]
        - 10% 替换为随机 Token
        - 10% 保持原词不变
        返回 (masked_ids, labels, mask_positions)
        """
        masked_ids = token_ids.copy()
        labels = np.full_like(token_ids, -100)  # -100 忽略计算损失
        seq_len = token_ids.shape[1]
        
        num_to_mask = max(1, int(seq_len * self.mask_ratio))
        mask_indices = np.random.choice(seq_len, size=num_to_mask, replace=False)
        mask_positions = np.zeros(seq_len, dtype=bool)
        mask_positions[mask_indices] = True

        for idx in mask_indices:
            labels[0, idx] = token_ids[0, idx]
            prob = np.random.rand()
            if prob < 0.8:
                masked_ids[0, idx] = self.mask_token_id
            elif prob < 0.9:
                masked_ids[0, idx] = np.random.randint(1, self.vocab_size)
            # 剩余 10% 保持不变

        return masked_ids, labels, mask_positions

    def forward(self, masked_ids: np.ndarray) -> np.ndarray:
        """
        前向双向特征提取与全词表 Logits 计算。
        返回 logits shape: (B, seq_len, vocab_size)
        内部缓存 Transformer 最终隐藏层输出 _last_hidden 用于反向传播。
        """
        x = self.embed.forward(masked_ids)
        x = self.pos.forward(x)
        # MLM 不使用因果掩码 (全向双向注意力)
        for block in self.blocks:
            x, _ = block.forward(x, mask=None)
        self._last_hidden = x  # 缓存用于真实梯度计算
        logits = np.dot(x, self.head_w) + self.head_b
        return logits

    def mlm_loss(self, logits: np.ndarray, labels: np.ndarray, mask_positions: np.ndarray) -> float:
        """仅在被遮蔽位置计算交叉熵损失"""
        active_logits = logits[0, mask_positions]  # (num_masked, vocab_size)
        active_labels = labels[0, mask_positions]  # (num_masked,)

        exp_l = np.exp(active_logits - np.max(active_logits, axis=-1, keepdims=True))
        probs = exp_l / (np.sum(exp_l, axis=-1, keepdims=True) + 1e-12)

        n_masked = len(active_labels)
        loss = -np.mean(np.log(probs[np.arange(n_masked), active_labels] + 1e-12))
        return float(loss)

    def train_step(self, token_ids: np.ndarray, lr: float = 0.05) -> float:
        """
        微型单步真实梯度训练更新。

        反向传播解析梯度推导：
            Softmax 残差: grad_output = P - Y  (shape: num_masked × vocab_size)
            分类头权重梯度: dW = H_active.T @ grad_output / N_masked
            分类头偏置梯度: db = grad_output.mean(axis=0)
        """
        masked_ids, labels, mask_positions = self.create_mlm_batch(token_ids)
        logits = self.forward(masked_ids)
        loss = self.mlm_loss(logits, labels, mask_positions)

        # ---- 真实解析梯度反向传播 ----
        # 1. 计算 Softmax 交叉熵对 Logits 的梯度: dL/dz = P - Y
        active_logits = logits[0, mask_positions]
        active_labels = labels[0, mask_positions]
        n_masked = len(active_labels)

        exp_l = np.exp(active_logits - np.max(active_logits, axis=-1, keepdims=True))
        grad_output = exp_l / (np.sum(exp_l, axis=-1, keepdims=True) + 1e-12)
        grad_output[np.arange(n_masked), active_labels] -= 1.0
        grad_output /= n_masked  # 均值化

        # 2. 利用缓存的隐藏层激活计算分类头权重的真实梯度
        #    logits = H @ W + b  =>  dW = H.T @ dL/dz,  db = sum(dL/dz)
        active_hidden = self._last_hidden[0, mask_positions]  # (num_masked, d_model)
        grad_w = np.dot(active_hidden.T, grad_output)        # (d_model, vocab_size)
        grad_b = np.sum(grad_output, axis=0)                 # (vocab_size,)

        # 3. SGD 真实梯度下降更新
        self.head_w -= lr * grad_w
        self.head_b -= lr * grad_b
        return loss


class CausalLanguageModel:
    """
    GPT 式单向自回归因果语言模型 (Causal Language Model / CLM)。
    预训练目标：从左往右自回归预测下一个词 $L_{CLM} = -\\sum_{t=1}^T \\log p(w_t | w_{<t})$。
    """

    def __init__(
        self,
        vocab_size: int = 100,
        d_model: int = 32,
        num_heads: int = 2,
        num_layers: int = 1,
    ) -> None:
        self.vocab_size = vocab_size
        self.d_model = d_model

        self.embed = Embedding(vocab_size=vocab_size, d_model=d_model)
        self.pos = PositionalEncoding(max_len=64, d_model=d_model)
        self.blocks = [
            TransformerBlock(d_model=d_model, num_heads=num_heads, d_ff=d_model * 2)
            for _ in range(num_layers)
        ]
        self.head_w = np.random.randn(d_model, vocab_size) * np.sqrt(2.0 / d_model)
        self.head_b = np.zeros(vocab_size)

    def create_clm_batch(self, token_ids: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """输入为 tokens[:-1], 预测目标为 tokens[1:]"""
        x_input = token_ids[:, :-1]
        y_target = token_ids[:, 1:]
        return x_input, y_target

    def forward(self, input_ids: np.ndarray) -> np.ndarray:
        """带下三角因果掩码的前向传播"""
        seq_len = input_ids.shape[1]
        x = self.embed.forward(input_ids)
        x = self.pos.forward(x)
        mask = causal_mask(seq_len)
        for block in self.blocks:
            x, _ = block.forward(x, mask=mask)
        self._last_hidden = x  # 缓存用于真实梯度计算
        logits = np.dot(x, self.head_w) + self.head_b
        return logits

    def clm_loss(self, logits: np.ndarray, targets: np.ndarray) -> float:
        """所有时间步的 Next-Token 交叉熵损失"""
        flat_logits = logits.reshape(-1, self.vocab_size)
        flat_targets = targets.reshape(-1)

        exp_l = np.exp(flat_logits - np.max(flat_logits, axis=-1, keepdims=True))
        probs = exp_l / (np.sum(exp_l, axis=-1, keepdims=True) + 1e-12)

        loss = -np.mean(np.log(probs[np.arange(len(flat_targets)), flat_targets] + 1e-12))
        return float(loss)

    def train_step(self, token_ids: np.ndarray, lr: float = 0.05) -> float:
        """
        微型单步自回归真实梯度训练更新。

        反向传播解析梯度推导：
            Softmax 残差: grad_output = P - Y  (shape: N_tokens × vocab_size)
            分类头权重梯度: dW = H_flat.T @ grad_output / N_tokens
            分类头偏置梯度: db = grad_output.mean(axis=0)
        """
        x_in, y_tgt = self.create_clm_batch(token_ids)
        logits = self.forward(x_in)
        loss = self.clm_loss(logits, y_tgt)

        # ---- 真实解析梯度反向传播 ----
        flat_logits = logits.reshape(-1, self.vocab_size)
        flat_targets = y_tgt.reshape(-1)
        n_tokens = len(flat_targets)

        exp_l = np.exp(flat_logits - np.max(flat_logits, axis=-1, keepdims=True))
        grad_output = exp_l / (np.sum(exp_l, axis=-1, keepdims=True) + 1e-12)
        grad_output[np.arange(n_tokens), flat_targets] -= 1.0
        grad_output /= n_tokens  # 均值化

        flat_hidden = self._last_hidden.reshape(-1, self.d_model)
        grad_w = np.dot(flat_hidden.T, grad_output)
        grad_b = np.sum(grad_output, axis=0)

        # SGD 真实梯度下降更新
        self.head_w -= lr * grad_w
        self.head_b -= lr * grad_b
        return loss


class ContrastiveLearning:
    """
    对比学习 (Contrastive Learning / NT-Xent Loss)。
    """

    def __init__(self, d_model: int = 32) -> None:
        self.d_model = d_model

    def create_positive_pairs(self, embeddings: np.ndarray, noise_std: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
        """通过高斯数据增强构建同源正样本对 (z_i, z_j)"""
        z_i = embeddings + np.random.randn(*embeddings.shape) * noise_std
        z_j = embeddings + np.random.randn(*embeddings.shape) * noise_std
        # L2 归一化
        z_i = z_i / (np.linalg.norm(z_i, axis=1, keepdims=True) + 1e-12)
        z_j = z_j / (np.linalg.norm(z_j, axis=1, keepdims=True) + 1e-12)
        return z_i, z_j

    def nt_xent_loss(self, z_i: np.ndarray, z_j: np.ndarray, temperature: float = 0.5) -> float:
        """
        NT-Xent 对比损失 (Normalized Temperature-scaled Cross Entropy Loss)。
        """
        N = z_i.shape[0]
        z = np.concatenate([z_i, z_j], axis=0)  # (2N, D)
        sim_matrix = np.dot(z, z.T) / temperature

        # 屏蔽自身对自身相似度
        np.fill_diagonal(sim_matrix, -1e9)

        # 正样本目标索引: (i, i+N) 与 (i+N, i)
        labels = np.concatenate([np.arange(N) + N, np.arange(N)])

        exp_s = np.exp(sim_matrix - np.max(sim_matrix, axis=-1, keepdims=True))
        probs = exp_s / (np.sum(exp_s, axis=-1, keepdims=True) + 1e-12)

        loss = -np.mean(np.log(probs[np.arange(2 * N), labels] + 1e-12))
        return float(loss)


class MaskedAutoEncoder:
    """
    MAE (Masked Autoencoder) 视觉高比例图块掩码自编码器。
    """

    def __init__(self, num_patches: int = 16, d_model: int = 32, mask_ratio: float = 0.75) -> None:
        self.num_patches = num_patches
        self.d_model = d_model
        self.mask_ratio = mask_ratio

        # 简单解码重建 MLP
        self.decoder = np.random.randn(d_model, d_model) * 0.1

    def create_mae_batch(self, patches: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        patches: (B, num_patches, patch_dim)
        随机遮蔽 mask_ratio (默认 75%) 的图块
        返回 (masked_patches, mask_indices, unmasked_indices)
        """
        B, N, D = patches.shape
        num_masked = int(N * self.mask_ratio)
        perm = np.random.permutation(N)
        mask_indices = perm[:num_masked]
        unmasked_indices = perm[num_masked:]

        masked_patches = patches.copy()
        masked_patches[:, mask_indices] = 0.0  # 遮蔽置零
        return masked_patches, mask_indices, unmasked_indices

    def reconstruct(self, masked_patches: np.ndarray) -> np.ndarray:
        """重建全图图块"""
        # 简单模拟自注意力全局补全
        return masked_patches + np.dot(masked_patches, self.decoder) * 0.5

    def reconstruction_loss(self, pred_patches: np.ndarray, true_patches: np.ndarray, mask_indices: np.ndarray) -> float:
        """仅在被遮蔽的 75% 图块上计算 MSE 损失"""
        diff = pred_patches[:, mask_indices] - true_patches[:, mask_indices]
        return float(np.mean(diff ** 2))


class PretrainingComparator:
    """
    预训练范式向下游任务迁移能力画像。
    """

    @staticmethod
    def get_transfer_scores() -> dict[str, dict[str, int]]:
        return {
            "MLM (BERT 完形填空)": {"文本分类": 94, "阅读理解": 90, "自回归生成": 35, "跨模态检索": 65},
            "CLM (GPT 因果接龙)": {"文本分类": 80, "阅读理解": 78, "自回归生成": 96, "跨模态检索": 55},
            "Contrastive (CLIP 对比)": {"文本分类": 72, "阅读理解": 60, "自回归生成": 40, "跨模态检索": 98},
            "MAE (视觉掩码自编码)": {"文本分类": 88, "阅读理解": 50, "自回归生成": 30, "跨模态检索": 62},
        }
