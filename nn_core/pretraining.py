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

    def create_mlm_batch(self, token_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
        grad_w = np.dot(active_hidden.T, grad_output)  # (d_model, vocab_size)
        grad_b = np.sum(grad_output, axis=0)  # (vocab_size,)

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

    def create_clm_batch(self, token_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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

    def create_positive_pairs(
        self, embeddings: np.ndarray, noise_std: float = 0.1
    ) -> tuple[np.ndarray, np.ndarray]:
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

    def create_mae_batch(self, patches: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        patches: (B, num_patches, patch_dim)
        随机遮蔽 mask_ratio (默认 75%) 的图块
        返回 (masked_patches, mask_indices, unmasked_indices)
        """
        _B, N, _D = patches.shape
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

    def reconstruction_loss(
        self, pred_patches: np.ndarray, true_patches: np.ndarray, mask_indices: np.ndarray
    ) -> float:
        """仅在被遮蔽的 75% 图块上计算 MSE 损失"""
        diff = pred_patches[:, mask_indices] - true_patches[:, mask_indices]
        return float(np.mean(diff**2))


class PretrainingComparator:
    """
    预训练范式向下游任务迁移能力画像。
    """

    @staticmethod
    def get_transfer_scores() -> dict[str, dict[str, int]]:
        return {
            "MLM (BERT 完形填空)": {
                "文本分类": 94,
                "阅读理解": 90,
                "自回归生成": 35,
                "跨模态检索": 65,
            },
            "CLM (GPT 因果接龙)": {
                "文本分类": 80,
                "阅读理解": 78,
                "自回归生成": 96,
                "跨模态检索": 55,
            },
            "Contrastive (CLIP 对比)": {
                "文本分类": 72,
                "阅读理解": 60,
                "自回归生成": 40,
                "跨模态检索": 98,
            },
            "MAE (视觉掩码自编码)": {
                "文本分类": 88,
                "阅读理解": 50,
                "自回归生成": 30,
                "跨模态检索": 62,
            },
        }


class ScalingLawEngine:
    """
    大模型扩展定律计算引擎 (Scaling Laws for Neural Language Models)。

    基于 DeepMind Chinchilla (Hoffmann et al., 2022) 经验拟合公式与拉格朗日最优算力分配：
        $$L(N, D) = E + \\frac{A}{N^\\alpha} + \\frac{B}{D^\\beta}$$
        算力预算约束：$$C \\approx 6 N D$$

    最优分配解析解：
        $$N_{\\text{opt}}(C) = G \\left(\\frac{C}{6}\\right)^a, \\quad D_{\\text{opt}}(C) = \\frac{1}{G} \\left(\\frac{C}{6}\\right)^b$$
        其中 $a = \\frac{\\beta}{\\alpha + \\beta} \\approx 0.456$, $b = \\frac{\\alpha}{\\alpha + \\beta} \\approx 0.544$。
    """

    # Hoffmann et al. (2022) Table A3 拟合真实参数
    E = 1.6934  # 不可约熵损失 (Irreducible loss / 语言内在熵)
    A = 406.4  # 参数规模常数
    ALPHA = 0.3392
    B = 410.7  # 数据规模常数
    BETA = 0.2849

    @classmethod
    def compute_optimal_allocation(cls, compute_flops: float) -> dict:
        """
        根据给定的总浮点计算预算 (FLOPs)，计算 Chinchilla 最优参数量 N 与最优训练 Token 数 D。
        """
        C = float(compute_flops)
        a = cls.BETA / (cls.ALPHA + cls.BETA)  # ~0.456
        b = cls.ALPHA / (cls.ALPHA + cls.BETA)  # ~0.544
        G = ((cls.ALPHA * cls.A) / (cls.BETA * cls.B)) ** (1.0 / (cls.ALPHA + cls.BETA))  # ~1.300

        c_div_6 = C / 6.0
        n_opt = G * (c_div_6**a)
        d_opt = (1.0 / G) * (c_div_6**b)

        loss = cls.estimate_loss(n_opt, d_opt)

        # 换算硬件卡天数 (以 NVIDIA H100 SXM5 80GB 为基准，FP16/BF16 峰值 989 TFLOPs，取实际工业 MFU=45% -> 445 TFLOPs)
        h100_effective_flops_per_sec = 989e12 * 0.45
        gpu_seconds = C / h100_effective_flops_per_sec
        h100_gpu_days = gpu_seconds / 86400.0

        return {
            "compute_flops": C,
            "optimal_params_N": float(n_opt),
            "optimal_tokens_D": float(d_opt),
            "token_param_ratio": float(d_opt / n_opt),
            "predicted_loss": float(loss),
            "h100_gpu_days": float(h100_gpu_days),
        }

    @classmethod
    def estimate_loss(cls, N: float, D: float) -> float:
        """计算给定参数量 N 和 Token 数 D 下的预测交叉熵损失"""
        term_n = cls.A / (float(N) ** cls.ALPHA)
        term_d = cls.B / (float(D) ** cls.BETA)
        return float(cls.E + term_n + term_d)

    @staticmethod
    def get_historical_benchmarks() -> list[dict]:
        """工业界真实里程碑大模型在 Scaling 曲线上的实际位置与配置"""
        return [
            {
                "name": "GPT-3 (2020)",
                "params": 175e9,
                "tokens": 300e9,
                "flops": 6 * 175e9 * 300e9,  # 3.15e23
                "status": "严重欠训练 (Param Heavy / Under-trained)",
                "note": "早期受 Kaplan 定律影响，参数量过大而数据量不足",
            },
            {
                "name": "Chinchilla (2022)",
                "params": 70e9,
                "tokens": 1400e9,
                "flops": 6 * 70e9 * 1400e9,  # 5.88e23
                "status": "算力最优 (Compute-Optimal)",
                "note": "严格遵循 D ≈ 20N 黄金法则，70B 击败 175B GPT-3",
            },
            {
                "name": "LLaMA-1 (2023)",
                "params": 7e9,
                "tokens": 1000e9,
                "flops": 6 * 7e9 * 1000e9,  # 4.2e22
                "status": "超训练 (Over-trained)",
                "note": "牺牲预训练最优性，换取极致的下游轻量推理部署成本",
            },
            {
                "name": "LLaMA-3 (2024)",
                "params": 8e9,
                "tokens": 15000e9,
                "flops": 6 * 8e9 * 15000e9,  # 7.2e23
                "status": "极限超训练 (Extreme Over-trained)",
                "note": "15T Token 持续注水，单模型性能逼近上代 70B",
            },
            {
                "name": "DeepSeek-V3 (2024/2025)",
                "params": 671e9,  # 37B 激活 MoE
                "tokens": 14800e9,
                "flops": 6 * 37e9 * 14800e9,  # MoE 激活计算量
                "status": "稀疏 MoE 算力最优",
                "note": "通过 37B 激活实现前沿性能，极致压缩训练算力消耗",
            },
        ]


class SimpleBPE:
    """
    字节对编码 (Byte-Pair Encoding / BPE) 纯 NumPy/Python 算法引擎。
    展示大模型分词器从字符到子词 (Subword) 的动态合并与词表膨胀机制。
    """

    def __init__(self, vocab_size: int = 50) -> None:
        self.target_vocab_size = vocab_size
        self.merges: list[tuple[str, str]] = []
        self.vocab: dict[str, int] = {}

    def _get_stats(self, word_freqs: dict[tuple[str, ...], int]) -> dict[tuple[str, str], int]:
        """统计所有相邻子词对的共现频次"""
        pairs: dict[tuple[str, str], int] = {}
        for word, freq in word_freqs.items():
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                pairs[pair] = pairs.get(pair, 0) + freq
        return pairs

    def _merge_pair(
        self, pair: tuple[str, str], word_freqs: dict[tuple[str, ...], int]
    ) -> dict[tuple[str, ...], int]:
        """在所有单词切分序列中将目标字符对合并为新子词"""
        new_word_freqs: dict[tuple[str, ...], int] = {}
        bigram = pair
        for word, freq in word_freqs.items():
            new_word: list[str] = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == bigram[0] and word[i + 1] == bigram[1]:
                    new_word.append(bigram[0] + bigram[1])
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word_freqs[tuple(new_word)] = freq
        return new_word_freqs

    def train(self, texts: list[str]) -> list[dict]:
        """
        在输入语料上迭代训练 BPE 合并规则，返回每一步的合并记录。
        """
        # 1. 初始化词频字典，单词末尾附带词边界标记 '</w>'
        word_counts: dict[str, int] = {}
        for text in texts:
            for w in text.strip().split():
                word_counts[w] = word_counts.get(w, 0) + 1

        word_freqs: dict[tuple[str, ...], int] = {
            tuple([*list(w), "</w>"]): count for w, count in word_counts.items()
        }

        # 初始基础单字符词表
        base_chars = set()
        for w in word_freqs:
            for c in w:
                base_chars.add(c)

        self.vocab = {c: i for i, c in enumerate(sorted(list(base_chars)))}
        self.merges = []
        merge_history: list[dict] = []

        num_merges = max(0, self.target_vocab_size - len(self.vocab))
        for step in range(num_merges):
            pairs = self._get_stats(word_freqs)
            if not pairs:
                break
            # 贪心选取出现频次最高的相邻对
            best_pair = max(pairs, key=lambda pair: pairs[pair])
            best_freq = pairs[best_pair]
            if best_freq < 1:
                break

            word_freqs = self._merge_pair(best_pair, word_freqs)
            self.merges.append(best_pair)
            new_token = best_pair[0] + best_pair[1]
            self.vocab[new_token] = len(self.vocab)

            merge_history.append(
                {
                    "step": step + 1,
                    "merged_pair": f"'{best_pair[0]}' + '{best_pair[1]}'",
                    "new_token": new_token,
                    "frequency": best_freq,
                    "vocab_size": len(self.vocab),
                }
            )

        return merge_history

    def tokenize(self, text: str) -> list[str]:
        """利用训练好的 merge 规则切分文本"""
        tokens: list[str] = []
        for word in text.strip().split():
            splits = [*list(word), "</w>"]
            for pair in self.merges:
                i = 0
                new_splits = []
                while i < len(splits):
                    if i < len(splits) - 1 and splits[i] == pair[0] and splits[i + 1] == pair[1]:
                        new_splits.append(pair[0] + pair[1])
                        i += 2
                    else:
                        new_splits.append(splits[i])
                        i += 1
                splits = new_splits
            tokens.extend(splits)
        return tokens

    def get_compression_stats(self, text: str) -> dict:
        """评估 BPE 分词的压缩率与 Token 效率"""
        raw_chars = len(text.replace(" ", ""))
        tokens = self.tokenize(text)
        num_tokens = len(tokens)
        ratio = raw_chars / max(1, num_tokens)
        return {
            "raw_characters": raw_chars,
            "token_count": num_tokens,
            "compression_ratio": float(ratio),
            "tokens": tokens,
        }


class DataMixtureEngine:
    """
    前沿大模型预训练语料配比 (Data Mixture) 与清洗流水线标准。
    """

    @staticmethod
    def get_mixtures() -> dict[str, dict[str, float]]:
        """工业界经典模型的真实语料配比分布 (百分比)"""
        return {
            "LLaMA-3 (Meta 2024, 15T)": {
                "通用高质量网页 (CommonCrawl / FineWeb)": 50.0,
                "源代码与技术文档 (GitHub / StackOverflow)": 25.0,
                "学术论文与科学数据 (ArXiv / PubMed)": 10.0,
                "多语言语料 (30+ 种语言)": 10.0,
                "高难度数学与推理合成语料": 5.0,
            },
            "DeepSeek-V3 (2024/2025, 14.8T)": {
                "通用高质量网页 (清洗过滤语料)": 45.0,
                "高质量代码 (含代码执行轨迹)": 25.0,
                "中文与多语言本土高质量语料": 15.0,
                "数学与逻辑推理合成数据 (Math/Reasoning)": 10.0,
                "图书与学术期刊": 5.0,
            },
            "FineWeb-Edu (HuggingFace 2024)": {
                "高教育价值网页 (Edu Score >= 3)": 60.0,
                "STEM 理工科专业知识": 20.0,
                "人文社会与学术百科": 15.0,
                "高质量问答与教科书": 5.0,
            },
        }

    @staticmethod
    def get_cleaning_pipeline() -> list[dict]:
        """工业级预训练数据清洗 4 大阶段与核心规则"""
        return [
            {
                "stage": "Stage 1: 文本提取与语言识别",
                "rules": "HTML 标签清除、Trafilatura 核心正文提取、FastText/CLD3 语言分类器判定（置信度 > 0.65）。",
                "filter_rate": "过滤约 30% 杂质",
            },
            {
                "stage": "Stage 2: 启发式与质量过滤 (Heuristic Filtering)",
                "rules": "Gopher/C4 经典规则：行重复率 > 30% 丢弃、无标点或异常符号密度高丢弃、词均长度不在 3~10 之间丢弃、停用词占比过低丢弃。",
                "filter_rate": "过滤约 40% 低质内容",
            },
            {
                "stage": "Stage 3: 模糊与精确去重 (Deduplication)",
                "rules": "MinHash + LSH (Locality Sensitive Hashing) 算法：基于 5-gram Jaccard 相似度 > 0.8 消除跨页面模板与机器抓取重复文本。",
                "filter_rate": "消除约 20% 冗余",
            },
            {
                "stage": "Stage 4: 毒性内容与隐私脱敏 (PII & Toxicity)",
                "rules": "个人隐私（邮箱、电话、身份证、IP 地址）正则掩蔽脱敏；有毒有害分类器（Toxicity Classifier）过滤低俗暴力语料。",
                "filter_rate": "净化约 5% 高危数据",
            },
        ]
