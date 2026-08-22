# Copyright (c) 2026 Yy1 (yuan278501381) | MIT License
"""
nn_core.vit - Vision Transformer (ViT) 纯 NumPy 视觉特征提取与分类模型

包含：
- `PatchEmbedding`: 图像切片、展平与线性 Token 化投影层（支持 [CLS] Token 与位置编码）
- `VisionTransformer`: 教学级纯 NumPy 视觉 Transformer 架构
"""

import logging

import numpy as np

from nn_core.layernorm import LayerNorm
from nn_core.transformer import TransformerBlock

logger = logging.getLogger("nn_core.vit")


class PatchEmbedding:
    """
    Vision Transformer 图像切片与嵌入层 (Patch Embedding)。

    数学映射：
        将图像 $X \\in \\mathbb{R}^{B \\times C \\times H \\times W}$ 切割为 $N = (H/P) \\times (W/P)$ 个
        大小为 $P \\times P$ 的图块，展平后经线性矩阵 $W_{proj}$ 投影为 $D$ 维向量序列，
        并在序列起始位置拼接可学习的 $[\\text{CLS}]$ 标记，最后注入 1D 位置编码：
        $$Z_0 = [x_{class}; x_p^1 E; x_p^2 E; \\dots; x_p^N E] + E_{pos}$$
    """

    def __init__(
        self,
        img_size: int = 32,
        patch_size: int = 8,
        in_channels: int = 1,
        d_model: int = 32,
    ) -> None:
        assert img_size % patch_size == 0, "img_size 必须能被 patch_size 整除"
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.d_model = d_model

        self.num_patches = (img_size // patch_size) ** 2
        self.patch_dim = in_channels * patch_size * patch_size

        # 线性投影权重与偏置
        self.proj_weights = np.random.randn(self.patch_dim, d_model) * np.sqrt(2.0 / self.patch_dim)
        self.proj_bias = np.zeros(d_model)

        # [CLS] Token (可学习参数)
        self.cls_token = np.random.randn(1, 1, d_model) * 0.02

        # 位置编码矩阵 (N + 1, d_model) - 预计算 1D 正弦位置编码
        self.pos_embedding = self._build_sinusoidal_pos_embedding(self.num_patches + 1, d_model)

    def _build_sinusoidal_pos_embedding(self, num_tokens: int, d_model: int) -> np.ndarray:
        pos = np.arange(num_tokens)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        pe = np.zeros((1, num_tokens, d_model))
        pe[0, :, 0::2] = np.sin(pos * div_term)
        pe[0, :, 1::2] = np.cos(pos * div_term)
        return pe

    def forward(self, img: np.ndarray) -> np.ndarray:
        """
        前向图块展平与投影。
        img shape: (B, C, H, W)
        返回 token 序列: (B, num_patches + 1, d_model)
        """
        B, C, H, W = img.shape
        P = self.patch_size
        gh = H // P
        gw = W // P

        # 空间图块切分: (B, C, gh, P, gw, P) -> (B, gh*gw, C*P*P)
        patches = img.reshape(B, C, gh, P, gw, P).transpose(0, 2, 4, 1, 3, 5)
        patches = patches.reshape(B, gh * gw, C * P * P)

        # 线性投影: (B, num_patches, d_model)
        embeddings = np.dot(patches, self.proj_weights) + self.proj_bias

        # 拼接 [CLS] token: (B, 1 + num_patches, d_model)
        cls_tokens = np.repeat(self.cls_token, B, axis=0)
        x = np.concatenate([cls_tokens, embeddings], axis=1)

        # 叠加位置编码
        x = x + self.pos_embedding
        return x


class VisionTransformer:
    """
    教学级微型 Vision Transformer (ViT) 模型。
    """

    def __init__(
        self,
        img_size: int = 32,
        patch_size: int = 8,
        in_channels: int = 1,
        d_model: int = 32,
        num_heads: int = 2,
        num_layers: int = 2,
        d_ff: int = 64,
        num_classes: int = 10,
    ) -> None:
        self.patch_embed = PatchEmbedding(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            d_model=d_model,
        )
        self.blocks = [
            TransformerBlock(d_model=d_model, num_heads=num_heads, d_ff=d_ff)
            for _ in range(num_layers)
        ]
        self.ln = LayerNorm(d_model)
        self.head_w = np.random.randn(d_model, num_classes) * np.sqrt(2.0 / d_model)
        self.head_b = np.zeros(num_classes)

    def forward(self, img: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
        """
        前向图像分类与注意力图提取。
        返回 (logits, attention_weights_list)
        logits shape: (B, num_classes)
        """
        x = self.patch_embed.forward(img)
        all_attns = []

        for block in self.blocks:
            x, attn_weights = block.forward(x)
            all_attns.append(attn_weights)

        x = self.ln.forward(x)
        # 提取 [CLS] token 表征作为整图特征
        cls_feat = x[:, 0, :]
        logits = np.dot(cls_feat, self.head_w) + self.head_b
        return logits, all_attns
