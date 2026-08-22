# Copyright (c) 2026 Yy1 (yuan278501381) | MIT License
"""
nn_core.clip - 对比语言-图像预训练 (CLIP) 双塔架构与对比学习损失

包含：
- `CLIPDualEncoder`: 图文跨模态对齐双塔网络
- `contrastive_loss`: InfoNCE 对称对比学习损失函数
- `get_pretrained_clip_data`: 预置教学级图文对齐语义嵌入数据集
"""

import logging

import numpy as np

from nn_core.embeddings import Embedding, PositionalEncoding
from nn_core.transformer import TransformerBlock
from nn_core.vit import VisionTransformer

logger = logging.getLogger("nn_core.clip")


def contrastive_loss(similarity_matrix: np.ndarray, temperature: float = 0.07) -> float:
    """
    计算 CLIP 的对称 InfoNCE 对比学习损失。

    数学公式：
        $$L_{img} = -\\frac{1}{N} \\sum_{i=1}^N \\log \\frac{e^{S_{i,i}/\\tau}}{\\sum_j e^{S_{i,j}/\\tau}}$$
        $$L_{txt} = -\\frac{1}{N} \\sum_{i=1}^N \\log \\frac{e^{S_{i,i}/\\tau}}{\\sum_j e^{S_{j,i}/\\tau}}$$
        $$L_{total} = \\frac{1}{2} (L_{img} + L_{txt})$$
    """
    N = similarity_matrix.shape[0]
    logits = similarity_matrix / temperature
    labels = np.arange(N)

    # 图像到文本损失 (按行 Softmax)
    exp_row = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    prob_row = exp_row / np.sum(exp_row, axis=1, keepdims=True)
    loss_img = -np.mean(np.log(prob_row[np.arange(N), labels] + 1e-12))

    # 文本到图像损失 (按列 Softmax)
    exp_col = np.exp(logits - np.max(logits, axis=0, keepdims=True))
    prob_col = exp_col / np.sum(exp_col, axis=0, keepdims=True)
    loss_txt = -np.mean(np.log(prob_col[labels, np.arange(N)] + 1e-12))

    return float(0.5 * (loss_img + loss_txt))


class CLIPDualEncoder:
    """
    纯 NumPy 教学级 CLIP 双塔模型。
    """

    def __init__(
        self,
        vocab_size: int = 100,
        img_size: int = 32,
        patch_size: int = 8,
        d_model: int = 32,
        embed_dim: int = 16,
    ) -> None:
        self.d_model = d_model
        self.embed_dim = embed_dim

        # 文本塔 (Text Encoder)
        self.text_embed = Embedding(vocab_size=vocab_size, d_model=d_model)
        self.text_pos = PositionalEncoding(max_len=32, d_model=d_model)
        self.text_block = TransformerBlock(d_model=d_model, num_heads=2, d_ff=64)
        self.text_proj = np.random.randn(d_model, embed_dim) * np.sqrt(2.0 / d_model)

        # 图像塔 (Image Encoder)
        self.image_vit = VisionTransformer(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=1,
            d_model=d_model,
            num_heads=2,
            num_layers=1,
            num_classes=embed_dim,
        )

    def encode_text(self, token_ids: np.ndarray) -> np.ndarray:
        """提取文本特征并 L2 归一化"""
        x = self.text_embed.forward(token_ids)
        x = self.text_pos.forward(x)
        x, _ = self.text_block.forward(x)
        # 取平均池化作为句向量
        feat = np.mean(x, axis=1) @ self.text_proj
        norm = np.linalg.norm(feat, axis=1, keepdims=True) + 1e-12
        return feat / norm

    def encode_image(self, img: np.ndarray) -> np.ndarray:
        """提取图像特征并 L2 归一化"""
        feat, _ = self.image_vit.forward(img)
        norm = np.linalg.norm(feat, axis=1, keepdims=True) + 1e-12
        return feat / norm

    def compute_similarity(self, img_embeds: np.ndarray, txt_embeds: np.ndarray) -> np.ndarray:
        """
        计算图像与文本嵌入向量的余弦相似度矩阵: $S = I \\cdot T^T$
        返回 shape: (N_img, N_txt)
        """
        return np.dot(img_embeds, txt_embeds.T)


def get_pretrained_clip_data() -> tuple[list[str], list[str], np.ndarray]:
    """
    返回预置的 8 组图文概念、文本描述与对齐的余弦相似度矩阵（用于教学与可视化演示）。
    """
    labels = [
        "Cat (猫咪)",
        "Dog (小狗)",
        "Red Car (红色跑车)",
        "Sunset (日落风景)",
        "Happy Face (微笑人脸)",
        "Snow Mountain (雪山)",
        "Coffee Cup (咖啡杯)",
        "Blue Ocean (蔚蓝海洋)",
    ]
    texts = [
        "a photo of a cute domestic cat",
        "a photo of a playing dog in park",
        "a fast red sports car driving",
        "a golden sunset over the horizon",
        "a portrait of a happy smiling person",
        "a cold mountain covered with white snow",
        "a steaming hot cup of espresso coffee",
        "deep blue ocean water with gentle waves",
    ]

    # 构建每个类别的正交基础嵌入向量
    N = 8
    dim = 16
    np.random.seed(42)

    # 主导基底 (每个类别占领一个主要坐标轴，辅以少量语义关联)
    base_matrix = np.eye(N, dim) * 2.0
    # 增加语义相关性：cat与dog(0和1)有0.4相关，mountain与ocean(5和7)有0.3相关
    base_matrix[0, 1] += 0.4
    base_matrix[1, 0] += 0.4
    base_matrix[5, 7] += 0.3
    base_matrix[7, 5] += 0.3

    img_vecs = []
    txt_vecs = []

    for i in range(N):
        iv = base_matrix[i] + np.random.randn(dim) * 0.05
        tv = base_matrix[i] + np.random.randn(dim) * 0.05
        img_vecs.append(iv / np.linalg.norm(iv))
        txt_vecs.append(tv / np.linalg.norm(tv))

    img_mat = np.array(img_vecs)
    txt_mat = np.array(txt_vecs)
    similarity = np.dot(img_mat, txt_mat.T)
    return labels, texts, similarity
