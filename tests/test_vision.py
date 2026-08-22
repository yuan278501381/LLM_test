# Copyright (c) 2026 Yy1 (yuan278501381) | MIT License
"""
tests/test_vision.py - 视觉感知模块单元测试与梯度检验
"""

import numpy as np
import pytest

from nn_core.conv2d import Conv2D, MaxPool2D, im2col, col2im
from nn_core.vit import PatchEmbedding, VisionTransformer
from nn_core.clip import CLIPDualEncoder, contrastive_loss, get_pretrained_clip_data


def test_conv2d_forward_shapes():
    """测试 Conv2D 在不同 stride, padding 组合下的前向尺寸"""
    x = np.random.randn(2, 1, 8, 8)
    conv = Conv2D(in_channels=1, out_channels=3, kernel_size=3, stride=1, padding=0)
    out = conv.forward(x)
    assert out.shape == (2, 3, 6, 6)

    conv_padded = Conv2D(in_channels=1, out_channels=4, kernel_size=3, stride=2, padding=1)
    out_p = conv_padded.forward(x)
    assert out_p.shape == (2, 4, 4, 4)


def test_conv2d_gradient_central_difference():
    """使用双侧中心差分法验证 Conv2D 权重和输入的反向传播梯度"""
    np.random.seed(42)
    x = np.random.randn(1, 1, 4, 4)
    conv = Conv2D(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=0)
    
    # 解析梯度
    out = conv.forward(x)
    dout = np.ones_like(out)
    dx = conv.backward(dout)
    analytical_dw = conv.grad_weights.copy()
    analytical_dx = dx.copy()

    eps = 1e-5
    # 检验权重梯度
    numeric_dw = np.zeros_like(conv.weights)
    for idx in np.ndindex(conv.weights.shape):
        orig = conv.weights[idx]
        
        conv.weights[idx] = orig + eps
        out1 = conv.forward(x)
        loss1 = np.sum(out1)

        conv.weights[idx] = orig - eps
        out2 = conv.forward(x)
        loss2 = np.sum(out2)

        conv.weights[idx] = orig
        numeric_dw[idx] = (loss1 - loss2) / (2 * eps)

    rel_err_w = np.max(np.abs(analytical_dw - numeric_dw) / (np.abs(analytical_dw) + np.abs(numeric_dw) + 1e-8))
    assert rel_err_w < 1e-4, f"Conv2D 权重梯度相对误差过大: {rel_err_w}"


def test_maxpool2d_forward_backward():
    """测试 MaxPool2D 前向池化与反向路由"""
    x = np.array([[[[1.0, 3.0, 2.0, 4.0],
                    [5.0, 6.0, 7.0, 8.0],
                    [9.0, 2.0, 3.0, 1.0],
                    [4.0, 8.0, 6.0, 5.0]]]])
    pool = MaxPool2D(pool_size=2, stride=2)
    out = pool.forward(x)
    assert out.shape == (1, 1, 2, 2)
    np.testing.assert_allclose(out[0, 0], [[6.0, 8.0], [9.0, 6.0]])

    dout = np.ones_like(out)
    dx = pool.backward(dout)
    assert dx.shape == x.shape
    # 最大值位置为 1，其余为 0
    assert dx[0, 0, 1, 1] == 1.0
    assert dx[0, 0, 1, 3] == 1.0
    assert dx[0, 0, 2, 0] == 1.0
    assert dx[0, 0, 0, 0] == 0.0


def test_patch_embedding_shape():
    """测试 PatchEmbedding 的 Token 数量与维度"""
    img = np.random.randn(2, 1, 32, 32)
    patch_embed = PatchEmbedding(img_size=32, patch_size=8, in_channels=1, d_model=32)
    tokens = patch_embed.forward(img)
    # (32/8)^2 = 16 patches + 1 [CLS] = 17 tokens
    assert tokens.shape == (2, 17, 32)


def test_vit_forward():
    """测试 VisionTransformer 完整前向分类"""
    img = np.random.randn(2, 1, 32, 32)
    vit = VisionTransformer(img_size=32, patch_size=8, in_channels=1, d_model=32, num_classes=5)
    logits, attns = vit.forward(img)
    assert logits.shape == (2, 5)
    assert len(attns) == 2  # 2 层 block


def test_clip_similarity_and_loss():
    """测试 CLIP 对比学习语义对齐与 InfoNCE 损失"""
    labels, texts, sim = get_pretrained_clip_data()
    assert sim.shape == (8, 8)
    # 验证对角线上的同类相似度显著高于非对角线
    for i in range(8):
        assert sim[i, i] == np.max(sim[i, :]), f"第 {i} 个概念与自身的图文相似度未达到最大"

    loss = contrastive_loss(sim, temperature=0.07)
    assert loss >= 0.0
