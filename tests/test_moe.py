# Copyright (c) 2026 Yy1 (yuan278501381) | MIT License
"""
tests/test_moe.py - 混合专家模型 (MoE) 与动态门控路由单元测试集
"""

import numpy as np

from nn_core.moe import ExpertFFN, MoELayer, TopKGating


def test_expert_ffn_forward():
    """验证单个专家前馈网络的前向传播与形状一致性"""
    x = np.random.randn(5, 16)
    expert_swiglu = ExpertFFN(d_model=16, d_ff=32, activation_type="swiglu")
    out_swiglu = expert_swiglu.forward(x)
    assert out_swiglu.shape == (5, 16)

    expert_relu = ExpertFFN(d_model=16, d_ff=32, activation_type="relu")
    out_relu = expert_relu.forward(x)
    assert out_relu.shape == (5, 16)


def test_topk_gating_weights_and_aux_loss():
    """验证稀疏门控路由器的 Top-K 权重归一化与辅助损失"""
    num_tokens = 20
    d_model = 16
    num_experts = 4
    top_k = 2

    router = TopKGating(d_model=d_model, num_experts=num_experts, top_k=top_k, aux_loss_weight=0.01)
    x = np.random.randn(num_tokens, d_model)

    weights, indices, aux_loss, stats = router.forward(x)

    assert weights.shape == (num_tokens, top_k)
    assert indices.shape == (num_tokens, top_k)
    # 验证选中的 Top-K 权重和为 1.0 (归一化特性)
    np.testing.assert_allclose(np.sum(weights, axis=-1), np.ones(num_tokens), rtol=1e-5)
    assert aux_loss >= 0.0
    assert "expert_dispatch_fractions" in stats
    assert len(stats["expert_dispatch_fractions"]) == num_experts


def test_moe_layer_forward_2d_and_3d():
    """验证 MoE 层的端到端前向传播与多维批次处理"""
    moe = MoELayer(d_model=16, d_ff=32, num_experts=4, top_k=2, seed=42)

    # 2D 输入: (num_tokens, d_model)
    x_2d = np.random.randn(10, 16)
    out_2d, _aux_loss_2d, tele_2d = moe.forward(x_2d)
    assert out_2d.shape == (10, 16)
    assert tele_2d["total_tokens"] == 10
    assert tele_2d["active_parameter_ratio"] == 0.5

    # 3D 输入: (batch_size, seq_len, d_model)
    x_3d = np.random.randn(2, 8, 16)
    out_3d, _aux_loss_3d, tele_3d = moe.forward(x_3d)
    assert out_3d.shape == (2, 8, 16)
    assert tele_3d["total_tokens"] == 16


def test_moe_dynamic_bias_adjustment():
    """验证动态偏置自平衡调整机制"""
    router = TopKGating(d_model=16, num_experts=4, top_k=1, aux_loss_weight=0.01)
    # 模拟极端不平衡的分配
    f_e = np.array([0.9, 0.1, 0.0, 0.0])
    router.update_dynamic_bias(target_fraction=0.25, gamma=0.1, f_e=f_e)

    # 专家 0 过热，bias 应被压降为负值；专家 2/3 冷门，bias 应被提升为正值
    assert router.expert_bias[0] < 0.0
    assert router.expert_bias[2] > 0.0
