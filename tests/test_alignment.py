# Copyright (c) 2026 Yy1 (yuan278501381) | MIT License
"""
tests/test_alignment.py - 后训练与对齐工程模块单元测试与梯度检验
"""

import numpy as np
import pytest

from nn_core.lora import LoRALayer, compute_param_savings
from nn_core.posttraining import AlignmentPipeline, generate_before_after_examples
from nn_core.rlhf import DPOLoss, PPOClipObjective, RewardModel


def test_reward_model_and_preference_loss():
    """测试奖励模型与 Bradley-Terry 偏好损失"""
    rm = RewardModel(input_dim=16, hidden_dim=32)
    x = np.random.randn(2, 16)
    scores = rm.forward(x)
    assert scores.shape == (2, 1)

    # chosen 得分高于 rejected 时，loss 应显著小于反向情况
    loss_good = RewardModel.preference_loss(reward_chosen=3.0, reward_rejected=-1.0)
    loss_bad = RewardModel.preference_loss(reward_chosen=-1.0, reward_rejected=3.0)
    assert loss_good < loss_bad
    assert loss_good > 0.0


def test_ppo_clip_objective():
    """测试 PPO 概率比率与截断目标"""
    ppo = PPOClipObjective(epsilon=0.2)
    log_p_new = np.array([0.0, 0.5])
    log_p_old = np.array([0.0, 0.0])
    ratio = ppo.compute_ratio(log_p_new, log_p_old)
    assert ratio[0] == 1.0
    assert ratio[1] > 1.0

    adv = np.array([1.0, 1.0])
    clipped_obj = ppo.clip_objective(ratio, adv)
    assert clipped_obj.shape == adv.shape
    # ratio[1] 约为 1.648，由于 eps=0.2 截断上限为 1.2
    assert clipped_obj[1] == pytest.approx(1.2, rel=1e-3)

    traj = PPOClipObjective.simulate_rlhf_trajectory(n_steps=15)
    assert len(traj["step"]) == 15
    assert traj["reward"][-1] > traj["reward"][0]


def test_dpo_loss():
    """测试 DPO 隐式奖励闭式损失"""
    dpo = DPOLoss(beta=0.1)
    # 当策略模型更偏好 chosen (pi_w > ref_w, pi_l < ref_l) 时，loss 较低
    loss_good = dpo.forward(
        pi_logprobs_w=-0.2, pi_logprobs_l=-1.5, ref_logprobs_w=-0.8, ref_logprobs_l=-0.8
    )
    loss_bad = dpo.forward(
        pi_logprobs_w=-1.5, pi_logprobs_l=-0.2, ref_logprobs_w=-0.8, ref_logprobs_l=-0.8
    )
    assert loss_good < loss_bad
    assert loss_good > 0.0


def test_lora_layer_forward_backward_merge():
    """测试 LoRA 旁路运算、反向传播与零延迟权重融合"""
    d_in = 16
    d_out = 16
    rank = 4
    W0 = np.random.randn(d_in, d_out)
    lora = LoRALayer(original_weight=W0, rank=rank, alpha=4.0)

    x = np.random.randn(2, d_in)
    # 初始状态 B=0，故 forward(x) 严格等于 x @ W0
    out_init = lora.forward(x)
    np.testing.assert_allclose(out_init, np.dot(x, W0))

    # 手动给 B 赋值扰动测试
    lora.B = np.random.randn(rank, d_out) * 0.1
    out_lora = lora.forward(x)
    dout = np.ones_like(out_lora)
    dx = lora.backward(dout)
    assert dx.shape == x.shape
    assert lora.grad_A.shape == (d_in, rank)
    assert lora.grad_B.shape == (rank, d_out)

    # 验证权重融合 merge() 结果与 forward 完全一致
    W_merged = lora.merge()
    assert W_merged.shape == (d_in, d_out)
    out_merged = np.dot(x, W_merged)
    np.testing.assert_allclose(out_lora, out_merged, rtol=1e-5)

    stats = compute_param_savings(d_model=512, rank=4)
    assert stats["compression_ratio"] == 64.0
    assert stats["saved_percent"] > 98.0


def test_posttraining_alignment_pipeline():
    """测试能力画像与对比示例库"""
    stages = AlignmentPipeline.get_all_stages()
    assert len(stages) == 4
    sft_scores = AlignmentPipeline.get_stage_scores("SFT")
    assert "有用性" in sft_scores
    assert sft_scores["指令跟随"] > 80

    examples = generate_before_after_examples()
    assert len(examples) == 5
    for ex in examples:
        assert "prompt" in ex
        assert "pretrain" in ex
        assert "sft" in ex
        assert "rlhf" in ex
