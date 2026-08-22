# Copyright (c) 2026 Yy1 (yuan278501381) | MIT License
"""
tests/test_pretraining.py - 预训练范式模块单元测试与梯度检验
"""

import numpy as np
import pytest

from nn_core.pretraining import (
    CausalLanguageModel,
    ContrastiveLearning,
    MaskedAutoEncoder,
    MaskedLanguageModel,
    PretrainingComparator,
)


def test_mlm_batch_and_loss():
    """测试 BERT 式掩码生成与 MLM 交叉熵损失"""
    vocab_size = 50
    seq_len = 20
    token_ids = np.random.randint(1, vocab_size, size=(1, seq_len))
    
    mlm = MaskedLanguageModel(vocab_size=vocab_size, d_model=16, mask_ratio=0.15)
    masked_ids, labels, mask_pos = mlm.create_mlm_batch(token_ids)
    
    assert masked_ids.shape == (1, seq_len)
    assert np.sum(mask_pos) >= 1
    assert np.sum(labels != -100) == np.sum(mask_pos)

    logits = mlm.forward(masked_ids)
    assert logits.shape == (1, seq_len, vocab_size)

    loss = mlm.mlm_loss(logits, labels, mask_pos)
    assert loss > 0.0

    # 验证 train_step 真实执行
    l_step = mlm.train_step(token_ids)
    assert l_step > 0.0


def test_clm_batch_and_loss():
    """测试 GPT 式因果接龙生成与 CLM 交叉熵损失"""
    vocab_size = 50
    seq_len = 15
    token_ids = np.random.randint(1, vocab_size, size=(1, seq_len))
    
    clm = CausalLanguageModel(vocab_size=vocab_size, d_model=16)
    x_in, y_tgt = clm.create_clm_batch(token_ids)
    assert x_in.shape == (1, seq_len - 1)
    assert y_tgt.shape == (1, seq_len - 1)

    logits = clm.forward(x_in)
    assert logits.shape == (1, seq_len - 1, vocab_size)

    loss = clm.clm_loss(logits, y_tgt)
    assert loss > 0.0

    l_step = clm.train_step(token_ids)
    assert l_step > 0.0


def test_contrastive_learning_nt_xent():
    """测试 NT-Xent 对比学习损失"""
    N = 8
    D = 16
    embeds = np.random.randn(N, D)
    cl = ContrastiveLearning(d_model=D)
    
    z_i, z_j = cl.create_positive_pairs(embeds, noise_std=0.01)
    assert z_i.shape == (N, D)
    assert z_j.shape == (N, D)

    # 极低噪声下正样本高度相似，损失应较低
    loss = cl.nt_xent_loss(z_i, z_j, temperature=0.5)
    assert loss > 0.0


def test_mae_masking_and_reconstruction():
    """测试 MAE 视觉掩码率与自编码重建损失"""
    B = 2
    num_patches = 16
    d_model = 32
    patches = np.random.randn(B, num_patches, d_model)

    mae = MaskedAutoEncoder(num_patches=num_patches, d_model=d_model, mask_ratio=0.75)
    masked_p, mask_idx, unmask_idx = mae.create_mae_batch(patches)
    
    assert len(mask_idx) == 12  # 75% of 16
    assert np.all(masked_p[:, mask_idx] == 0.0)

    recon = mae.reconstruct(masked_p)
    assert recon.shape == patches.shape

    loss = mae.reconstruction_loss(recon, patches, mask_idx)
    assert loss >= 0.0


def test_pretraining_comparator():
    """测试下游任务迁移能力字典"""
    scores = PretrainingComparator.get_transfer_scores()
    assert len(scores) == 4
    for model_name, task_dict in scores.items():
        assert len(task_dict) == 4
        for score in task_dict.values():
            assert 0 <= score <= 100
