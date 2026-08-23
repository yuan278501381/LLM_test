# Copyright (c) 2026 Yy1 (yuan278501381) | MIT License
"""
tests/test_flash_attention.py - FlashAttention-2 核心算法与数学等价性回归测试集
"""

import numpy as np
import pytest

from nn_core.attention import causal_mask, scaled_dot_product_attention
from nn_core.flash_attention import flash_attention_2_forward


def test_flash_attention_mathematical_equivalence_non_causal():
    """验证 FlashAttention-2 与标准注意力在非因果模式下的数学严格等价性 (容差 1e-5)"""
    np.random.seed(42)
    seq_len = 64
    d_k = 32
    d_v = 32

    q = np.random.randn(2, seq_len, d_k).astype(np.float64)
    k = np.random.randn(2, seq_len, d_k).astype(np.float64)
    v = np.random.randn(2, seq_len, d_v).astype(np.float64)

    std_out, _ = scaled_dot_product_attention(q, k, v)
    flash_out, tele = flash_attention_2_forward(
        q, k, v, block_size_r=16, block_size_c=16, is_causal=False
    )

    np.testing.assert_allclose(flash_out, std_out, rtol=1e-5, atol=1e-6)
    assert tele["total_flops"] > 0
    assert tele["flash_hbm_io_bytes"] > 0

    # 在实际硬件典型分块 Br=64 与长序列 N=1024 下，FlashAttention-2 显著减少 HBM 显存 IO 访存总量 (O(N) vs O(N^2))
    q_long = np.random.randn(1, 1024, d_k).astype(np.float64)
    k_long = np.random.randn(1, 1024, d_k).astype(np.float64)
    v_long = np.random.randn(1, 1024, d_v).astype(np.float64)
    _, tele_long = flash_attention_2_forward(
        q_long, k_long, v_long, block_size_r=64, block_size_c=64
    )
    assert tele_long["standard_hbm_io_bytes"] > tele_long["flash_hbm_io_bytes"]
    assert tele_long["io_reduction_ratio"] > 1.0


def test_flash_attention_mathematical_equivalence_causal():
    """验证 FlashAttention-2 与标准注意力在因果掩码模式下的数学严格等价性"""
    np.random.seed(123)
    seq_len = 48
    d_k = 16

    q = np.random.randn(1, seq_len, d_k).astype(np.float64)
    k = np.random.randn(1, seq_len, d_k).astype(np.float64)
    v = np.random.randn(1, seq_len, d_k).astype(np.float64)

    mask = causal_mask(seq_len)
    std_out, _ = scaled_dot_product_attention(q, k, v, mask=mask)
    flash_out, tele = flash_attention_2_forward(
        q, k, v, block_size_r=16, block_size_c=16, is_causal=True
    )

    np.testing.assert_allclose(flash_out, std_out, rtol=1e-5, atol=1e-6)
    assert len(tele["steps_trace"]) > 0


@pytest.mark.parametrize("br,bc", [(8, 8), (16, 8), (8, 16), (32, 32)])
def test_flash_attention_varying_block_sizes(br: int, bc: int):
    """验证不同分块大小 Br, Bc 计算结果均保持绝对一致"""
    np.random.seed(42)
    seq_len = 32
    d_k = 16

    q = np.random.randn(1, seq_len, d_k)
    k = np.random.randn(1, seq_len, d_k)
    v = np.random.randn(1, seq_len, d_k)

    std_out, _ = scaled_dot_product_attention(q, k, v)
    flash_out, _ = flash_attention_2_forward(q, k, v, block_size_r=br, block_size_c=bc)

    np.testing.assert_allclose(flash_out, std_out, rtol=1e-5, atol=1e-6)


def test_flash_attention_input_validation():
    """验证异常输入的健壮性防御断言"""
    q_invalid = np.zeros((10,))
    k_valid = np.zeros((10, 16))
    v_valid = np.zeros((10, 16))

    with pytest.raises(ValueError, match="至少需要序列维与特征维"):
        flash_attention_2_forward(q_invalid, k_valid, v_valid)

    q_dim_mismatch = np.zeros((10, 32))
    with pytest.raises(ValueError, match="特征维度不匹配"):
        flash_attention_2_forward(q_dim_mismatch, k_valid, v_valid)

    k_len_mismatch = np.zeros((12, 16))
    with pytest.raises(ValueError, match="序列长度不匹配"):
        flash_attention_2_forward(k_valid, k_len_mismatch, v_valid)
