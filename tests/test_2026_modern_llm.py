# Copyright (c) 2026 Yy1 (yuan278501381) | MIT License
"""
tests.test_2026_modern_llm - 2026 现代 LLM 核心前沿算子单元测试套件

测试清单:
    - TestBPE: 字节级初始词表、频次合并、encode/decode 一致性、可视化 chunks
    - TestRoPE: 预计算频次、成对切片旋转、内积相对衰减、广播前向
    - TestGQA: MHA/GQA/MQA 模式切换、repeat_kv 维度变换、KV-Cache 压缩比计算
    - TestSwiGLU: SiLU 激活函数精度、SwiGLU 门控前向、参数协议提取
    - TestKVCache: 动态追加、序列截断防护、显存计量、重置
"""

import numpy as np
import pytest

from nn_core.bpe import BytePairEncoder
from nn_core.gqa import GroupedQueryAttention, repeat_kv
from nn_core.kv_cache import KVCache
from nn_core.rope import RotaryPositionalEmbedding, apply_rope, precompute_freqs_cis
from nn_core.swiglu import SwiGLU, silu
from nn_core.tensor import set_seed


@pytest.fixture(autouse=True)
def _seed():
    set_seed(42)


# ==========================================
# 1. BPE 分词工程测试
# ==========================================
class TestBPE:
    """BytePairEncoder 单元测试"""

    def test_初始单字节词表大小(self):
        """初始词表应精确包含 256 个单字节"""
        bpe = BytePairEncoder(vocab_size=260)
        assert len(bpe.vocab) == 256

    def test_训练与合并规则生成(self):
        """训练后 merges 数量应为 vocab_size - 256"""
        bpe = BytePairEncoder(vocab_size=260)
        corpus = "the king and the queen ruled the kingdom and the queen was happy"
        bpe.train(corpus)
        assert len(bpe.merges) <= 4
        assert len(bpe.merge_history) <= 4

    def test_encode_decode_自洽性(self):
        """对任意文本 encode 后 decode 必须精确复原原文本"""
        bpe = BytePairEncoder(vocab_size=270)
        corpus = "machine learning and artificial intelligence in python"
        bpe.train(corpus)

        test_cases = [
            "machine learning",
            "artificial intelligence",
            "hello world! 123",
            "王族与大语言模型 (UTF-8 中文兼容)",
        ]
        for text in test_cases:
            ids = bpe.encode(text)
            decoded = bpe.decode(ids)
            assert decoded == text, f"预期 '{text}'，解码得到 '{decoded}'"

    def test_可视化切片块提取(self):
        """tokenize_visual_chunks 应返回文本切片与 Token ID 列表"""
        bpe = BytePairEncoder(vocab_size=265)
        corpus = "banana band bandana"
        bpe.train(corpus)
        chunks = bpe.tokenize_visual_chunks("banana")
        assert len(chunks) > 0
        reconstructed = "".join([c[0] for c in chunks])
        assert reconstructed == "banana"


# ==========================================
# 2. RoPE 旋转位置编码测试
# ==========================================
class TestRoPE:
    """RotaryPositionalEmbedding 单元测试"""

    def test_预计算频率矩阵维度(self):
        """cos 和 sin 应为 (max_seq_len, dim)"""
        cos, sin = precompute_freqs_cis(dim=16, max_seq_len=64)
        assert cos.shape == (64, 16)
        assert sin.shape == (64, 16)

    def test_位置零不旋转(self):
        """在 position=0 处，cos=1, sin=0，向量保持不变"""
        cos, sin = precompute_freqs_cis(dim=8, max_seq_len=10)
        x = np.random.randn(1, 1, 8)
        x_rot = apply_rope(x, cos, sin)
        # pos=0 处应与原输入相等
        np.testing.assert_allclose(x_rot[0, 0], x[0, 0], atol=1e-7)

    def test_成对旋转正交性模长守恒(self):
        """旋转变换为正交变换，应用 RoPE 后向量 L2 模长必须守恒"""
        rope = RotaryPositionalEmbedding(dim=16, max_seq_len=32)
        q = np.random.randn(2, 5, 4, 16)
        k = np.random.randn(2, 5, 4, 16)
        q_rot, _k_rot = rope.forward(q, k)

        norm_q_orig = np.linalg.norm(q, axis=-1)
        norm_q_rot = np.linalg.norm(q_rot, axis=-1)
        np.testing.assert_allclose(norm_q_orig, norm_q_rot, atol=1e-6)

    def test_显式支持两种四维布局(self):
        rope = RotaryPositionalEmbedding(dim=8, max_seq_len=16)
        rng = np.random.default_rng(11)
        bshd = rng.normal(size=(2, 5, 3, 8))
        bhsd = bshd.transpose(0, 2, 1, 3)
        out_bshd, _ = rope.forward(bshd, bshd, seq_axis=1)
        out_bhsd, _ = rope.forward(bhsd, bhsd, seq_axis=2)
        np.testing.assert_allclose(out_bshd.transpose(0, 2, 1, 3), out_bhsd)

    def test_布局与长度契约错误被拒绝(self):
        cos, sin = precompute_freqs_cis(dim=8, max_seq_len=4)
        with pytest.raises(ValueError, match="超过"):
            apply_rope(np.ones((1, 5, 8)), cos, sin, seq_axis=1)
        with pytest.raises(ValueError, match="seq_axis"):
            apply_rope(np.ones((1, 4, 8)), cos, sin, seq_axis=2)

    def test_相对位置衰减矩阵计算(self):
        """相对衰减矩阵主对角线应为 1.0 (自身内积最大)"""
        rope = RotaryPositionalEmbedding(dim=8, max_seq_len=10)
        decay = rope.compute_relative_decay_matrix(seq_len=6)
        assert decay.shape == (6, 6)
        np.testing.assert_allclose(np.diag(decay), 1.0, atol=1e-6)


# ==========================================
# 3. GQA 分组查询注意力测试
# ==========================================
class TestGQA:
    """GroupedQueryAttention 单元测试"""

    def test_repeat_kv_广播倍增(self):
        """repeat_kv 应将头维度扩展为 n_rep 倍"""
        # (B=2, kv_heads=2, S=4, D=8)
        kv = np.random.randn(2, 2, 4, 8)
        kv_rep = repeat_kv(kv, n_rep=3)
        assert kv_rep.shape == (2, 6, 4, 8)
        # 验证重复块的一致性
        np.testing.assert_array_equal(kv_rep[:, 0, :, :], kv[:, 0, :, :])
        np.testing.assert_array_equal(kv_rep[:, 1, :, :], kv[:, 0, :, :])
        np.testing.assert_array_equal(kv_rep[:, 2, :, :], kv[:, 0, :, :])

    def test_gqa_前向输出维度(self):
        """GQA 输出形状应为 (batch, seq_len, d_model)"""
        gqa = GroupedQueryAttention(d_model=32, num_heads=8, num_kv_heads=2)
        x = np.random.randn(2, 6, 32)
        out, attn_weights = gqa.forward(x)
        assert out.shape == (2, 6, 32)
        assert attn_weights.shape == (2, 8, 6, 6)

    def test_kv_cache_显存节省计算(self):
        """num_heads=8, num_kv_heads=2 时，压缩比应为 4x，节省 75% 显存"""
        gqa = GroupedQueryAttention(d_model=64, num_heads=8, num_kv_heads=2)
        stats = gqa.get_kv_cache_savings()
        assert stats["compression_ratio"] == 4.0
        assert abs(stats["memory_saved_percent"] - 75.0) < 1e-5


# ==========================================
# 4. SwiGLU 门控 FFN 测试
# ==========================================
class TestSwiGLU:
    """SwiGLU 门控前馈网络测试"""

    def test_silu_激活函数(self):
        """SiLU(0) = 0, SiLU(x) > 0 for x > 0"""
        x = np.array([0.0, 2.0, -2.0])
        y = silu(x)
        assert abs(y[0] - 0.0) < 1e-10
        assert y[1] > 0.0
        assert y[2] < 0.0

    def test_swiglu_输出形状(self):
        """SwiGLU 输出形状应与输入一致 (batch, seq_len, d_model)"""
        swiglu = SwiGLU(d_model=16, d_ff=32)
        x = np.random.randn(2, 5, 16)
        out = swiglu.forward(x)
        assert out.shape == (2, 5, 16)

    def test_通用参数协议提取(self):
        """get_params_and_grads 应返回 6 组权重与偏置元组"""
        swiglu = SwiGLU(d_model=8, d_ff=16)
        params = swiglu.get_params_and_grads()
        assert len(params) == 6
        for p, g in params:
            assert p.shape == g.shape


# ==========================================
# 5. KVCache 容器测试
# ==========================================
class TestKVCache:
    """KVCache 推理加速容器测试"""

    def test_动态追加与序列推进(self):
        """update 应逐步积累序列并保持历史 KV"""
        cache = KVCache(num_layers=2, max_batch_size=1, max_seq_len=10, num_kv_heads=2, head_dim=4)

        # 步 1: Prompt 长度为 3
        k1 = np.random.randn(1, 2, 3, 4)
        v1 = np.random.randn(1, 2, 3, 4)
        full_k, _full_v = cache.update(0, k1, v1)
        _full_k2, _full_v2 = cache.update(1, k1, v1)

        assert full_k.shape == (1, 2, 3, 4)
        assert cache.current_seq_len == 3

        # 步 2: 新生成 1 个 Token
        k2 = np.random.randn(1, 2, 1, 4)
        v2 = np.random.randn(1, 2, 1, 4)
        full_k_step2, _ = cache.update(0, k2, v2)
        cache.update(1, k2, v2)

        assert full_k_step2.shape == (1, 2, 4, 4)
        assert cache.current_seq_len == 4

    def test_显存计量与重置(self):
        """get_memory_stats 应返回正数显存，reset 后清零"""
        cache = KVCache(num_layers=1, max_batch_size=1, max_seq_len=20, num_kv_heads=2, head_dim=8)
        cache.update(0, np.random.randn(1, 2, 5, 8), np.random.randn(1, 2, 5, 8))
        stats = cache.get_memory_stats()
        assert stats["current_tokens"] == 5.0
        assert stats["used_kb"] > 0

        cache.reset()
        assert cache.current_seq_len == 0
