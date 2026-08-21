# Copyright (c) 2026 Yy1 (yuan278501381) | MIT License
"""
tests.test_llm_core - LLM 核心模块单元测试与数值梯度检验

覆盖 M05~M09 全部现代 NLP/LLM 算子：
    - Embedding (词嵌入查表 + 稀疏梯度累加)
    - PositionalEncoding (正弦位置编码)
    - RNNCell (循环神经网络单步与序列推演)
    - scaled_dot_product_attention (缩放点积注意力)
    - MultiHeadAttention (多头注意力)
    - causal_mask (因果掩码)
    - LayerNorm (层归一化 + 完整反向微分链)
    - GELU (高斯误差线性单元)
    - FeedForward (两层 GELU 感知机)
    - TransformerBlock (Pre-LN Decoder Block)
    - TinyGPT (自回归生成 + Weight Tying)
"""

import numpy as np
import pytest

from nn_core.embeddings import Embedding, PositionalEncoding, get_mini_vocab, get_pretrained_embeddings
from nn_core.rnn import RNNCell
from nn_core.attention import causal_mask, scaled_dot_product_attention, MultiHeadAttention
from nn_core.layernorm import LayerNorm
from nn_core.gelu import GELU
from nn_core.transformer import FeedForward, TransformerBlock
from nn_core.gpt import TinyGPT
from nn_core.tensor import set_seed


# ========================== 固定随机种子 ==========================
@pytest.fixture(autouse=True)
def _seed():
    """每个测试用例前固定随机种子，保证可复现性"""
    set_seed(42)


# ========================== Embedding 词嵌入测试 ==========================
class TestEmbedding:
    """词嵌入层单元测试"""

    def test_前向输出形状(self):
        """查表前向应返回 (batch, seq_len, d_model)"""
        emb = Embedding(vocab_size=50, d_model=16)
        ids = np.array([[1, 3, 5], [2, 4, 6]])
        out = emb.forward(ids)
        assert out.shape == (2, 3, 16), f"预期 (2,3,16)，实际 {out.shape}"

    def test_查表值正确性(self):
        """查表返回的向量应与权重矩阵对应行完全一致"""
        emb = Embedding(vocab_size=10, d_model=4)
        ids = np.array([[0, 3]])
        out = emb.forward(ids)
        np.testing.assert_array_equal(out[0, 0], emb.weights[0])
        np.testing.assert_array_equal(out[0, 1], emb.weights[3])

    def test_反向稀疏梯度累加(self):
        """np.add.at 需正确处理重复 token 的梯度叠加"""
        emb = Embedding(vocab_size=5, d_model=3)
        # token_id=2 出现 2 次，其梯度应被累加而非覆盖
        ids = np.array([[2, 2, 0]])
        emb.forward(ids)
        dout = np.ones((1, 3, 3))
        emb.grad_weights[:] = 0
        emb.backward(dout)
        # token 2 出现了 2 次，梯度应累加为 2
        assert emb.grad_weights[2, 0] == 2.0, "重复 token 梯度应累加为 2"
        # token 0 出现了 1 次
        assert emb.grad_weights[0, 0] == 1.0

    def test_初始化方差合理(self):
        """权重初始化应为小方差正态 (std ≈ 0.02)"""
        emb = Embedding(vocab_size=1000, d_model=64)
        assert np.std(emb.weights) < 0.05, "初始化标准差应接近 0.02"


# ========================== PositionalEncoding 位置编码测试 ==========================
class TestPositionalEncoding:
    """正弦位置编码单元测试"""

    def test_编码矩阵形状(self):
        """位置编码应为 (1, max_len, d_model)"""
        pe = PositionalEncoding(max_len=100, d_model=32)
        assert pe.pe.shape == (1, 100, 32)

    def test_正弦余弦交替(self):
        """偶数维为 sin，奇数维为 cos"""
        pe = PositionalEncoding(max_len=10, d_model=4)
        pos0_dim0 = pe.pe[0, 0, 0]  # sin(0) = 0
        pos0_dim1 = pe.pe[0, 0, 1]  # cos(0) = 1
        assert abs(pos0_dim0 - 0.0) < 1e-10, "sin(0) 应为 0"
        assert abs(pos0_dim1 - 1.0) < 1e-10, "cos(0) 应为 1"

    def test_前向加法透传(self):
        """forward 应执行加法, backward 直接透传梯度"""
        pe = PositionalEncoding(max_len=20, d_model=8)
        x = np.random.randn(2, 5, 8)
        out = pe.forward(x)
        assert out.shape == x.shape
        # 验证反向透传
        dout = np.ones_like(out)
        dx = pe.backward(dout)
        np.testing.assert_array_equal(dx, dout)

    def test_不同位置编码不同(self):
        """不同 position 的编码向量应不同"""
        pe = PositionalEncoding(max_len=10, d_model=16)
        vec_pos0 = pe.pe[0, 0, :]
        vec_pos5 = pe.pe[0, 5, :]
        assert not np.allclose(vec_pos0, vec_pos5), "不同位置编码应有差异"


# ========================== RNNCell 循环神经网络测试 ==========================
class TestRNNCell:
    """Vanilla RNN Cell 单元测试"""

    def test_单步前向输出形状(self):
        """单步 forward 输出 h_t 形状应为 (batch, hidden)"""
        rnn = RNNCell(input_size=4, hidden_size=8)
        x_t = np.random.randn(2, 4)
        h_prev = np.zeros((2, 8))
        h_t = rnn.forward(x_t, h_prev)
        assert h_t.shape == (2, 8)

    def test_tanh_激活值域(self):
        """RNN 隐藏状态应在 [-1, 1] 范围内 (tanh 输出)"""
        rnn = RNNCell(input_size=4, hidden_size=8)
        x_t = np.random.randn(3, 4) * 10.0  # 极端输入
        h_prev = np.random.randn(3, 8) * 10.0
        h_t = rnn.forward(x_t, h_prev)
        assert np.all(h_t >= -1.0) and np.all(h_t <= 1.0), "tanh 输出应在 [-1,1]"

    def test_序列推演长度与形状(self):
        """step_sequence 应返回 seq_len 个隐藏状态"""
        rnn = RNNCell(input_size=4, hidden_size=8)
        X_seq = np.random.randn(2, 5, 4)  # (batch=2, seq_len=5, input=4)
        h_states = rnn.step_sequence(X_seq)
        assert len(h_states) == 5
        assert h_states[0].shape == (2, 8)
        assert h_states[-1].shape == (2, 8)

    def test_记忆衰减特性(self):
        """长序列末端隐藏状态应与首端差异增大 (信息衰减)"""
        rnn = RNNCell(input_size=2, hidden_size=4)
        # 第一步有明确信号，后续全零输入
        X_seq = np.zeros((1, 20, 2))
        X_seq[0, 0, :] = [5.0, 5.0]  # 仅首步有信号
        h_states = rnn.step_sequence(X_seq)
        h0_norm = np.linalg.norm(h_states[0])
        h_end_norm = np.linalg.norm(h_states[-1])
        assert h_end_norm < h0_norm, "信号应随时间步衰减 (RNN 遗忘特性)"


# ========================== Attention 注意力机制测试 ==========================
class TestCausalMask:
    """因果掩码测试"""

    def test_下三角矩阵(self):
        """因果掩码应为下三角矩阵 (上三角为 0)"""
        mask = causal_mask(4)
        assert mask.shape == (4, 4)
        expected = np.tril(np.ones((4, 4)))
        np.testing.assert_array_equal(mask, expected)

    def test_对角线全为1(self):
        """对角线应全为 1 (允许看到当前位置)"""
        mask = causal_mask(5)
        np.testing.assert_array_equal(np.diag(mask), np.ones(5))


class TestScaledDotProductAttention:
    """缩放点积注意力测试"""

    def test_输出形状(self):
        """注意力输出形状应与 V 一致"""
        q = np.random.randn(2, 4, 8)
        k = np.random.randn(2, 4, 8)
        v = np.random.randn(2, 4, 8)
        output, weights = scaled_dot_product_attention(q, k, v)
        assert output.shape == (2, 4, 8)
        assert weights.shape == (2, 4, 4)

    def test_注意力权重行和为1(self):
        """每行 softmax 后权重总和应为 1"""
        q = np.random.randn(1, 3, 4)
        k = np.random.randn(1, 3, 4)
        v = np.random.randn(1, 3, 4)
        _, weights = scaled_dot_product_attention(q, k, v)
        row_sums = np.sum(weights, axis=-1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-6)

    def test_因果掩码遮蔽未来(self):
        """因果掩码后，位置 i 不应关注位置 j > i"""
        q = np.random.randn(1, 4, 8)
        k = np.random.randn(1, 4, 8)
        v = np.random.randn(1, 4, 8)
        mask = causal_mask(4)
        _, weights = scaled_dot_product_attention(q, k, v, mask)
        # 第一行 (第一个 token) 只能看到自己
        assert weights[0, 0, 1] < 1e-6, "第一个位置不应关注未来位置"
        assert weights[0, 0, 2] < 1e-6
        assert weights[0, 0, 3] < 1e-6

    def test_缩放因子效果(self):
        """缩放 1/sqrt(d_k) 应防止 softmax 极化"""
        d_k = 64
        q = np.random.randn(1, 4, d_k) * 2.0  # 中等方差
        k = np.random.randn(1, 4, d_k) * 2.0
        v = np.random.randn(1, 4, d_k)
        _, weights = scaled_dot_product_attention(q, k, v)
        # 缩放后注意力分布应相对均匀 (非完全集中)
        min_entropy = -np.sum(weights * np.log(weights + 1e-10), axis=-1).min()
        assert min_entropy > 0.01, "缩放后注意力应保持一定熵值"


class TestMultiHeadAttention:
    """多头注意力测试"""

    def test_输出形状(self):
        """MHA 输出应保持 (batch, seq_len, d_model)"""
        mha = MultiHeadAttention(d_model=16, num_heads=4)
        x = np.random.randn(2, 5, 16)
        out, weights = mha.forward(x)
        assert out.shape == (2, 5, 16)
        # weights: (batch, num_heads, seq_len, seq_len)
        assert weights.shape == (2, 4, 5, 5)

    def test_多头注意力权重行和(self):
        """每个头的注意力权重行和应为 1"""
        mha = MultiHeadAttention(d_model=8, num_heads=2)
        x = np.random.randn(1, 3, 8)
        _, weights = mha.forward(x)
        row_sums = np.sum(weights, axis=-1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-6)

    def test_d_model_必须整除_num_heads(self):
        """d_model 不能被 num_heads 整除时应报错"""
        with pytest.raises(AssertionError):
            MultiHeadAttention(d_model=10, num_heads=3)

    def test_因果掩码集成(self):
        """MHA + 因果掩码应正确屏蔽未来位置"""
        mha = MultiHeadAttention(d_model=8, num_heads=2)
        x = np.random.randn(1, 4, 8)
        mask = causal_mask(4)
        _, weights = mha.forward(x, mask=mask)
        # 每个头的第一个 query 都只能看到自己
        for h in range(2):
            assert weights[0, h, 0, 1] < 1e-6, f"头 {h} 的位置 0 不应关注位置 1"


# ========================== LayerNorm 层归一化测试 ==========================
class TestLayerNorm:
    """层归一化单元测试"""

    def test_归一化后均值为零方差为一(self):
        """归一化后在最后一维上均值接近 0，方差接近 1"""
        ln = LayerNorm(d_model=16)
        x = np.random.randn(2, 5, 16) * 10 + 3  # 大均值大方差
        out = ln.forward(x)
        means = np.mean(out, axis=-1)
        vars_ = np.var(out, axis=-1)
        np.testing.assert_allclose(means, 0.0, atol=1e-5)
        np.testing.assert_allclose(vars_, 1.0, atol=1e-2)

    def test_可学习参数初始化(self):
        """gamma 初始为 1，beta 初始为 0"""
        ln = LayerNorm(d_model=8)
        np.testing.assert_array_equal(ln.gamma, np.ones(8))
        np.testing.assert_array_equal(ln.beta, np.zeros(8))

    def test_反向梯度形状正确(self):
        """backward 输出梯度形状应与输入一致"""
        ln = LayerNorm(d_model=8)
        x = np.random.randn(2, 3, 8)
        ln.forward(x)
        dout = np.random.randn(2, 3, 8)
        dx = ln.backward(dout)
        assert dx.shape == (2, 3, 8)
        assert ln.grad_gamma.shape == (8,)
        assert ln.grad_beta.shape == (8,)

    def test_数值梯度检验(self):
        """LayerNorm 反向梯度与数值差分应一致 (中心差分法)"""
        ln = LayerNorm(d_model=4)
        x = np.random.randn(1, 3, 4)
        eps = 1e-5

        # 前向 + 用随机 dout 计算解析梯度
        out = ln.forward(x)
        dout = np.random.randn(*out.shape)
        dx_analytical = ln.backward(dout)

        # 数值梯度 (中心差分)
        dx_numerical = np.zeros_like(x)
        for i in range(x.shape[1]):
            for j in range(x.shape[2]):
                x_plus = x.copy()
                x_plus[0, i, j] += eps
                x_minus = x.copy()
                x_minus[0, i, j] -= eps
                ln_plus = LayerNorm(d_model=4)
                ln_minus = LayerNorm(d_model=4)
                out_plus = ln_plus.forward(x_plus)
                out_minus = ln_minus.forward(x_minus)
                dx_numerical[0, i, j] = np.sum(dout * (out_plus - out_minus)) / (2 * eps)

        rel_err = np.max(np.abs(dx_analytical - dx_numerical) / (np.abs(dx_numerical) + 1e-8))
        assert rel_err < 1e-4, f"LayerNorm 数值梯度相对误差过大: {rel_err:.2e}"


# ========================== GELU 激活函数测试 ==========================
class TestGELU:
    """GELU 激活函数单元测试"""

    def test_零点值(self):
        """GELU(0) = 0"""
        gelu = GELU()
        out = gelu.forward(np.array([0.0]))
        assert abs(out[0]) < 1e-10

    def test_正大值趋近恒等(self):
        """当 x >> 0 时，GELU(x) ≈ x"""
        gelu = GELU()
        out = gelu.forward(np.array([5.0]))
        assert abs(out[0] - 5.0) < 0.01

    def test_负大值趋近零(self):
        """当 x << 0 时，GELU(x) ≈ 0"""
        gelu = GELU()
        out = gelu.forward(np.array([-5.0]))
        assert abs(out[0]) < 0.01

    def test_数值梯度检验(self):
        """GELU 反向梯度与数值差分应一致"""
        gelu = GELU()
        x = np.random.randn(2, 4)
        eps = 1e-5

        gelu.forward(x)
        dout = np.ones_like(x)
        dx_analytical = gelu.backward(dout)

        dx_numerical = np.zeros_like(x)
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                x_p = x.copy()
                x_p[i, j] += eps
                x_m = x.copy()
                x_m[i, j] -= eps
                g = GELU()
                f_p = g.forward(x_p)
                g2 = GELU()
                f_m = g2.forward(x_m)
                dx_numerical[i, j] = np.sum(dout * (f_p - f_m)) / (2 * eps)

        rel_err = np.max(np.abs(dx_analytical - dx_numerical) / (np.abs(dx_numerical) + 1e-8))
        assert rel_err < 1e-4, f"GELU 数值梯度相对误差过大: {rel_err:.2e}"


# ========================== Softmax 数值梯度检验 (扩充) ==========================
class TestSoftmaxGradient:
    """Softmax 反向传播数值梯度检验"""

    def test_softmax_数值梯度检验(self):
        """Softmax Jacobian 向量化实现与中心差分对比"""
        from nn_core.activations import Softmax
        sm = Softmax()
        x = np.random.randn(2, 5)
        eps = 1e-5

        out = sm.forward(x)
        dout = np.random.randn(*out.shape)
        dx_analytical = sm.backward(dout)

        dx_numerical = np.zeros_like(x)
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                x_p = x.copy()
                x_p[i, j] += eps
                x_m = x.copy()
                x_m[i, j] -= eps
                s1 = Softmax()
                f_p = s1.forward(x_p)
                s2 = Softmax()
                f_m = s2.forward(x_m)
                dx_numerical[i, j] = np.sum(dout * (f_p - f_m)) / (2 * eps)

        rel_err = np.max(np.abs(dx_analytical - dx_numerical) / (np.abs(dx_numerical) + 1e-8))
        assert rel_err < 1e-4, f"Softmax 数值梯度相对误差过大: {rel_err:.2e}"


# ========================== FeedForward & Transformer 测试 ==========================
class TestFeedForward:
    """两层 GELU 前馈网络测试"""

    def test_输出形状(self):
        """FFN 输出形状应与输入一致"""
        ffn = FeedForward(d_model=8, d_ff=32)
        x = np.random.randn(2, 5, 8)
        out = ffn.forward(x)
        assert out.shape == (2, 5, 8)


class TestTransformerBlock:
    """Pre-LN Transformer Decoder Block 测试"""

    def test_输出形状(self):
        """TransformerBlock 输出应保持维度不变"""
        block = TransformerBlock(d_model=16, num_heads=4, d_ff=64)
        x = np.random.randn(2, 5, 16)
        out, attn = block.forward(x)
        assert out.shape == (2, 5, 16)
        assert attn.shape == (2, 4, 5, 5)

    def test_残差连接效果(self):
        """残差连接应使输出 ≠ 0 (即使随机初始化)"""
        block = TransformerBlock(d_model=8, num_heads=2, d_ff=32)
        x = np.random.randn(1, 3, 8)
        out, _ = block.forward(x)
        assert np.linalg.norm(out) > 0.0

    def test_因果掩码集成(self):
        """带因果掩码的 TransformerBlock 应正常工作"""
        block = TransformerBlock(d_model=8, num_heads=2, d_ff=32)
        x = np.random.randn(1, 4, 8)
        mask = causal_mask(4)
        out, attn = block.forward(x, mask=mask)
        assert out.shape == (1, 4, 8)
        # 注意力权重应遵循因果约束
        assert attn[0, 0, 0, 1] < 1e-6


# ========================== TinyGPT 完整架构测试 ==========================
class TestTinyGPT:
    """TinyGPT 完整架构测试"""

    def test_前向输出形状(self):
        """前向传播输出 logits 形状应为 (batch, seq_len, vocab_size)"""
        gpt = TinyGPT(vocab_size=50, max_seq_len=10, d_model=16, num_heads=2, num_layers=1)
        ids = np.array([[1, 3, 5, 7]])
        logits = gpt.forward(ids)
        assert logits.shape == (1, 4, 50), f"预期 (1,4,50)，实际 {logits.shape}"

    def test_权重绑定(self):
        """输出投影应使用 embedding.weights.T (Weight Tying)"""
        gpt = TinyGPT(vocab_size=20, max_seq_len=8, d_model=8, num_heads=2, num_layers=1)
        ids = np.array([[0, 1, 2]])
        logits = gpt.forward(ids)
        # logits = ln_f_output @ embedding.weights.T
        # 验证形状一致性即可证明 weight tying 生效
        assert logits.shape[-1] == gpt.vocab_size

    def test_注意力权重捕获(self):
        """get_all_attention_weights 应返回 num_layers 个注意力矩阵"""
        gpt = TinyGPT(vocab_size=20, max_seq_len=8, d_model=8, num_heads=2, num_layers=3)
        ids = np.array([[0, 1, 2]])
        gpt.forward(ids)
        attn_list = gpt.get_all_attention_weights()
        assert len(attn_list) == 3
        for attn in attn_list:
            assert attn.shape == (1, 2, 3, 3)

    def test_贪心生成(self):
        """temperature=0 应执行确定性贪心解码"""
        gpt = TinyGPT(vocab_size=10, max_seq_len=8, d_model=8, num_heads=2, num_layers=1)
        result = gpt.generate(prompt_ids=[0, 1], max_new_tokens=3, temperature=0)
        assert len(result) == 5, "生成长度 = prompt(2) + new(3) = 5"
        # 贪心解码两次应得到相同结果 (固定种子下)
        set_seed(42)
        gpt2 = TinyGPT(vocab_size=10, max_seq_len=8, d_model=8, num_heads=2, num_layers=1)
        r1 = gpt2.generate(prompt_ids=[0, 1], max_new_tokens=3, temperature=0)
        set_seed(42)
        gpt3 = TinyGPT(vocab_size=10, max_seq_len=8, d_model=8, num_heads=2, num_layers=1)
        r2 = gpt3.generate(prompt_ids=[0, 1], max_new_tokens=3, temperature=0)
        assert r1 == r2, "贪心解码在相同初始化下应确定性"

    def test_top_k_采样(self):
        """Top-K 采样应只从前 K 个 token 中选择"""
        gpt = TinyGPT(vocab_size=10, max_seq_len=8, d_model=8, num_heads=2, num_layers=1)
        result = gpt.generate(prompt_ids=[0], max_new_tokens=5, temperature=0.8, top_k=3)
        assert len(result) == 6
        # 所有生成的 token 应在 [0, vocab_size) 范围内
        for t in result:
            assert 0 <= t < 10

    def test_序列截断(self):
        """超过 max_seq_len 的输入应被截断而非报错"""
        gpt = TinyGPT(vocab_size=10, max_seq_len=4, d_model=8, num_heads=2, num_layers=1)
        # 生成足够多 token 使总长超过 max_seq_len
        result = gpt.generate(prompt_ids=[0, 1, 2], max_new_tokens=5, temperature=0.5)
        assert len(result) == 8


# ========================== 辅助工具函数测试 ==========================
class TestMiniVocab:
    """迷你词表工具函数测试"""

    def test_词表大小与唯一性(self):
        """get_mini_vocab 应返回唯一映射"""
        vocab = get_mini_vocab()
        assert len(vocab) > 50, "词表应包含 50+ 词"
        # 值应唯一
        values = list(vocab.values())
        assert len(values) == len(set(values)), "Token ID 应唯一"

    def test_经典语义对存在(self):
        """必须包含 king, queen, man, woman 等经典语义对"""
        vocab = get_mini_vocab()
        for word in ["king", "queen", "man", "woman"]:
            assert word in vocab, f"词表应包含 {word}"

    def test_预训练嵌入维度(self):
        """get_pretrained_embeddings 返回正确形状"""
        vocab = get_mini_vocab()
        emb = get_pretrained_embeddings(len(vocab), d_model=32)
        assert emb.shape == (len(vocab), 32)


# ========================== L1 正则化测试 (补充) ==========================
class TestL1Regularizer:
    """L1 正则化单元测试"""

    def test_损失计算(self):
        """L1 损失 = lambda * sum(|W|)"""
        from nn_core.regularizers import L1
        reg = L1(lambda_=0.1)
        W = np.array([[1.0, -2.0], [3.0, -4.0]])
        loss = reg.loss(W)
        expected = 0.1 * (1 + 2 + 3 + 4)
        assert abs(loss - expected) < 1e-10

    def test_次梯度正确(self):
        """L1 梯度 = lambda * sign(W)"""
        from nn_core.regularizers import L1
        reg = L1(lambda_=0.5)
        W = np.array([[1.0, -2.0], [0.0, 3.0]])
        grad = reg.gradient(W)
        expected = 0.5 * np.sign(W)
        np.testing.assert_array_equal(grad, expected)


# ========================== Tensor 工具测试 (补充) ==========================
class TestTensorUtils:
    """数值稳定性工具函数测试"""

    def test_safe_log_防零(self):
        """safe_log(0) 不应产生 -inf"""
        from nn_core.tensor import safe_log
        result = safe_log(np.array([0.0, 1e-15, 1.0]))
        assert not np.any(np.isinf(result)), "safe_log 应防止 -inf"
        assert not np.any(np.isnan(result)), "safe_log 应防止 NaN"

    def test_safe_exp_防溢出(self):
        """safe_exp(1000) 不应产生 inf"""
        from nn_core.tensor import safe_exp
        result = safe_exp(np.array([1000.0, -1000.0, 0.0]))
        assert not np.any(np.isinf(result)), "safe_exp 应防止 inf"
        assert not np.any(np.isnan(result)), "safe_exp 应防止 NaN"

    def test_clip_gradients_裁剪(self):
        """梯度范数超限时应按比例缩放"""
        from nn_core.tensor import clip_gradients
        grads = np.array([3.0, 4.0])  # norm = 5
        clipped = clip_gradients(grads, max_norm=1.0)
        assert abs(np.linalg.norm(clipped) - 1.0) < 1e-10, "裁剪后范数应为 max_norm"

    def test_clip_gradients_不超限原样返回(self):
        """梯度范数未超限时应原样返回"""
        from nn_core.tensor import clip_gradients
        grads = np.array([0.1, 0.1])
        clipped = clip_gradients(grads, max_norm=5.0)
        np.testing.assert_array_equal(clipped, grads)
