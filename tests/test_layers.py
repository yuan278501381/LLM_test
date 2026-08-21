# Copyright (c) 2026 Yy1 (yuan278501381) | MIT License
"""网络层单元测试"""

import numpy as np
import pytest

from nn_core.layers import Dense, Dropout
from nn_core.regularizers import L2
from nn_core.tensor import set_seed


class TestDense:
    """Dense 全连接层测试"""

    def test_forward_输出形状正确(self):
        """Dense(3, 5) 输入 (10, 3) 应输出 (10, 5)"""
        layer = Dense(3, 5)
        x = np.random.randn(10, 3)
        out = layer.forward(x)
        assert out.shape == (10, 5)

    def test_forward_线性计算正确(self):
        """手动设置 W 和 b，验证 Z = XW + b"""
        layer = Dense(2, 2, initializer="zeros")
        layer.weights = np.array([[1.0, 0.0], [0.0, 1.0]])
        layer.biases = np.array([[0.5, -0.5]])
        x = np.array([[1.0, 2.0]])
        out = layer.forward(x)
        expected = np.array([[1.5, 1.5]])  # [1*1+2*0+0.5, 1*0+2*1-0.5]
        np.testing.assert_allclose(out, expected, atol=1e-10)

    def test_backward_输出形状正确(self):
        """backward 返回形状 == forward 输入形状"""
        layer = Dense(3, 5)
        x = np.random.randn(10, 3)
        layer.forward(x)
        dout = np.random.randn(10, 5)
        dx = layer.backward(dout)
        assert dx.shape == (10, 3)

    def test_backward_梯度累积正确(self):
        """验证 grad_weights = X.T @ dout"""
        set_seed(42)
        layer = Dense(2, 3, initializer="random")
        x = np.array([[1.0, 2.0], [3.0, 4.0]])
        layer.forward(x)
        dout = np.ones((2, 3))
        layer.backward(dout)
        expected_gw = x.T @ dout
        np.testing.assert_allclose(layer.grad_weights, expected_gw, atol=1e-10)

    def test_backward_偏置梯度正确(self):
        """验证 grad_biases = sum(dout, axis=0)"""
        layer = Dense(2, 3, initializer="random")
        x = np.random.randn(5, 2)
        layer.forward(x)
        dout = np.random.randn(5, 3)
        layer.backward(dout)
        expected_gb = np.sum(dout, axis=0, keepdims=True)
        np.testing.assert_allclose(layer.grad_biases, expected_gb, atol=1e-10)

    def test_不同初始化方式_zeros(self):
        """zeros 初始化: 权重全部为零"""
        layer = Dense(3, 3, initializer="zeros")
        np.testing.assert_array_equal(layer.weights, np.zeros((3, 3)))

    def test_不同初始化方式_xavier(self):
        """xavier 初始化: 方差近似 2/(n_in+n_out)"""
        set_seed(42)
        layer = Dense(100, 100, initializer="xavier")
        expected_var = 2.0 / (100 + 100)
        actual_var = np.var(layer.weights)
        np.testing.assert_allclose(actual_var, expected_var, rtol=0.3)

    def test_不同初始化方式_he(self):
        """he 初始化: 方差近似 2/n_in"""
        set_seed(42)
        layer = Dense(100, 100, initializer="he")
        expected_var = 2.0 / 100
        actual_var = np.var(layer.weights)
        np.testing.assert_allclose(actual_var, expected_var, rtol=0.3)

    def test_正则化梯度_L2(self):
        """带 L2 正则化时，grad_weights 包含正则化项"""
        reg = L2(lambda_=0.1)
        layer = Dense(2, 2, initializer="random", regularizer=reg)
        x = np.random.randn(3, 2)
        layer.forward(x)
        dout = np.random.randn(3, 2)
        layer.backward(dout)
        # grad_weights 应该 = X.T @ dout + lambda * weights
        expected = x.T @ dout + 0.1 * layer.weights
        np.testing.assert_allclose(layer.grad_weights, expected, atol=1e-10)

    def test_参数校验(self):
        """非法参数应抛出异常"""
        with pytest.raises(ValueError):
            Dense(0, 5)
        with pytest.raises(ValueError):
            Dense(5, -1)
        with pytest.raises(ValueError):
            Dense(5, 5, initializer="unknown")


class TestDropout:
    """Dropout 层测试"""

    def test_训练模式有神经元被丢弃(self):
        """训练模式下应有部分输出为零"""
        set_seed(42)
        d = Dropout(rate=0.5)
        x = np.ones((100, 100))
        out = d.forward(x, training=True)
        # 大约一半被置零
        zero_ratio = np.mean(out == 0)
        assert 0.3 < zero_ratio < 0.7

    def test_推理模式输出不变(self):
        """推理模式下 dropout 不生效，输出应与输入完全相同"""
        d = Dropout(rate=0.5)
        x = np.random.randn(10, 5)
        out = d.forward(x, training=False)
        np.testing.assert_array_equal(out, x)

    def test_inverted_dropout_期望不变(self):
        """inverted dropout 保持输出期望不变"""
        set_seed(42)
        d = Dropout(rate=0.3)
        x = np.ones((10000, 10))
        out = d.forward(x, training=True)
        # 经过 inverted dropout 后，均值应接近 1.0
        np.testing.assert_allclose(np.mean(out), 1.0, atol=0.1)

    def test_backward_mask一致(self):
        """backward 中的 mask 应与 forward 中一致"""
        set_seed(42)
        d = Dropout(rate=0.5)
        x = np.random.randn(5, 3)
        out = d.forward(x, training=True)
        dout = np.ones_like(x)
        dx = d.backward(dout)
        # 被丢弃的位置梯度也为零
        assert dx.shape == x.shape
        zero_mask_out = out == 0
        zero_mask_grad = dx == 0
        np.testing.assert_array_equal(zero_mask_out, zero_mask_grad)

    def test_rate为零等价于恒等(self):
        """rate=0 时不丢弃任何神经元"""
        d = Dropout(rate=0.0)
        x = np.random.randn(5, 3)
        out = d.forward(x, training=True)
        np.testing.assert_array_equal(out, x)

    def test_参数校验(self):
        """非法 rate 应抛出异常"""
        with pytest.raises(ValueError):
            Dropout(rate=1.0)
        with pytest.raises(ValueError):
            Dropout(rate=-0.1)
