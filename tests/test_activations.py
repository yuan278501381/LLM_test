# Copyright (c) 2026 Yy1 (yuan278501381) | MIT License
"""激活函数单元测试"""

import numpy as np

from nn_core.activations import LeakyReLU, ReLU, Sigmoid, Softmax, Tanh


class TestSigmoid:
    """Sigmoid 激活函数测试"""

    def test_forward_输出范围在0到1(self):
        """sigmoid 输出必须在 (0, 1) 范围内"""
        s = Sigmoid()
        x = np.random.randn(100, 10)
        out = s.forward(x)
        assert np.all(out > 0) and np.all(out < 1)

    def test_forward_零输入输出05(self):
        """sigmoid(0) = 0.5"""
        s = Sigmoid()
        out = s.forward(np.array([[0.0]]))
        np.testing.assert_allclose(out, 0.5, atol=1e-10)

    def test_forward_大输入数值稳定(self):
        """极大/极小输入不产生 NaN 或 Inf"""
        s = Sigmoid()
        x = np.array([[1000.0, -1000.0, 500.0, -500.0]])
        out = s.forward(x)
        assert not np.any(np.isnan(out))
        assert not np.any(np.isinf(out))
        np.testing.assert_allclose(out[0, 0], 1.0, atol=1e-6)
        np.testing.assert_allclose(out[0, 1], 0.0, atol=1e-6)

    def test_backward_梯度形状正确(self):
        """backward 输出形状 == forward 输入形状"""
        s = Sigmoid()
        x = np.random.randn(5, 3)
        s.forward(x)
        dout = np.ones_like(x)
        dx = s.backward(dout)
        assert dx.shape == x.shape

    def test_backward_梯度最大值在零点(self):
        """sigmoid 在 x=0 处梯度最大 (= 0.25)"""
        s = Sigmoid()
        s.forward(np.array([[0.0]]))
        dx = s.backward(np.array([[1.0]]))
        np.testing.assert_allclose(dx, 0.25, atol=1e-10)


class TestReLU:
    """ReLU 激活函数测试"""

    def test_forward_正数不变(self):
        x = np.array([[1.0, 2.0, 3.0]])
        out = ReLU().forward(x)
        np.testing.assert_array_equal(out, x)

    def test_forward_负数归零(self):
        x = np.array([[-1.0, -2.0, -3.0]])
        out = ReLU().forward(x)
        np.testing.assert_array_equal(out, np.zeros_like(x))

    def test_backward_正数梯度为1(self):
        r = ReLU()
        x = np.array([[1.0, -1.0, 2.0, -2.0]])
        r.forward(x)
        dx = r.backward(np.ones_like(x))
        expected = np.array([[1.0, 0.0, 1.0, 0.0]])
        np.testing.assert_array_equal(dx, expected)


class TestTanh:
    """Tanh 激活函数测试"""

    def test_forward_输出范围负1到1(self):
        t = Tanh()
        x = np.random.randn(100, 10)
        out = t.forward(x)
        assert np.all(out >= -1) and np.all(out <= 1)

    def test_forward_零输入输出零(self):
        t = Tanh()
        out = t.forward(np.array([[0.0]]))
        np.testing.assert_allclose(out, 0.0, atol=1e-10)

    def test_backward_零点梯度为1(self):
        """tanh'(0) = 1 - tanh²(0) = 1"""
        t = Tanh()
        t.forward(np.array([[0.0]]))
        dx = t.backward(np.array([[1.0]]))
        np.testing.assert_allclose(dx, 1.0, atol=1e-10)


class TestLeakyReLU:
    """LeakyReLU 激活函数测试"""

    def test_forward_正数不变(self):
        lr = LeakyReLU(alpha=0.01)
        x = np.array([[1.0, 2.0, 3.0]])
        out = lr.forward(x)
        np.testing.assert_array_equal(out, x)

    def test_forward_负数乘alpha(self):
        lr = LeakyReLU(alpha=0.1)
        x = np.array([[-1.0, -2.0]])
        out = lr.forward(x)
        np.testing.assert_allclose(out, [[-0.1, -0.2]])

    def test_backward_负数梯度为alpha(self):
        alpha = 0.05
        lr = LeakyReLU(alpha=alpha)
        x = np.array([[1.0, -1.0]])
        lr.forward(x)
        dx = lr.backward(np.ones_like(x))
        np.testing.assert_allclose(dx, [[1.0, alpha]])


class TestSoftmax:
    """Softmax 激活函数测试"""

    def test_forward_输出和为1(self):
        """每行概率之和应为 1"""
        sm = Softmax()
        x = np.random.randn(10, 5)
        out = sm.forward(x)
        row_sums = np.sum(out, axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-10)

    def test_forward_输出全为正(self):
        sm = Softmax()
        x = np.random.randn(10, 5)
        out = sm.forward(x)
        assert np.all(out > 0)

    def test_forward_大输入数值稳定(self):
        """极大输入不溢出"""
        sm = Softmax()
        x = np.array([[1000.0, 1001.0, 999.0]])
        out = sm.forward(x)
        assert not np.any(np.isnan(out))
        assert not np.any(np.isinf(out))
        np.testing.assert_allclose(np.sum(out), 1.0, atol=1e-10)

    def test_backward_形状正确(self):
        sm = Softmax()
        x = np.random.randn(5, 4)
        sm.forward(x)
        dout = np.random.randn(5, 4)
        dx = sm.backward(dout)
        assert dx.shape == x.shape
