# Copyright (c) 2026 Yy1 (yuan278501381) | MIT License
"""损失函数单元测试"""

import numpy as np

from nn_core.losses import MSE, BinaryCrossEntropy, CategoricalCrossEntropy


class TestMSE:
    """均方误差损失测试"""

    def test_forward_相同输入损失为零(self):
        loss_fn = MSE()
        y = np.array([[1.0], [0.5], [0.0]])
        val = loss_fn.forward(y, y)
        np.testing.assert_allclose(val, 0.0, atol=1e-10)

    def test_forward_已知值计算正确(self):
        loss_fn = MSE()
        y_pred = np.array([[1.0], [0.0]])
        y_true = np.array([[0.0], [1.0]])
        # MSE = mean(1² + 1²) = 1.0
        val = loss_fn.forward(y_pred, y_true)
        np.testing.assert_allclose(val, 1.0, atol=1e-10)

    def test_backward_梯度形状正确(self):
        loss_fn = MSE()
        y_pred = np.random.randn(10, 3)
        y_true = np.random.randn(10, 3)
        loss_fn.forward(y_pred, y_true)
        grad = loss_fn.backward()
        assert grad.shape == y_pred.shape

    def test_backward_梯度方向正确(self):
        """当 pred > true 时，梯度为正（应减小预测值）"""
        loss_fn = MSE()
        y_pred = np.array([[2.0]])
        y_true = np.array([[1.0]])
        loss_fn.forward(y_pred, y_true)
        grad = loss_fn.backward()
        assert grad[0, 0] > 0


class TestBinaryCrossEntropy:
    """二元交叉熵损失测试"""

    def test_forward_完美预测损失接近零(self):
        loss_fn = BinaryCrossEntropy()
        y_pred = np.array([[0.9999], [0.0001]])
        y_true = np.array([[1.0], [0.0]])
        val = loss_fn.forward(y_pred, y_true)
        assert val < 0.01

    def test_forward_已知值计算正确(self):
        loss_fn = BinaryCrossEntropy()
        y_pred = np.array([[0.5]])
        y_true = np.array([[1.0]])
        # BCE = -[1*log(0.5) + 0*log(0.5)] = log(2) ≈ 0.6931
        val = loss_fn.forward(y_pred, y_true)
        np.testing.assert_allclose(val, np.log(2), atol=1e-6)

    def test_backward_梯度形状正确(self):
        loss_fn = BinaryCrossEntropy()
        y_pred = np.random.uniform(0.1, 0.9, (10, 1))
        y_true = np.random.randint(0, 2, (10, 1)).astype(float)
        loss_fn.forward(y_pred, y_true)
        grad = loss_fn.backward()
        assert grad.shape == y_pred.shape

    def test_forward_无nan(self):
        """边界值不产生 NaN"""
        loss_fn = BinaryCrossEntropy()
        y_pred = np.array([[1e-15], [1.0 - 1e-15]])
        y_true = np.array([[1.0], [1.0]])
        val = loss_fn.forward(y_pred, y_true)
        assert not np.isnan(val)


class TestCategoricalCrossEntropy:
    """分类交叉熵损失测试"""

    def test_forward_完美预测损失接近零(self):
        loss_fn = CategoricalCrossEntropy()
        y_pred = np.array([[0.98, 0.01, 0.01]])
        y_true = np.array([[1.0, 0.0, 0.0]])
        val = loss_fn.forward(y_pred, y_true)
        assert val < 0.05

    def test_forward_均匀预测损失为log_n_classes(self):
        """均匀分布预测的损失 = log(n_classes)"""
        loss_fn = CategoricalCrossEntropy()
        n_classes = 4
        y_pred = np.full((1, n_classes), 1.0 / n_classes)
        y_true = np.array([[1.0, 0.0, 0.0, 0.0]])
        val = loss_fn.forward(y_pred, y_true)
        np.testing.assert_allclose(val, np.log(n_classes), atol=1e-6)

    def test_backward_梯度形状正确(self):
        loss_fn = CategoricalCrossEntropy()
        y_pred = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]])
        y_true = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        loss_fn.forward(y_pred, y_true)
        grad = loss_fn.backward()
        assert grad.shape == y_pred.shape
