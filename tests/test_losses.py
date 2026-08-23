# Copyright (c) 2026 Yy1 (yuan278501381) | MIT License
"""损失函数单元测试与数学/契约全覆盖"""

import numpy as np
import pytest

from nn_core.losses import (
    MSE,
    BCEWithLogitsLoss,
    BinaryCrossEntropy,
    CategoricalCrossEntropy,
    CategoricalCrossEntropyWithLogits,
)


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

    def test_input_validation_empty_and_nan(self):
        loss_fn = MSE()
        with pytest.raises(ValueError, match="不能为空"):
            loss_fn.forward(np.array([]), np.array([]))
        with pytest.raises(ValueError, match="包含 NaN 或 Inf"):
            loss_fn.forward(np.array([[np.nan]]), np.array([[1.0]]))
        with pytest.raises(ValueError, match="形状不匹配"):
            loss_fn.forward(np.array([[1.0], [2.0]]), np.array([[1.0]]))


class TestBinaryCrossEntropy:
    """二元交叉熵损失测试 (严格概率域)"""

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
        val = loss_fn.forward(y_pred, y_true)
        np.testing.assert_allclose(val, np.log(2), atol=1e-6)

    def test_backward_梯度形状正确(self):
        loss_fn = BinaryCrossEntropy()
        y_pred = np.random.uniform(0.1, 0.9, (10, 1))
        y_true = np.random.randint(0, 2, (10, 1)).astype(float)
        loss_fn.forward(y_pred, y_true)
        grad = loss_fn.backward()
        assert grad.shape == y_pred.shape

    def test_forward_边界端点无nan(self):
        """极端边界值 (0.0 与 1.0) 不产生 NaN"""
        loss_fn = BinaryCrossEntropy()
        y_pred = np.array([[0.0], [1.0]])
        y_true = np.array([[0.0], [1.0]])
        val = loss_fn.forward(y_pred, y_true)
        assert not np.isnan(val)
        assert val < 1e-4

    def test_forward_非法与越界输入抛出明确ValueError(self):
        """当概率输入超出 [0, 1] 时，BCE 必须抛出 ValueError，杜绝静默掩盖模型契约错误"""
        loss_fn = BinaryCrossEntropy()
        with pytest.raises(ValueError, match="BinaryCrossEntropy 期望输入为严格概率值"):
            loss_fn.forward(np.array([[2.0], [0.5]]), np.array([[1.0], [0.0]]))
        with pytest.raises(ValueError, match="BinaryCrossEntropy 期望输入为严格概率值"):
            loss_fn.forward(np.array([[-0.01], [0.5]]), np.array([[0.0], [1.0]]))

    def test_forward_非法标签抛出ValueError(self):
        loss_fn = BinaryCrossEntropy()
        with pytest.raises(ValueError, match="真实二分类标签 y_true 取值严格在"):
            loss_fn.forward(np.array([[0.8], [0.2]]), np.array([[2.0], [0.0]]))

    def test_forward_nan_inf_抛出ValueError(self):
        loss_fn = BinaryCrossEntropy()
        with pytest.raises(ValueError, match="包含 NaN 或 Inf"):
            loss_fn.forward(np.array([[np.nan]]), np.array([[1.0]]))
        with pytest.raises(ValueError, match="包含 NaN 或 Inf"):
            loss_fn.forward(np.array([[0.5]]), np.array([[np.inf]]))

    def test_backward_数值梯度一致性(self):
        """验证在合法概率定义域内解析梯度与中心差分数值梯度一致"""
        loss_fn = BinaryCrossEntropy()
        y_pred = np.array([[0.35], [0.75]])
        y_true = np.array([[1.0], [0.0]])
        loss_fn.forward(y_pred, y_true)
        analytical_grad = loss_fn.backward()

        eps = 1e-6
        num_grad = np.zeros_like(y_pred)
        for i in range(y_pred.shape[0]):
            yp_plus = y_pred.copy()
            yp_minus = y_pred.copy()
            yp_plus[i, 0] += eps
            yp_minus[i, 0] -= eps
            lp = loss_fn.forward(yp_plus, y_true)
            lm = loss_fn.forward(yp_minus, y_true)
            num_grad[i, 0] = (lp - lm) / (2.0 * eps)

        loss_fn.forward(y_pred, y_true)
        np.testing.assert_allclose(analytical_grad, num_grad, rtol=1e-4, atol=1e-5)


class TestBCEWithLogitsLoss:
    """带 Logits 的二元交叉熵损失测试 (实数域数值稳定)"""

    def test_forward_and_backward_全实数域数值稳定(self):
        """测试在极端正负 Logits 下无 NaN / Inf 且损失非负"""
        loss_fn = BCEWithLogitsLoss()
        logits = np.array([[-100.0], [0.0], [100.0]])
        y_true = np.array([[0.0], [1.0], [1.0]])
        val = loss_fn.forward(logits, y_true)
        assert not np.isnan(val)
        assert not np.isinf(val)
        assert val >= 0.0
        grad = loss_fn.backward()
        assert not np.any(np.isnan(grad))
        assert not np.any(np.isinf(grad))

    def test_forward_非法标签与异常值显式报错(self):
        loss_fn = BCEWithLogitsLoss()
        with pytest.raises(ValueError, match="真实二分类标签 y_true 取值严格在"):
            loss_fn.forward(np.array([[2.0], [1.0]]), np.array([[3.0], [0.0]]))
        with pytest.raises(ValueError, match="包含 NaN 或 Inf"):
            loss_fn.forward(np.array([[np.nan]]), np.array([[1.0]]))
        with pytest.raises(ValueError, match="包含 NaN 或 Inf"):
            loss_fn.forward(np.array([[np.inf]]), np.array([[1.0]]))

    def test_backward_logits_数值梯度一致性(self):
        """验证 BCEWithLogitsLoss 的解析梯度与中心差分数值梯度一致"""
        loss_fn = BCEWithLogitsLoss()
        logits = np.array([[-2.5], [0.8], [3.2]])
        y_true = np.array([[0.0], [1.0], [0.0]])
        loss_fn.forward(logits, y_true)
        analytical_grad = loss_fn.backward()

        eps = 1e-6
        num_grad = np.zeros_like(logits)
        for i in range(logits.shape[0]):
            lp_vec = logits.copy()
            lm_vec = logits.copy()
            lp_vec[i, 0] += eps
            lm_vec[i, 0] -= eps
            lp = loss_fn.forward(lp_vec, y_true)
            lm = loss_fn.forward(lm_vec, y_true)
            num_grad[i, 0] = (lp - lm) / (2.0 * eps)

        loss_fn.forward(logits, y_true)
        np.testing.assert_allclose(analytical_grad, num_grad, rtol=1e-4, atol=1e-5)


class TestCategoricalCrossEntropy:
    """分类交叉熵损失测试 (概率单纯形输入)"""

    def test_forward_完美预测损失接近零(self):
        loss_fn = CategoricalCrossEntropy()
        y_pred = np.array([[0.98, 0.01, 0.01]])
        y_true = np.array([[1.0, 0.0, 0.0]])
        val = loss_fn.forward(y_pred, y_true)
        assert val < 0.05

    def test_forward_均匀预测损失为log_n_classes(self):
        loss_fn = CategoricalCrossEntropy()
        n_classes = 4
        y_pred = np.full((1, n_classes), 1.0 / n_classes)
        y_true = np.array([[1.0, 0.0, 0.0, 0.0]])
        val = loss_fn.forward(y_pred, y_true)
        np.testing.assert_allclose(val, np.log(n_classes), atol=1e-6)

    def test_forward_非法输入与单纯形校验(self):
        loss_fn = CategoricalCrossEntropy()
        # 负概率值
        with pytest.raises(ValueError, match="合法概率值"):
            loss_fn.forward(np.array([[1.5, -0.5]]), np.array([[1.0, 0.0]]))
        # 概率和不为 1.0
        with pytest.raises(ValueError, match=r"每行概率和为 1\.0"):
            loss_fn.forward(np.array([[0.5, 0.2]]), np.array([[1.0, 0.0]]))

    def test_backward_数值梯度一致性(self):
        loss_fn = CategoricalCrossEntropy()
        y_pred = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]])
        y_true = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        loss_fn.forward(y_pred, y_true)
        analytical_grad = loss_fn.backward()

        eps = 1e-6
        num_grad = np.zeros_like(y_pred)
        for i in range(y_pred.shape[0]):
            for j in range(y_pred.shape[1]):
                yp_plus = y_pred.copy()
                yp_minus = y_pred.copy()
                yp_plus[i, j] += eps
                yp_minus[i, j] -= eps
                lp = -np.mean(np.sum(y_true * np.log(yp_plus), axis=1))
                lm = -np.mean(np.sum(y_true * np.log(yp_minus), axis=1))
                num_grad[i, j] = (lp - lm) / (2.0 * eps)

        np.testing.assert_allclose(analytical_grad, num_grad, rtol=1e-4, atol=1e-5)


class TestCategoricalCrossEntropyWithLogits:
    """带 Logits 的数值稳定多分类交叉熵测试"""

    def test_forward_and_backward_已知数值与梯度(self):
        loss_fn = CategoricalCrossEntropyWithLogits()
        logits = np.array([[2.0, 1.0, 0.1], [-1.0, 3.0, 0.5]])
        y_true = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        val = loss_fn.forward(logits, y_true)
        assert val > 0.0
        grad = loss_fn.backward()
        assert grad.shape == logits.shape

    def test_backward_logits_数值梯度一致性(self):
        loss_fn = CategoricalCrossEntropyWithLogits()
        logits = np.array([[1.2, -0.5, 0.8], [-2.0, 0.5, 1.5]])
        y_true = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        loss_fn.forward(logits, y_true)
        analytical_grad = loss_fn.backward()

        eps = 1e-6
        num_grad = np.zeros_like(logits)
        for i in range(logits.shape[0]):
            for j in range(logits.shape[1]):
                lp_vec = logits.copy()
                lm_vec = logits.copy()
                lp_vec[i, j] += eps
                lm_vec[i, j] -= eps
                lp = loss_fn.forward(lp_vec, y_true)
                lm = loss_fn.forward(lm_vec, y_true)
                num_grad[i, j] = (lp - lm) / (2.0 * eps)

        loss_fn.forward(logits, y_true)
        np.testing.assert_allclose(analytical_grad, num_grad, rtol=1e-4, atol=1e-5)
