# Copyright (c) 2026 Yy1 (yuan278501381) | MIT License
"""优化器单元测试"""

import numpy as np
import pytest

from nn_core.layers import Dense
from nn_core.losses import MSE
from nn_core.optimizers import SGD, Adam, Momentum, RMSProp
from nn_core.tensor import set_seed


class TestSGD:
    """SGD 优化器测试"""

    def test_一步更新方向正确(self):
        """梯度为正时，权重应减小"""
        set_seed(42)
        layer = Dense(2, 1, initializer="random")
        old_w = layer.weights.copy()

        # 设置正梯度
        layer.grad_weights = np.array([[0.1], [0.2]])
        layer.grad_biases = np.array([[0.05]])

        sgd = SGD(learning_rate=0.1)
        sgd.step([layer])

        # 权重应减小
        assert np.all(layer.weights < old_w)

    def test_学习率影响步长(self):
        """lr=0.1 的步长是 lr=0.01 的 10 倍"""
        set_seed(42)
        layer1 = Dense(2, 1, initializer="random")
        layer2 = Dense(2, 1, initializer="random")
        layer2.weights = layer1.weights.copy()
        layer2.biases = layer1.biases.copy()

        grad = np.array([[1.0], [1.0]])
        layer1.grad_weights = grad.copy()
        layer1.grad_biases = np.array([[1.0]])
        layer2.grad_weights = grad.copy()
        layer2.grad_biases = np.array([[1.0]])

        SGD(learning_rate=0.1).step([layer1])
        SGD(learning_rate=0.01).step([layer2])

        # 计算步长比
        w_orig = layer1.weights + 0.1 * grad  # 恢复原始
        step1 = np.abs(w_orig - layer1.weights)
        step2 = np.abs(w_orig - layer2.weights)
        np.testing.assert_allclose(step1 / step2, 10.0, atol=1e-6)


class TestMomentum:
    """Momentum 优化器测试"""

    def test_动量累积_步长逐渐增大(self):
        """连续多步相同梯度方向，步长应逐渐增大"""
        set_seed(42)
        layer = Dense(1, 1, initializer="random")

        mom = Momentum(learning_rate=0.01, beta=0.9)
        steps = []

        for _ in range(5):
            old_w = layer.weights.copy()
            layer.grad_weights = np.array([[1.0]])
            layer.grad_biases = np.array([[0.0]])
            mom.step([layer])
            step_size = float(np.abs(layer.weights - old_w).item())
            steps.append(step_size)

        # 步长应递增
        for i in range(1, len(steps)):
            assert steps[i] > steps[i - 1]


class TestRMSProp:
    """RMSProp 优化器测试"""

    def test_一步更新方向正确(self):
        """梯度为正时，权重应减小"""
        set_seed(42)
        layer = Dense(2, 1, initializer="random")
        old_w = layer.weights.copy()
        layer.grad_weights = np.array([[0.5], [0.5]])
        layer.grad_biases = np.array([[0.1]])

        rms = RMSProp(learning_rate=0.01)
        rms.step([layer])
        assert np.all(layer.weights < old_w)


class TestAdam:
    """Adam 优化器测试"""

    def test_一步更新方向正确(self):
        """梯度为正时，权重应减小"""
        set_seed(42)
        layer = Dense(2, 1, initializer="random")
        old_w = layer.weights.copy()
        layer.grad_weights = np.array([[0.5], [0.5]])
        layer.grad_biases = np.array([[0.1]])

        adam = Adam(learning_rate=0.01)
        adam.step([layer])
        assert np.all(layer.weights < old_w)

    def test_偏差修正_初始步长合理(self):
        """前几步经过偏差修正后步长应合理"""
        set_seed(42)
        layer = Dense(1, 1, initializer="random")

        adam = Adam(learning_rate=0.001, beta1=0.9, beta2=0.999)
        layer.grad_weights = np.array([[1.0]])
        layer.grad_biases = np.array([[0.0]])

        old_w = layer.weights.copy()
        adam.step([layer])

        # 第一步的更新不应该太小（因为偏差修正放大了估计）
        step = float(np.abs(layer.weights - old_w).item())
        assert step > 0.0005


class TestAllOptimizers:
    """所有优化器的通用收敛测试"""

    @pytest.mark.parametrize(
        "optimizer_cls,kwargs",
        [
            (SGD, {"learning_rate": 0.1}),
            (Momentum, {"learning_rate": 0.1, "beta": 0.9}),
            (RMSProp, {"learning_rate": 0.01}),
            (Adam, {"learning_rate": 0.01}),
        ],
    )
    def test_简单二次函数收敛(self, optimizer_cls, kwargs):
        """在 f(x)=x² 上应能收敛到 pred≈0"""
        set_seed(42)

        # 用 Dense(1,1) 模拟 pred = w*x + b，目标 pred → 0
        layer = Dense(1, 1, initializer="random")
        layer.weights = np.array([[2.0]])  # 初始远离最优

        optimizer = optimizer_cls(**kwargs)
        loss_fn = MSE()

        for _ in range(200):
            x = np.array([[1.0]])
            y_true = np.array([[0.0]])
            y_pred = layer.forward(x)
            loss_fn.forward(y_pred, y_true)
            dloss = loss_fn.backward()
            layer.backward(dloss)
            optimizer.step([layer])

        # 预测值应接近 0（模型有 w 和 b 两个参数，组合逼近目标）
        final_pred = layer.forward(np.array([[1.0]])).item()
        assert abs(final_pred) < 0.1, f"{optimizer_cls.__name__} 未收敛: pred={final_pred:.6f}"
