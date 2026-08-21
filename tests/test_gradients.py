# Copyright (c) 2026 Yy1 (yuan278501381) | MIT License
"""
梯度数值校验 — 最关键的测试

通过中心差分法计算数值梯度，与反向传播的解析梯度对比，
验证整个梯度链（层、激活函数、损失函数）的正确性。

数值梯度: ∂L/∂θ ≈ (L(θ+ε) - L(θ-ε)) / (2ε)
如果解析梯度与数值梯度的相对误差 < 1e-5，则认为正确。
"""

import numpy as np
import pytest

from nn_core.activations import LeakyReLU, ReLU, Sigmoid, Tanh
from nn_core.layers import Dense
from nn_core.losses import MSE, BinaryCrossEntropy
from nn_core.model import Sequential
from nn_core.regularizers import L2
from nn_core.tensor import set_seed


def numerical_gradient(
    loss_func: callable,
    param: np.ndarray,
    epsilon: float = 1e-5,
) -> np.ndarray:
    """
    用中心差分法计算数值梯度。

    对参数数组中的每个元素，分别加减 epsilon 计算损失差值。

    Args:
        loss_func: 接受无参数并返回标量损失的函数
                   （函数内部通过闭包引用要扰动的参数）
        param: 要计算梯度的参数数组
        epsilon: 扰动大小

    Returns:
        数值梯度，与 param 同形状
    """
    grad = np.zeros_like(param)
    it = np.nditer(param, flags=["multi_index"], op_flags=["readwrite"])

    while not it.finished:
        idx = it.multi_index
        old_val = param[idx]

        # f(θ + ε)
        param[idx] = old_val + epsilon
        loss_plus = loss_func()

        # f(θ - ε)
        param[idx] = old_val - epsilon
        loss_minus = loss_func()

        # 中心差分
        grad[idx] = (loss_plus - loss_minus) / (2.0 * epsilon)

        # 恢复原值
        param[idx] = old_val
        it.iternext()

    return grad


def relative_error(a: np.ndarray, b: np.ndarray) -> float:
    """计算两个数组的最大相对误差"""
    return float(np.max(np.abs(a - b) / np.maximum(np.abs(a) + np.abs(b), 1e-8)))


class TestGradientChecking:
    """梯度数值校验测试"""

    def test_dense_权重梯度(self):
        """验证 Dense 层权重的反向传播梯度"""
        set_seed(42)
        layer = Dense(3, 2, initializer="random")
        loss_fn = MSE()

        x = np.random.randn(5, 3)
        y = np.random.randn(5, 2)

        # 定义完整的前向+损失计算（扰动参数后需要重跑前向）
        def compute_loss():
            pred = layer.forward(x)
            return loss_fn.forward(pred, y)

        # 先跑一次前向+反向获取解析梯度
        compute_loss()
        dloss = loss_fn.backward()
        layer.backward(dloss)
        analytic_grad = layer.grad_weights.copy()

        # 数值梯度（内部每次扰动都会重新 forward）
        num_grad = numerical_gradient(compute_loss, layer.weights)

        err = relative_error(analytic_grad, num_grad)
        assert err < 1e-5, f"Dense 权重梯度误差过大: {err:.2e}"

    def test_dense_偏置梯度(self):
        """验证 Dense 层偏置的反向传播梯度"""
        set_seed(42)
        layer = Dense(3, 2, initializer="random")
        loss_fn = MSE()

        x = np.random.randn(5, 3)
        y = np.random.randn(5, 2)

        def compute_loss():
            pred = layer.forward(x)
            return loss_fn.forward(pred, y)

        compute_loss()
        dloss = loss_fn.backward()
        layer.backward(dloss)
        analytic_grad = layer.grad_biases.copy()

        num_grad = numerical_gradient(compute_loss, layer.biases)
        err = relative_error(analytic_grad, num_grad)
        assert err < 1e-5, f"Dense 偏置梯度误差过大: {err:.2e}"

    @pytest.mark.parametrize("activation_cls", [Sigmoid, ReLU, Tanh, LeakyReLU])
    def test_激活函数梯度(self, activation_cls):
        """验证各激活函数的反向传播梯度"""
        set_seed(42)
        act = activation_cls()

        x = np.random.randn(5, 3) * 0.5  # 避免 ReLU 的零点
        dout = np.random.randn(5, 3)

        # 解析梯度
        act.forward(x)
        analytic_grad = act.backward(dout)

        # 数值梯度：对输入 x 求梯度
        def compute_output_sum():
            out = activation_cls().forward(x)
            return float(np.sum(out * dout))

        # 使用扰动法直接验证
        epsilon = 1e-5
        num_grad = np.zeros_like(x)
        it = np.nditer(x, flags=["multi_index"], op_flags=["readwrite"])
        while not it.finished:
            idx = it.multi_index
            old = x[idx]
            x[idx] = old + epsilon
            f_plus = compute_output_sum()
            x[idx] = old - epsilon
            f_minus = compute_output_sum()
            num_grad[idx] = (f_plus - f_minus) / (2 * epsilon)
            x[idx] = old
            it.iternext()

        err = relative_error(analytic_grad, num_grad)
        assert err < 1e-5, f"{activation_cls.__name__} 梯度误差过大: {err:.2e}"

    @pytest.mark.parametrize("loss_cls", [MSE, BinaryCrossEntropy])
    def test_损失函数梯度(self, loss_cls):
        """验证损失函数的反向传播梯度"""
        set_seed(42)

        if loss_cls == BinaryCrossEntropy:
            y_pred = np.random.uniform(0.1, 0.9, (5, 1))
            y_true = np.random.randint(0, 2, (5, 1)).astype(float)
        else:
            y_pred = np.random.randn(5, 2)
            y_true = np.random.randn(5, 2)

        # 每次都需要新的 loss 实例来避免缓存干扰
        def compute_loss():
            fn = loss_cls()
            return fn.forward(y_pred, y_true)

        # 解析梯度
        loss_fn = loss_cls()
        loss_fn.forward(y_pred, y_true)
        analytic_grad = loss_fn.backward()

        # 数值梯度
        num_grad = numerical_gradient(compute_loss, y_pred)
        err = relative_error(analytic_grad, num_grad)
        assert err < 1e-5, f"{loss_cls.__name__} 梯度误差过大: {err:.2e}"

    def test_多层网络端到端梯度(self):
        """
        最关键的测试 — 验证 2 层网络的端到端反向传播链。

        网络: Dense(2,4) → ReLU → Dense(4,1) → Sigmoid
        损失: BinaryCrossEntropy
        """
        set_seed(42)

        layer1 = Dense(2, 4, initializer="random")
        act1 = ReLU()
        layer2 = Dense(4, 1, initializer="random")
        act2 = Sigmoid()
        loss_fn = BinaryCrossEntropy()

        x = np.random.randn(5, 2)
        y = np.random.randint(0, 2, (5, 1)).astype(float)

        # 完整前向传播函数
        def full_forward():
            z = layer1.forward(x)
            a = act1.forward(z)
            z2 = layer2.forward(a)
            a2 = act2.forward(z2)
            return loss_fn.forward(a2, y)

        # 前向 + 反向
        full_forward()
        dloss = loss_fn.backward()
        da2 = act2.backward(dloss)
        dz2 = layer2.backward(da2)
        da1 = act1.backward(dz2)
        layer1.backward(da1)

        # 验证 layer1 的权重梯度
        analytic_grad_1 = layer1.grad_weights.copy()
        num_grad_1 = numerical_gradient(full_forward, layer1.weights)
        err1 = relative_error(analytic_grad_1, num_grad_1)
        assert err1 < 1e-5, f"Layer1 权重梯度误差: {err1:.2e}"

        # 重新反向传播以获取 layer2 的正确解析梯度
        full_forward()
        dloss = loss_fn.backward()
        da2 = act2.backward(dloss)
        layer2.backward(da2)

        analytic_grad_2 = layer2.grad_weights.copy()
        num_grad_2 = numerical_gradient(full_forward, layer2.weights)
        err2 = relative_error(analytic_grad_2, num_grad_2)
        assert err2 < 1e-5, f"Layer2 权重梯度误差: {err2:.2e}"

    def test_带正则化的梯度(self):
        """验证 L2 正则化下的梯度正确性"""
        set_seed(42)
        reg = L2(lambda_=0.1)
        layer = Dense(3, 2, initializer="random", regularizer=reg)
        loss_fn = MSE()

        x = np.random.randn(5, 3)
        y = np.random.randn(5, 2)

        # 包含正则化项的完整损失
        def compute_total_loss():
            pred = layer.forward(x)
            data_loss = loss_fn.forward(pred, y)
            reg_loss = reg.loss(layer.weights)
            return data_loss + reg_loss

        compute_total_loss()
        dloss = loss_fn.backward()
        layer.backward(dloss)
        analytic_grad = layer.grad_weights.copy()

        num_grad = numerical_gradient(compute_total_loss, layer.weights)
        err = relative_error(analytic_grad, num_grad)
        assert err < 1e-5, f"正则化梯度误差: {err:.2e}"

    def test_三层网络端到端(self):
        """验证 3 层深度网络的梯度链"""
        set_seed(42)

        model = Sequential()
        model.add(Dense(2, 8, initializer="random"))
        model.add(Tanh())
        model.add(Dense(8, 4, initializer="random"))
        model.add(ReLU())
        model.add(Dense(4, 1, initializer="random"))
        model.add(Sigmoid())

        loss_fn = BinaryCrossEntropy()
        x = np.random.randn(5, 2)
        y = np.random.randint(0, 2, (5, 1)).astype(float)

        def full_forward():
            p = model.forward(x, training=False)
            return loss_fn.forward(p, y)

        # 前向 + 反向
        full_forward()
        dloss = loss_fn.backward()
        model.backward(dloss)

        # 检查每个 Dense 层的梯度
        dense_layers = [l for l in model.layers if isinstance(l, Dense)]

        for idx, layer in enumerate(dense_layers):
            analytic = layer.grad_weights.copy()
            num = numerical_gradient(full_forward, layer.weights)
            err = relative_error(analytic, num)
            assert err < 1e-4, f"Dense_{idx} 权重梯度误差: {err:.2e}"
            # 重新前向+反向以刷新下一层的解析梯度
            full_forward()
            dloss = loss_fn.backward()
            model.backward(dloss)
