# Copyright (c) 2026 Yy1 (yuan278501381) | MIT License
"""
nn_core.conv2d - 纯 NumPy 2D 卷积与池化计算引擎 (基于 im2col 向量化实现)

包含：
- `im2col` / `col2im`: 高性能图像张量展开与逆重构算子
- `Conv2D`: 具备完整解析梯度与通用参数协议的 2D 卷积层
- `MaxPool2D`: 2D 最大池化与 argmax 梯度反向路由层
"""

import logging

import numpy as np

logger = logging.getLogger("nn_core.conv2d")


def im2col(x: np.ndarray, kh: int, kw: int, stride: int = 1, pad: int = 0) -> np.ndarray:
    """
    将 4D 图像张量 (N, C, H, W) 按照卷积核感受野滑动展开为 2D 矩阵 (N*out_h*out_w, C*kh*kw)。

    数学原理：
    通过零填充与步长滑动采样，将原本多层循环的局部滑动点积转化为单次高效的 BLAS 矩阵乘法。
    """
    N, C, H, W = x.shape
    out_h = (H + 2 * pad - kh) // stride + 1
    out_w = (W + 2 * pad - kw) // stride + 1

    img = np.pad(x, [(0, 0), (0, 0), (pad, pad), (pad, pad)], mode="constant") if pad > 0 else x

    col = np.zeros((N, C, kh, kw, out_h, out_w), dtype=x.dtype)

    for y in range(kh):
        y_max = y + stride * out_h
        for x_idx in range(kw):
            x_max = x_idx + stride * out_w
            col[:, :, y, x_idx, :, :] = img[:, :, y:y_max:stride, x_idx:x_max:stride]

    # 维度转置重排: (N, out_h, out_w, C, kh, kw) -> (N*out_h*out_w, C*kh*kw)
    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N * out_h * out_w, -1)
    return col


def col2im(
    col: np.ndarray,
    x_shape: tuple[int, int, int, int],
    kh: int,
    kw: int,
    stride: int = 1,
    pad: int = 0,
) -> np.ndarray:
    """
    im2col 的精确逆运算，用于反向传播中将 2D 梯度累加重构回 4D 图像梯度张量 (N, C, H, W)。
    """
    N, C, H, W = x_shape
    out_h = (H + 2 * pad - kh) // stride + 1
    out_w = (W + 2 * pad - kw) // stride + 1

    col_reshaped = col.reshape(N, out_h, out_w, C, kh, kw).transpose(0, 3, 4, 5, 1, 2)
    img = np.zeros((N, C, H + 2 * pad + stride - 1, W + 2 * pad + stride - 1), dtype=col.dtype)

    for y in range(kh):
        y_max = y + stride * out_h
        for x_idx in range(kw):
            x_max = x_idx + stride * out_w
            img[:, :, y:y_max:stride, x_idx:x_max:stride] += col_reshaped[:, :, y, x_idx, :, :]

    if pad > 0:
        return img[:, :, pad : H + pad, pad : W + pad]
    return img[:, :, :H, :W]


class Conv2D:
    """
    纯 NumPy 2D 卷积层 (基于 im2col 高效矩阵化运算)。

    前向计算：
        Y = im2col(X) @ W_col + b
    反向微分：
        dW = col.T @ dout_reshaped
        db = sum(dout_reshaped, axis=0)
        dX = col2im(dout_reshaped @ W_col.T)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 0,
    ) -> None:
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # 权重初始化：He / Kaiming 正态初始化
        fan_in = in_channels * kernel_size * kernel_size
        self.weights = np.random.randn(out_channels, in_channels, kernel_size, kernel_size).astype(
            np.float64
        ) * np.sqrt(2.0 / fan_in)
        self.biases = np.zeros(out_channels, dtype=np.float64)

        # 梯度容器
        self.grad_weights = np.zeros_like(self.weights)
        self.grad_biases = np.zeros_like(self.biases)

        # 缓存
        self._x: np.ndarray | None = None
        self._col: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        前向卷积传播。
        x shape: (N, in_channels, H, W)
        返回 shape: (N, out_channels, out_h, out_w)
        """
        self._x = x
        N, _C, H, W = x.shape
        kh = kw = self.kernel_size
        out_h = (H + 2 * self.padding - kh) // self.stride + 1
        out_w = (W + 2 * self.padding - kw) // self.stride + 1

        col = im2col(x, kh, kw, self.stride, self.padding)
        self._col = col

        # 权重矩阵展平为 (out_channels, C * kh * kw).T -> (C * kh * kw, out_channels)
        W_col = self.weights.reshape(self.out_channels, -1).T
        out = np.dot(col, W_col) + self.biases

        # 重构回 (N, out_channels, out_h, out_w)
        out = out.reshape(N, out_h, out_w, self.out_channels).transpose(0, 3, 1, 2)
        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """
        反向微分传播。
        dout shape: (N, out_channels, out_h, out_w)
        返回 dx shape: (N, in_channels, H, W)
        """
        if self._x is None or self._col is None:
            raise RuntimeError("必须先调用 forward() 方可执行 backward()")

        kh = kw = self.kernel_size
        # dout 转置重塑为 (N * out_h * out_w, out_channels)
        dout_reshaped = dout.transpose(0, 2, 3, 1).reshape(-1, self.out_channels)

        self.grad_biases = np.sum(dout_reshaped, axis=0)
        # grad_weights: (out_channels, C * kh * kw) -> (out_channels, C, kh, kw)
        dW_col = np.dot(self._col.T, dout_reshaped)
        self.grad_weights = dW_col.T.reshape(self.weights.shape)

        W_col = self.weights.reshape(self.out_channels, -1).T
        dcol = np.dot(dout_reshaped, W_col.T)
        dx = col2im(dcol, self._x.shape, kh, kw, self.stride, self.padding)
        return dx

    def get_params_and_grads(self) -> list[tuple[np.ndarray, np.ndarray]]:
        """实现优化器通用参数协议"""
        return [
            (self.weights, self.grad_weights),
            (self.biases, self.grad_biases),
        ]


class MaxPool2D:
    """
    纯 NumPy 2D 最大池化层。
    """

    def __init__(self, pool_size: int = 2, stride: int = 2) -> None:
        self.pool_size = pool_size
        self.stride = stride
        self._x: np.ndarray | None = None
        self._arg_max: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        前向最大池化。
        x shape: (N, C, H, W)
        """
        self._x = x
        N, C, H, W = x.shape
        ph = pw = self.pool_size
        out_h = (H - ph) // self.stride + 1
        out_w = (W - pw) // self.stride + 1

        col = im2col(x, ph, pw, self.stride, pad=0)
        col = col.reshape(-1, ph * pw)

        arg_max = np.argmax(col, axis=1)
        out = np.max(col, axis=1)
        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)

        self._arg_max = arg_max
        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """
        反向梯度路由。
        """
        if self._x is None or self._arg_max is None:
            raise RuntimeError("必须先调用 forward() 方可执行 backward()")

        dout_reshaped = dout.transpose(0, 2, 3, 1).flatten()
        ph = pw = self.pool_size

        dcol = np.zeros((dout_reshaped.size, ph * pw), dtype=dout.dtype)
        dcol[np.arange(self._arg_max.size), self._arg_max.flatten()] = dout_reshaped

        dx = col2im(dcol, self._x.shape, ph, pw, self.stride, pad=0)
        return dx
