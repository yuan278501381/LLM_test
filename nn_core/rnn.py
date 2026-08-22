# Copyright (c) 2026 Yy1 (yuan278501381) | MIT License
"""
nn_core.rnn - 循环神经网络模块

实现基础的循环神经网络结构。
"""

import logging
import uuid

import numpy as np

logger = logging.getLogger(__name__)


class RNNCell:
    r"""
    最简 Vanilla RNN 单元。

    数学公式:
        $h_t = \tanh(x_t \cdot W_{xh} + h_{prev} \cdot W_{hh} + b_h)$
    """

    def __init__(self, input_size: int, hidden_size: int) -> None:
        self.input_size = input_size
        self.hidden_size = hidden_size

        # 初始化权重 (Xavier 初始化)
        limit_xh = np.sqrt(6 / (input_size + hidden_size))
        self.W_xh: np.ndarray = np.random.uniform(-limit_xh, limit_xh, (input_size, hidden_size))

        limit_hh = np.sqrt(6 / (hidden_size + hidden_size))
        self.W_hh: np.ndarray = np.random.uniform(-limit_hh, limit_hh, (hidden_size, hidden_size))

        self.b_h: np.ndarray = np.zeros((1, hidden_size))

        # 缓存
        self.cache: list = []

        tid = uuid.uuid4().hex[:8]
        logger.info("[%s] RNNCell 已创建: input=%d, hidden=%d", tid, input_size, hidden_size)

    def forward(self, x_t: np.ndarray, h_prev: np.ndarray) -> np.ndarray:
        """
        前向传播单个时间步。
        """
        # z_t = x_t @ W_xh + h_prev @ W_hh + b_h
        z_t = x_t @ self.W_xh + h_prev @ self.W_hh + self.b_h
        h_t = np.tanh(z_t)

        self.cache.append((x_t, h_prev, h_t))
        return h_t

    def step_sequence(self, X_seq: np.ndarray) -> list[np.ndarray]:
        """
        对整个序列逐步执行。

        Args:
            X_seq: 形状 (seq_len, batch_size, input_size) 或 (batch_size, seq_len, input_size)
                   假设采用 (batch_size, seq_len, input_size)。

        Returns:
            所有隐藏状态列表
        """
        self.cache = []
        batch_size, seq_len, _ = X_seq.shape
        h_prev = np.zeros((batch_size, self.hidden_size))

        h_states = []
        for t in range(seq_len):
            x_t = X_seq[:, t, :]
            h_prev = self.forward(x_t, h_prev)
            h_states.append(h_prev)

        return h_states

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """
        反向传播 (BPTT)。
        为教学用途，当前省略 backward 实现，主要展示前向瓶颈与遗忘特性。
        """
        # 教学用，只展示前向和遗忘瓶颈
        raise NotImplementedError("教学用模块：RNN 反向传播（BPTT）已省略")
