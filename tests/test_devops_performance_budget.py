# Copyright (c) 2026 Yy1 (yuan278501381) | MIT License
"""
tests/test_devops_performance_budget.py - 核心算法性能预算与延迟守卫 (Tier-1 DevOps Gate)

严格锁定核心神经网络算子与算法的计算耗时基线 (Performance Budget)。
严防在重构中引入低效循环、冗余内存拷贝或算法复杂度退化。
"""

import time

import numpy as np

from nn_core.layers import Dense
from nn_core.reinforcement import BellmanSolver, GridWorldEnv
from nn_core.transformer import TransformerBlock


def test_reinforcement_learning_performance_budget():
    """强化学习 Bellman 值迭代必须在 50ms 内完成解析收敛"""
    env = GridWorldEnv(grid_type="cliff")
    solver = BellmanSolver(env, gamma=0.95, theta=1e-4)

    t0 = time.perf_counter()
    _v, _pi, steps = solver.solve()
    elapsed_ms = (time.perf_counter() - t0) * 1000

    assert steps > 0
    assert elapsed_ms < 100.0, f"BellmanSolver 耗时超标 ({elapsed_ms:.2f}ms > 100ms)"


def test_dense_forward_backward_performance_budget():
    """全连接层 1000 样本前向 + 反向计算延迟必须 < 10ms"""
    layer = Dense(64, 32, initializer="he")
    X = np.random.randn(1000, 64)
    d_out = np.random.randn(1000, 32)

    t0 = time.perf_counter()
    for _ in range(10):
        out = layer.forward(X)
        dX = layer.backward(d_out)
    elapsed_ms = ((time.perf_counter() - t0) / 10) * 1000

    assert out.shape == (1000, 32)
    assert dX.shape == (1000, 64)
    assert elapsed_ms < 15.0, f"Dense forward/backward 耗时超标 ({elapsed_ms:.2f}ms > 15ms)"


def test_transformer_block_performance_budget():
    """Transformer 结构块前向推理单步延迟必须 < 25ms"""
    block = TransformerBlock(d_model=64, num_heads=4, d_ff=256)
    x = np.random.randn(2, 16, 64)

    t0 = time.perf_counter()
    for _ in range(5):
        out, _weights = block.forward(x)
    elapsed_ms = ((time.perf_counter() - t0) / 5) * 1000

    assert out.shape == (2, 16, 64)
    assert elapsed_ms < 40.0, f"TransformerBlock 耗时超标 ({elapsed_ms:.2f}ms > 40ms)"
