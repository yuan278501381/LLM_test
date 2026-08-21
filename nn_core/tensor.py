# Copyright (c) 2026 Yy1 (yuan278501381) | MIT License
"""
nn_core.tensor - 数值稳定性工具模块

提供底层数值运算的安全封装，防止在深度学习计算中出现
NaN、Inf 等数值溢出问题。同时提供随机种子管理确保实验可复现。
"""

import logging
import uuid

import numpy as np

# ---------------------------------------------------------------------------
# 日志配置
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)


def _trace_id() -> str:
    """生成短格式 TraceID，用于日志追踪"""
    return uuid.uuid4().hex[:8]


# ---------------------------------------------------------------------------
# 随机种子管理
# ---------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    """
    设置全局随机种子，确保实验结果可复现。

    Args:
        seed: 随机种子值
    """
    tid = _trace_id()
    np.random.seed(seed)
    logger.info("[%s] 随机种子已设置为 %d", tid, seed)


# ---------------------------------------------------------------------------
# 数值安全工具
# ---------------------------------------------------------------------------
def clip_gradients(grads: np.ndarray, max_norm: float = 5.0) -> np.ndarray:
    """
    梯度裁剪 - 防止梯度爆炸。

    当梯度的 L2 范数超过 max_norm 时，按比例缩放梯度，
    使其范数恰好等于 max_norm。

    数学公式:
        如果 ||grads|| > max_norm:
            grads = grads * max_norm / ||grads||

    Args:
        grads: 梯度数组
        max_norm: 允许的最大梯度范数，默认 5.0

    Returns:
        裁剪后的梯度数组（原数组不被修改）
    """
    norm = np.linalg.norm(grads)
    if norm > max_norm:
        logger.debug("梯度裁剪: ||grad||=%.4f > max_norm=%.4f", norm, max_norm)
        return grads * max_norm / norm
    return grads


def safe_log(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    安全对数 - 避免 log(0) 导致 -inf。

    将输入裁剪到 [eps, +∞) 范围后再取对数。

    Args:
        x: 输入数组（通常是概率值，范围 [0, 1]）
        eps: 下界裁剪值，默认 1e-12

    Returns:
        安全的对数值
    """
    return np.log(np.clip(x, eps, None))


def safe_exp(x: np.ndarray, max_val: float = 500.0) -> np.ndarray:
    """
    安全指数 - 防止 exp() 上溢出为 inf。

    将输入裁剪到 [-max_val, max_val] 范围后再取指数。

    Args:
        x: 输入数组
        max_val: 裁剪范围上界，默认 500.0（exp(500) ≈ 1.4e217）

    Returns:
        安全的指数值
    """
    return np.exp(np.clip(x, -max_val, max_val))
