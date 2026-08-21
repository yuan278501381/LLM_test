# Copyright (c) 2026 Yy1 (yuan278501381) | MIT License
"""
nn_core.gelu - GELU 激活函数模块

实现高斯误差线性单元 (Gaussian Error Linear Unit)。
"""

import logging
import uuid
import numpy as np
from nn_core.activations import Activation

logger = logging.getLogger(__name__)


class GELU(Activation):
    r"""
    GELU 激活函数。
    
    数学公式 (近似):
        $GELU(x) = 0.5 x (1 + \tanh(\sqrt{2/\pi} (x + 0.044715 x^3)))$
    """

    def __init__(self) -> None:
        super().__init__()
        tid = uuid.uuid4().hex[:8]
        logger.debug("[%s] GELU 激活函数已创建", tid)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """前向传播"""
        self.input_cache = x
        inner = np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)
        self.output_cache = 0.5 * x * (1.0 + np.tanh(inner))
        return self.output_cache

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """反向传播"""
        x = self.input_cache
        inner = np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)
        tanh_inner = np.tanh(inner)
        
        # d/dx [ 0.5 * x * (1 + tanh(inner)) ]
        left = 0.5 * (1.0 + tanh_inner)
        
        # d/dx [ tanh(inner) ] = (1 - tanh^2(inner)) * d(inner)/dx
        d_inner = np.sqrt(2.0 / np.pi) * (1.0 + 3.0 * 0.044715 * x**2)
        right = 0.5 * x * (1.0 - tanh_inner**2) * d_inner
        
        dx = dout * (left + right)
        return dx
