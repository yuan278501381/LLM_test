# Copyright (c) 2026 Yy1 (yuan278501381) | MIT License
"""
nn_core.world_model - 世界模型下一帧动力学推演与 Diffusion 扩散去噪调度器 (对标 Sora / DiT)

包含：
- `NextFramePredictor`: 基于上下文序列嵌入自回归推演未来帧像素的世界模型预测头
- `DiffusionScheduler`: 连续高斯前向加噪与去噪方差调度引擎 (DDPM / DiT 架构基石)
- `visualize_diffusion_process`: 扩散时间步前向轨迹模拟
"""

import logging
from typing import Optional, Tuple
import numpy as np

logger = logging.getLogger("nn_core.world_model")


class NextFramePredictor:
    """
    纯 NumPy 教学级自回归下一帧世界模型预测器。
    通过输入历史帧的 Transformer 上下文表征，推演下一帧的物理画面像素分布。
    """

    def __init__(self, d_model: int = 32, frame_pixels: int = 1024) -> None:
        self.d_model = d_model
        self.frame_pixels = frame_pixels

        # 两层解码投影 MLP
        hidden_dim = 128
        self.W1 = np.random.randn(d_model, hidden_dim) * np.sqrt(2.0 / d_model)
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, frame_pixels) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros(frame_pixels)

    def forward(self, context_embeds: np.ndarray) -> np.ndarray:
        """
        context_embeds: (B, d_model) 代表前序时间步的历史聚合特征
        返回预测帧展平像素: (B, frame_pixels)，经过 Sigmoid 约束在 [0, 1] 区间
        """
        h = np.maximum(0.0, np.dot(context_embeds, self.W1) + self.b1)
        logits = np.dot(h, self.W2) + self.b2
        # Sigmoid 归一化为像素灰度 [0, 1]
        pred_pixels = 1.0 / (1.0 + np.exp(-np.clip(logits, -15.0, 15.0)))
        return pred_pixels

    def reconstruction_loss(self, pred_pixels: np.ndarray, true_pixels: np.ndarray) -> float:
        """计算预测帧与真实未来帧的像素级均方误差 (MSE)"""
        return float(np.mean((pred_pixels - true_pixels) ** 2))


class DiffusionScheduler:
    """
    扩散模型 (Diffusion Model) 前向加噪调度器 (DDPM / DiT 核心算法)。

    前向扩散加噪公式：
        $$q(x_t | x_0) = \\mathcal{N}\\left(x_t; \\sqrt{\\bar{\\alpha}_t} x_0, (1 - \\bar{\\alpha}_t) \\mathbf{I}\\right)$$
        $$x_t = \\sqrt{\\bar{\\alpha}_t} x_0 + \\sqrt{1 - \\bar{\\alpha}_t} \\epsilon, \\quad \\epsilon \\sim \\mathcal{N}(0, \\mathbf{I})$$

    支持：
        - `cosine`: 余弦调度 (Nichol & Dhariwal, 2021)，在小时间步 (T=20) 亦可保证末端信号完全衰减为纯高斯噪声
        - `linear`: 线性调度 (Ho et al., 2020)，适用于大时间步 (T=1000) 训练
    """

    def __init__(
        self,
        num_steps: int = 20,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        schedule_type: str = "cosine",
    ) -> None:
        self.num_steps = num_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.schedule_type = schedule_type

        if schedule_type == "cosine" or "余弦" in schedule_type:
            # Nichol & Dhariwal (2021) Cosine Schedule
            s = 0.008
            steps = np.arange(num_steps + 1, dtype=np.float64)
            f_t = np.cos(((steps / num_steps) + s) / (1.0 + s) * np.pi * 0.5) ** 2
            alphas_cumprod = f_t / f_t[0]
            # 裁剪防止数值下溢/溢出
            self.alphas_cumprod = np.clip(alphas_cumprod[1:], 1e-5, 0.9999)
            # 由 alpha_bar 反推 beta_t: alpha_bar_t / alpha_bar_{t-1} = 1 - beta_t
            alphas_cumprod_prev = np.concatenate([np.array([1.0]), self.alphas_cumprod[:-1]])
            self.betas = np.clip(1.0 - (self.alphas_cumprod / alphas_cumprod_prev), 1e-5, 0.999)
            self.alphas = 1.0 - self.betas
        else:
            # 经典线性方差表: beta_t
            self.betas = np.linspace(beta_start, beta_end, num_steps, dtype=np.float64)
            # alpha_t = 1 - beta_t
            self.alphas = 1.0 - self.betas
            # 累乘保留系数: alpha_bar_t = prod(alpha_1 ... alpha_t)
            self.alphas_cumprod = np.cumprod(self.alphas)

    def add_noise(
        self,
        x_0: np.ndarray,
        t: int,
        noise: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        在时间步 t 对清晰图像 x_0 进行单步闭式加噪。
        t ∈ [0, num_steps - 1]
        """
        if t < 0:
            return x_0
        t_idx = min(t, self.num_steps - 1)
        alpha_bar = self.alphas_cumprod[t_idx]

        if noise is None:
            noise = np.random.randn(*x_0.shape)

        # 闭式采样公式: sqrt(alpha_bar) * x_0 + sqrt(1 - alpha_bar) * epsilon
        sqrt_alpha_bar = np.sqrt(alpha_bar)
        sqrt_one_minus_alpha_bar = np.sqrt(1.0 - alpha_bar)

        noisy_sample = sqrt_alpha_bar * x_0 + sqrt_one_minus_alpha_bar * noise
        return noisy_sample.astype(np.float32)

    def get_schedule(self) -> dict[str, np.ndarray | str]:
        """返回完整的调度参数曲线"""
        return {
            "schedule_type": self.schedule_type,
            "steps": np.arange(self.num_steps),
            "betas": self.betas,
            "alphas": self.alphas,
            "alphas_cumprod": self.alphas_cumprod,
        }


def visualize_diffusion_process(
    x_0: np.ndarray,
    scheduler: DiffusionScheduler,
    steps_to_show: int = 5
) -> list[Tuple[int, np.ndarray]]:
    """
    生成从清晰原图到纯高斯白噪声的等间隔扩散演化快照。
    """
    np.random.seed(42)
    fixed_noise = np.random.randn(*x_0.shape)
    indices = np.linspace(0, scheduler.num_steps - 1, steps_to_show, dtype=int)
    snapshots = []

    for t in indices:
        x_t = scheduler.add_noise(x_0, t, noise=fixed_noise)
        snapshots.append((int(t), x_t))

    return snapshots
