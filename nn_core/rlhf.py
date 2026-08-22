# Copyright (c) 2026 Yy1 (yuan278501381) | MIT License
"""
nn_core.rlhf - 人类反馈强化学习 (RLHF) 奖励模型、PPO-Clip 策略梯度与 DPO 直接偏好优化

包含：
- `RewardModel`: 基于 Bradley-Terry 偏好概率建模的标量奖励模型
- `PPOClipObjective`: 近端策略优化 PPO-Clip 目标函数与训练轨迹模拟
- `DPOLoss`: 直接偏好优化 (Direct Preference Optimization) 隐式奖励闭式损失
"""

import logging

import numpy as np

logger = logging.getLogger("nn_core.rlhf")


class RewardModel:
    """
    人类偏好奖励模型 (Reward Model)。

    基于 Bradley-Terry 概率模型：
        $$P(y_w \\succ y_l | x) = \\sigma(r_\\theta(x, y_w) - r_\\theta(x, y_l))$$
        $$L_{RM} = -\\log \\sigma(r(x, y_w) - r(x, y_l))$$
    """

    def __init__(self, input_dim: int = 32, hidden_dim: int = 64) -> None:
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, 1) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros(1)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        输入样本表征 x: (B, input_dim)
        输出标量奖励得分: (B, 1)
        """
        h = np.maximum(0.0, np.dot(x, self.W1) + self.b1)
        score = np.dot(h, self.W2) + self.b2
        return score

    @staticmethod
    def preference_loss(
        reward_chosen: float | np.ndarray, reward_rejected: float | np.ndarray
    ) -> float:
        """计算 Bradley-Terry 对比偏好损失"""
        diff = np.asarray(reward_chosen) - np.asarray(reward_rejected)
        # 避免溢出: log(1 + exp(-diff))
        loss = np.log(1.0 + np.exp(-np.clip(diff, -20.0, 20.0)))
        return float(np.mean(loss))


class PPOClipObjective:
    """
    近端策略优化 (PPO-Clip) 核心目标函数。

    目标方程：
        $$L^{CLIP}(\\theta) = \\hat{\\mathbb{E}}_t \\left[ \\min(r_t(\\theta) \\hat{A}_t, \\text{clip}(r_t(\\theta), 1-\\epsilon, 1+\\epsilon) \\hat{A}_t) \\right]$$
    """

    def __init__(self, epsilon: float = 0.2) -> None:
        self.epsilon = epsilon

    @staticmethod
    def compute_ratio(log_probs_new: np.ndarray, log_probs_old: np.ndarray) -> np.ndarray:
        """计算重要性采样概率比率: $r_t = \\exp(\\log \\pi_{new} - \\log \\pi_{old})$"""
        return np.exp(np.clip(log_probs_new - log_probs_old, -10.0, 10.0))

    def clip_objective(
        self, ratio: np.ndarray, advantages: np.ndarray, epsilon: float | None = None
    ) -> np.ndarray:
        """计算带截断保护的策略梯度目标"""
        eps = self.epsilon if epsilon is None else epsilon
        surr1 = ratio * advantages
        surr2 = np.clip(ratio, 1.0 - eps, 1.0 + eps) * advantages
        # 强化学习最大化目标，等价于最小化负目标
        return np.minimum(surr1, surr2)

    @staticmethod
    def simulate_rlhf_trajectory(n_steps: int = 20, seed: int = 42) -> dict[str, list[float]]:
        """生成预设规则曲线；不是 RLHF 训练或日志。"""
        if n_steps <= 0:
            raise ValueError("n_steps 必须为正数")
        rng = np.random.default_rng(seed)
        steps = [float(step) for step in range(n_steps)]

        # 奖励逐步上升并平稳收敛
        t = np.linspace(0, 1, n_steps)
        rewards = (1.2 / (1.0 + np.exp(-6 * (t - 0.3))) + rng.normal(0.0, 0.03, n_steps)).tolist()

        # KL 散度被 KL 控制器约束在适度范围
        kl_divs = (0.05 + 0.3 * (1.0 - np.exp(-3 * t)) + rng.normal(0.0, 0.015, n_steps)).tolist()

        # 策略截断触发比例 (维持在 10%~20%)
        clip_fracs = (0.12 + 0.08 * np.sin(np.pi * t) + rng.normal(0.0, 0.01, n_steps)).tolist()

        return {
            "step": steps,
            "reward": rewards,
            "kl_div": kl_divs,
            "clip_fraction": clip_fracs,
        }


class DPOLoss:
    """
    直接偏好优化 (Direct Preference Optimization / DPO)。

    数学原理：
        通过数学代换，跳过显式奖励模型训练与强化学习 PPO 循环，直接在策略模型上优化偏好对：
        $$L_{DPO}(\\pi_\\theta; \\pi_{ref}) = -\\mathbb{E}_{(x, y_w, y_l)} \\left[ \\log \\sigma \\left( \\beta \\log \\frac{\\pi_\\theta(y_w|x)}{\\pi_{ref}(y_w|x)} - \\beta \\log \\frac{\\pi_\\theta(y_l|x)}{\\pi_{ref}(y_l|x)} \\right) \\right]$$
    """

    def __init__(self, beta: float = 0.1) -> None:
        self.beta = beta

    def forward(
        self,
        pi_logprobs_w: float | np.ndarray,
        pi_logprobs_l: float | np.ndarray,
        ref_logprobs_w: float | np.ndarray,
        ref_logprobs_l: float | np.ndarray,
    ) -> float:
        """计算 DPO 隐式偏好损失"""
        # 隐式奖励差：beta * ( (pi_w - ref_w) - (pi_l - ref_l) )
        implicit_reward_chosen = self.beta * (
            np.asarray(pi_logprobs_w) - np.asarray(ref_logprobs_w)
        )
        implicit_reward_rejected = self.beta * (
            np.asarray(pi_logprobs_l) - np.asarray(ref_logprobs_l)
        )

        diff = implicit_reward_chosen - implicit_reward_rejected
        loss = np.log(1.0 + np.exp(-np.clip(diff, -20.0, 20.0)))
        return float(np.mean(loss))
