# Copyright (c) 2026 Yy1 (yuan278501381) | MIT License
"""
nn_core.reinforcement - 纯 NumPy 经典与现代强化学习核心算法引擎

涵盖从经典 MDP 到 2026 前沿推理强化学习的完整算法阶梯：
1. GridWorldEnv: 经典网格马尔可夫决策环境 (MDP)
2. BellmanSolver: 贝尔曼最优方程与值迭代解析求解器 (Value Iteration)
3. QLearningAgent: 时序差分 Q-Learning 智能体 (TD Learning & ε-Greedy)
4. PolicyGradientAgent: REINFORCE 策略梯度网络 (Policy Gradients)
5. GRPORunner: 组内优势计算与规则曲线教学模拟（不是语言模型 GRPO 训练器）
"""

from typing import Any

import numpy as np


class GridWorldEnv:
    """
    经典离散二维网格马尔可夫决策过程 (MDP) 环境。

    网格代号定义：
    - 0: 空白通路 (Free space, step reward = -0.04 或 step_cost)
    - 1: 障碍物 (Wall/Obstacle, 无法通行)
    - 2: 危险陷阱 (Trap/Cliff, reward = -1.0, 终止)
    - 3: 目标终点 (Goal/Treasure, reward = +1.0, 终止)

    动作空间：
    0: 上 (Up), 1: 右 (Right), 2: 下 (Down), 3: 左 (Left)
    """

    ACTIONS = ((-1, 0), (0, 1), (1, 0), (0, -1))  # 上, 右, 下, 左
    ACTION_NAMES = ("↑ 上", "→ 右", "↓ 下", "← 左")

    def __init__(
        self,
        grid_type: str = "cliff",
        step_cost: float = -0.04,
        goal_reward: float = 1.0,
        trap_reward: float = -1.0,
    ) -> None:
        if grid_type not in {"cliff", "maze", "simple"}:
            raise ValueError("grid_type 必须是 cliff、maze 或 simple")
        self.step_cost = step_cost
        self.goal_reward = goal_reward
        self.trap_reward = trap_reward
        self.grid_type = grid_type
        self.grid, self.start_pos, self.goal_pos = self._build_grid(grid_type)
        self.height, self.width = self.grid.shape
        self.agent_pos = list(self.start_pos)
        self.done = False

    def _build_grid(self, grid_type: str) -> tuple[np.ndarray, tuple[int, int], tuple[int, int]]:
        """构建预设网格布局"""
        if grid_type == "cliff":
            # 4x6 悬崖漫步环境 (Cliff Walking)
            # 起点左下角 (3, 0)，终点右下角 (3, 5)，中间整条底边为悬崖陷阱
            grid = np.zeros((4, 6), dtype=int)
            grid[3, 1:5] = 2  # 悬崖
            grid[3, 5] = 3  # 宝藏
            return grid, (3, 0), (3, 5)
        elif grid_type == "maze":
            # 5x5 迷宫障碍环境
            grid = np.zeros((5, 5), dtype=int)
            grid[1, 1:4] = 1  # 墙壁
            grid[3, 0:3] = 1  # 墙壁
            grid[3, 3] = 2  # 陷阱
            grid[4, 4] = 3  # 宝藏
            return grid, (0, 0), (4, 4)
        else:  # simple
            # 4x4 极简网格世界
            grid = np.zeros((4, 4), dtype=int)
            grid[1, 1] = 1  # 障碍
            grid[1, 3] = 2  # 陷阱
            grid[3, 3] = 3  # 宝藏
            return grid, (0, 0), (3, 3)

    def reset(self) -> tuple[int, int]:
        """重置环境到初始起点"""
        self.agent_pos = list(self.start_pos)
        self.done = False
        return (int(self.agent_pos[0]), int(self.agent_pos[1]))

    def step(self, action: int) -> tuple[tuple[int, int], float, bool, dict[str, Any]]:
        """执行一个动作并转移状态"""
        if not 0 <= action < len(self.ACTIONS):
            raise ValueError(f"action 必须位于 [0, {len(self.ACTIONS) - 1}]")
        if self.done:
            state = (int(self.agent_pos[0]), int(self.agent_pos[1]))
            return state, 0.0, True, {"status": "already_done"}

        dr, dc = self.ACTIONS[action]
        nr = self.agent_pos[0] + dr
        nc = self.agent_pos[1] + dc

        # 越界或撞墙保护
        if nr < 0 or nr >= self.height or nc < 0 or nc >= self.width or self.grid[nr, nc] == 1:
            # 撞墙停留在原地
            nr, nc = self.agent_pos[0], self.agent_pos[1]

        self.agent_pos = [nr, nc]
        cell_type = self.grid[nr, nc]

        if cell_type == 3:
            # 抵达宝藏目标
            self.done = True
            reward = self.goal_reward
            info = {"event": "goal_reached"}
        elif cell_type == 2:
            # 跌落悬崖/陷阱
            self.done = True
            reward = self.trap_reward
            info = {"event": "fell_in_trap"}
        else:
            # 普通空地
            reward = self.step_cost
            info = {"event": "step"}

        state = (int(self.agent_pos[0]), int(self.agent_pos[1]))
        return state, reward, self.done, info


class BellmanSolver:
    """
    贝尔曼最优方程解析求解器 (Value Iteration 动态规划)。

    最优状态价值方程：
    V*(s) = max_a [ R(s, a) + γ * Σ P(s'|s, a) V*(s') ]
    """

    def __init__(self, env: GridWorldEnv, gamma: float = 0.95, theta: float = 1e-4) -> None:
        self.env = env
        self.gamma = gamma
        self.theta = theta
        self.H, self.W = env.height, env.width
        self.V = np.zeros((self.H, self.W), dtype=float)
        self.policy = np.zeros((self.H, self.W), dtype=int)

    def solve(self, max_iterations: int = 500) -> tuple[np.ndarray, np.ndarray, int]:
        """执行值迭代直至收敛，返回 (最优价值矩阵 V*, 最优策略矩阵 Policy*, 迭代轮数)"""
        for iteration in range(1, max_iterations + 1):
            delta = 0.0
            new_V = np.copy(self.V)

            for r in range(self.H):
                for c in range(self.W):
                    if self.env.grid[r, c] in {1, 2, 3}:
                        # 墙壁、陷阱和终点为终止/不可用状态
                        continue

                    # 计算 4 个动作的期望动作价值 Q(s, a)
                    q_values = []
                    for _action_idx, (dr, dc) in enumerate(self.env.ACTIONS):
                        nr, nc = r + dr, c + dc
                        if (
                            nr < 0
                            or nr >= self.H
                            or nc < 0
                            or nc >= self.W
                            or self.env.grid[nr, nc] == 1
                        ):
                            nr, nc = r, c

                        dest_type = self.env.grid[nr, nc]
                        if dest_type == 3:
                            reward = self.env.goal_reward
                            next_val = 0.0  # 终止态价值为 0
                        elif dest_type == 2:
                            reward = self.env.trap_reward
                            next_val = 0.0
                        else:
                            reward = self.env.step_cost
                            next_val = self.V[nr, nc]

                        q_sa = reward + self.gamma * next_val
                        q_values.append(q_sa)

                    best_q = max(q_values)
                    delta = max(delta, abs(self.V[r, c] - best_q))
                    new_V[r, c] = best_q
                    self.policy[r, c] = int(np.argmax(q_values))

            self.V = new_V
            if delta < self.theta:
                return self.V, self.policy, iteration

        return self.V, self.policy, max_iterations

    def bellman_residual(self, values: np.ndarray | None = None) -> float:
        """返回当前价值函数相对贝尔曼最优算子的最大绝对残差。"""

        source = self.V if values is None else np.asarray(values, dtype=float)
        if source.shape != (self.H, self.W):
            raise ValueError("values 形状必须与环境网格一致")
        residual = 0.0
        for r in range(self.H):
            for c in range(self.W):
                if self.env.grid[r, c] in {1, 2, 3}:
                    continue
                q_values = []
                for dr, dc in self.env.ACTIONS:
                    nr, nc = r + dr, c + dc
                    if (
                        nr < 0
                        or nr >= self.H
                        or nc < 0
                        or nc >= self.W
                        or self.env.grid[nr, nc] == 1
                    ):
                        nr, nc = r, c
                    dest_type = self.env.grid[nr, nc]
                    if dest_type == 3:
                        q_values.append(self.env.goal_reward)
                    elif dest_type == 2:
                        q_values.append(self.env.trap_reward)
                    else:
                        q_values.append(self.env.step_cost + self.gamma * source[nr, nc])
                residual = max(residual, abs(source[r, c] - max(q_values)))
        return float(residual)


class QLearningAgent:
    """
    时序差分 Q-Learning 强化学习智能体。

    更新公式：
    Q(s, a) ← Q(s, a) + α * [ r + γ * max_a' Q(s', a') - Q(s, a) ]
    """

    def __init__(
        self,
        height: int,
        width: int,
        n_actions: int = 4,
        lr: float = 0.1,
        gamma: float = 0.95,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        min_epsilon: float = 0.05,
        seed: int | None = 42,
        rng: np.random.Generator | None = None,
    ) -> None:
        if height <= 0 or width <= 0 or n_actions <= 0:
            raise ValueError("状态与动作维度必须为正整数")
        if not 0 < lr <= 1 or not 0 <= gamma <= 1:
            raise ValueError("lr 必须位于 (0,1]，gamma 必须位于 [0,1]")
        if not 0 <= epsilon <= 1 or not 0 <= min_epsilon <= 1 or not 0 < epsilon_decay <= 1:
            raise ValueError("epsilon、min_epsilon 或 epsilon_decay 配置无效")
        self.H = height
        self.W = width
        self.n_actions = n_actions
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.rng = rng if rng is not None else np.random.default_rng(seed)

        # Q-Table 形状: (Height, Width, Actions)
        self.q_table = np.zeros((height, width, n_actions), dtype=float)

    def select_action(self, state: tuple[int, int]) -> int:
        """ε-greedy 探索与利用动作选择"""
        r, c = state
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_actions))
        return int(np.argmax(self.q_table[r, c]))

    def update(
        self,
        state: tuple[int, int],
        action: int,
        reward: float,
        next_state: tuple[int, int],
        done: bool,
    ) -> float:
        """执行单步 Q 学习时序差分更新并返回 TD-Error"""
        r, c = state
        nr, nc = next_state
        current_q = self.q_table[r, c, action]
        target = reward if done else reward + self.gamma * np.max(self.q_table[nr, nc])
        td_error = target - current_q
        self.q_table[r, c, action] += self.lr * td_error
        return float(abs(td_error))

    def decay_epsilon(self) -> None:
        """衰减探索率 ε"""
        if self.epsilon > self.min_epsilon:
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def train_episodes(
        self, env: GridWorldEnv, n_episodes: int = 200, max_steps: int = 100
    ) -> dict[str, list[float]]:
        """训练指定轮数并返回训练历史指标"""
        returns = []
        steps_list = []
        td_errors = []

        for _ in range(n_episodes):
            state = env.reset()
            episode_return = 0.0
            episode_td = []
            steps = 0

            for _ in range(max_steps):
                action = self.select_action(state)
                next_state, reward, done, _ = env.step(action)
                err = self.update(state, action, reward, next_state, done)
                episode_td.append(err)
                episode_return += reward
                state = next_state
                steps += 1
                if done:
                    break

            self.decay_epsilon()
            returns.append(float(episode_return))
            steps_list.append(float(steps))
            td_errors.append(float(np.mean(episode_td)) if episode_td else 0.0)

        return {
            "returns": returns,
            "steps": steps_list,
            "td_errors": td_errors,
        }

    def greedy_rollout(self, env: GridWorldEnv, max_steps: int = 100) -> dict[str, Any]:
        """执行当前 Q 表的贪心策略，并显式报告成功、陷阱、循环或截断。"""

        if max_steps <= 0:
            raise ValueError("max_steps 必须为正整数")
        state = env.reset()
        path = [state]
        seen = {state}
        total_return = 0.0
        looped = False
        event = "truncated"
        for _ in range(max_steps):
            action = int(np.argmax(self.q_table[state[0], state[1]]))
            next_state, reward, done, info = env.step(action)
            path.append(next_state)
            total_return += reward
            event = str(info.get("event", info.get("status", "unknown")))
            if done:
                break
            if next_state in seen:
                looped = True
                event = "loop_detected"
                break
            seen.add(next_state)
            state = next_state
        return {
            "path": path,
            "return": float(total_return),
            "reached_goal": event == "goal_reached",
            "looped": looped,
            "event": event,
        }


class PolicyGradientAgent:
    """
    REINFORCE 策略梯度智能体 (基于 Softmax 动作分布与对数似然梯度)。

    目标函数：J(θ) = E_τ [ Σ_t R(s_t, a_t) ]
    梯度更新：θ ← θ + α * Σ_t ∇_θ log π_θ(a_t | s_t) * G_t
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int = 4,
        lr: float = 0.05,
        gamma: float = 0.95,
        seed: int | None = 42,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr
        self.gamma = gamma

        # 线性策略参数 θ: (state_dim, action_dim)
        self.rng = rng if rng is not None else np.random.default_rng(seed)
        self.weights = self.rng.normal(size=(state_dim, action_dim)) * 0.1

    def get_action_probs(self, state_vec: np.ndarray) -> np.ndarray:
        """计算 Softmax 动作概率分布"""
        logits = state_vec @ self.weights
        exp_l = np.exp(logits - np.max(logits))
        probs = exp_l / (np.sum(exp_l) + 1e-12)
        return probs

    def select_action(self, state_vec: np.ndarray) -> int:
        """根据概率分布采样动作"""
        probs = self.get_action_probs(state_vec)
        return int(self.rng.choice(self.action_dim, p=probs))

    def compute_discounted_returns(self, rewards: list[float]) -> np.ndarray:
        """计算折扣累积回报 G_t"""
        G = np.zeros(len(rewards), dtype=float)
        running_g = 0.0
        for t in reversed(range(len(rewards))):
            running_g = rewards[t] + self.gamma * running_g
            G[t] = running_g
        # 标准化基线
        if len(G) > 1 and np.std(G) > 1e-8:
            G = (G - np.mean(G)) / (np.std(G) + 1e-8)
        return G

    def update(
        self,
        states: list[np.ndarray],
        actions: list[int],
        rewards: list[float],
    ) -> float:
        """基于整条轨迹进行一次策略梯度更新"""
        if not states:
            return 0.0

        returns = self.compute_discounted_returns(rewards)
        total_grad = np.zeros_like(self.weights)

        for s, a, g in zip(states, actions, returns, strict=True):
            probs = self.get_action_probs(s)
            # ∇_θ log π_θ(a|s) = s^T * (one_hot(a) - probs)
            d_logits = -probs
            d_logits[a] += 1.0
            grad_step = np.outer(s, d_logits) * g
            total_grad += grad_step

        self.weights += self.lr * total_grad
        return float(np.mean(np.abs(total_grad)))


class GRPORunner:
    """
    组内相对优势计算与 DeepSeek-R1 相关现象的规则曲线教学模拟。

    本类没有语言模型、策略比率、裁剪目标、KL 项或参数更新，因此不是完整
    GRPO 优化器。`simulate_r1_reasoning_evolution` 的输出是合成情景，不是训练日志。

    核心机制：
    1. 对同一提示词 Prompt 采样一组回答 {y_1, y_2, ..., y_G}。
    2. 基于确定性可验证规则（如数学题正确性、代码单测、XML 格式）计算规则奖励 {r_1, ..., r_G}。
    3. 组内相对归一化优势计算：A_i = (r_i - mean(R)) / (std(R) + ε)。
    4. 无需训练复杂的 Critic 价值网络，驱动模型自发涌现“长思考链、反思、自我纠错（Aha Moment）”。
    """

    @staticmethod
    def compute_group_advantages(rewards: list[float], eps: float = 1e-6) -> np.ndarray:
        """计算 GRPO 组内相对优势 A_i"""
        r_arr = np.array(rewards, dtype=float)
        mean_r = np.mean(r_arr)
        std_r = np.std(r_arr)
        if std_r < eps:
            return np.zeros_like(r_arr)
        return (r_arr - mean_r) / (std_r + eps)

    @staticmethod
    def simulate_r1_reasoning_evolution(
        n_iterations: int = 15, seed: int | None = 42
    ) -> dict[str, Any]:
        """
        生成用于讲解的规则曲线与模板案例；不是 GRPO 训练，也不能证明能力涌现。
        """
        if n_iterations <= 0:
            raise ValueError("n_iterations 必须为正整数")
        rng = np.random.default_rng(seed)
        iters = list(range(1, n_iterations + 1))

        # 思考链 Token 长度自发暴涨曲线 (从 80 tokens 到 1200 tokens)
        cot_lengths = [int(75 + 70 * (i**1.1) + rng.integers(-15, 20)) for i in iters]

        # 任务准确率突破曲线 (从 20% 到 94%)
        accuracies = [
            min(0.96, float(0.18 + 0.78 / (1.0 + np.exp(-0.45 * (i - 6))))) for i in iters
        ]

        # 顿悟反思标记 (Aha Moment: 如 "Wait, let me double check...") 的出现频次
        aha_counts = [max(0, int(0.3 * (i - 4) ** 1.3)) if i >= 4 else 0 for i in iters]

        sample_cases = [
            {
                "step": "Early RL (Step 1-3)",
                "cot": "To solve 3x + 7 = 22, I subtract 7 to get 15, then divide by 3. The answer is 5.",
                "type": "Short Straightforward // 短链直接输出",
                "length": 85,
                "accuracy": "22%",
            },
            {
                "step": "Mid RL (Step 6-8)",
                "cot": "Let me re-verify. If 3x + 7 = 22, then 3x = 15. Let me plug 5 back: 3(5)+7=22. Checked.",
                "type": "Self-Verification // 开始出现自我验证",
                "length": 340,
                "accuracy": "68%",
            },
            {
                "step": "Late RL (Step 12-15 · DeepSeek-R1 Aha Moment)",
                "cot": "<think>\nLet's analyze the problem carefully. Suppose x is negative... Wait! That violates x > 0. Let me restart.\nLet me try contradiction: if x=5, 3(5)+7=22 holds. What about boundary conditions? None. Aha! The solution is strictly unique.\n</think>\nFinal Answer: 5",
                "type": "Emergent Self-Correction // 自发长思维链与顿悟纠错",
                "length": 1150,
                "accuracy": "94%",
            },
        ]

        return {
            "iterations": iters,
            "cot_lengths": cot_lengths,
            "accuracies": accuracies,
            "aha_counts": aha_counts,
            "cases": sample_cases,
        }
