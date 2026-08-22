# Copyright (c) 2026 Yy1 (yuan278501381) | MIT License
"""
tests.test_reinforcement - 强化学习核心算法引擎单元测试集

覆盖：
1. GridWorldEnv 环境动力学与边界保护
2. BellmanSolver 贝尔曼值迭代收敛性与最优策略检验
3. QLearningAgent 时序差分更新、ε-greedy 探索与迷宫训练
4. PolicyGradientAgent 策略梯度计算与折扣回报
5. GRPORunner 组相对优势归一化与 DeepSeek-R1 式思维链演进模拟
"""

import numpy as np
import pytest

from nn_core.reinforcement import (
    BellmanSolver,
    GridWorldEnv,
    GRPORunner,
    PolicyGradientAgent,
    QLearningAgent,
)


class TestGridWorldEnv:
    """测试离散网格环境动力学"""

    def test_env_initialization_and_reset(self):
        env = GridWorldEnv(grid_type="cliff")
        assert env.height == 4
        assert env.width == 6
        assert env.start_pos == (3, 0)
        assert env.goal_pos == (3, 5)

        state = env.reset()
        assert state == (3, 0)
        assert not env.done

    def test_env_wall_and_boundary_collision(self):
        env = GridWorldEnv(grid_type="maze")
        env.reset()
        # 起点 (0, 0)，向左撞边界
        s_next, r, done, _info = env.step(3)  # Left
        assert s_next == (0, 0)
        assert r == pytest.approx(-0.04)
        assert not done

    def test_env_goal_and_trap_termination(self):
        env = GridWorldEnv(grid_type="cliff")
        env.reset()
        # 从 (3, 0) 向右一步跌落悬崖 (3, 1)
        s_next, r, done, info = env.step(1)  # Right
        assert s_next == (3, 1)
        assert r == pytest.approx(-1.0)
        assert done
        assert info["event"] == "fell_in_trap"

    def test_invalid_environment_and_action_are_rejected(self):
        with pytest.raises(ValueError, match="grid_type"):
            GridWorldEnv(grid_type="typo")
        env = GridWorldEnv(grid_type="simple")
        with pytest.raises(ValueError, match="action"):
            env.step(4)


class TestBellmanSolver:
    """测试贝尔曼值迭代求解器"""

    def test_bellman_value_iteration_convergence(self):
        env = GridWorldEnv(grid_type="cliff")
        solver = BellmanSolver(env=env, gamma=0.9, theta=1e-4)
        V, policy, iters = solver.solve(max_iterations=200)

        assert iters < 200
        assert V.shape == (4, 6)
        assert policy.shape == (4, 6)
        # 靠近终点的格子价值必然高于起点的价值
        assert V[2, 5] > V[3, 0]
        assert solver.bellman_residual(V) < 1e-4


class TestQLearningAgent:
    """测试 Q-Learning 时序差分智能体"""

    def test_q_table_update(self):
        agent = QLearningAgent(height=4, width=6, lr=0.5, gamma=0.9, epsilon=0.0)
        # 单步更新测试
        state = (2, 4)
        next_state = (3, 5)  # 宝藏
        err = agent.update(state, action=2, reward=1.0, next_state=next_state, done=True)
        assert err > 0.0
        assert agent.q_table[2, 4, 2] == pytest.approx(0.5)

    def test_agent_training_loop(self):
        env = GridWorldEnv(grid_type="cliff")
        agent = QLearningAgent(height=4, width=6, lr=0.2, gamma=0.9, epsilon=0.5)
        history = agent.train_episodes(env, n_episodes=50, max_steps=40)

        assert len(history["returns"]) == 50
        assert len(history["steps"]) == 50
        assert len(history["td_errors"]) == 50

    def test_greedy_rollout_reports_failure_instead_of_calling_it_optimal(self):
        env = GridWorldEnv(grid_type="simple")
        agent = QLearningAgent(height=env.height, width=env.width, epsilon=0.0)
        rollout = agent.greedy_rollout(env, max_steps=10)
        assert not rollout["reached_goal"]
        assert rollout["looped"]
        assert rollout["event"] == "loop_detected"

    def test_q_learning_reaches_goal_across_most_fixed_seeds(self):
        successes = 0
        gaps = []
        env_reference = GridWorldEnv(grid_type="simple")
        solver = BellmanSolver(env_reference, gamma=0.95, theta=1e-8)
        optimal_values, _, _ = solver.solve()
        optimal_start_value = optimal_values[env_reference.start_pos]

        for seed in range(8):
            env = GridWorldEnv(grid_type="simple")
            agent = QLearningAgent(
                height=env.height,
                width=env.width,
                lr=0.3,
                gamma=0.95,
                epsilon=1.0,
                epsilon_decay=0.98,
                min_epsilon=0.05,
                seed=seed,
            )
            agent.train_episodes(env, n_episodes=400, max_steps=60)
            rollout = agent.greedy_rollout(env, max_steps=30)
            successes += int(rollout["reached_goal"])
            gaps.append(abs(agent.q_table[env.start_pos].max() - optimal_start_value))

        assert successes >= 7
        assert float(np.median(gaps)) < 0.1


class TestPolicyGradientAgent:
    """测试 REINFORCE 策略梯度智能体"""

    def test_action_probabilities(self):
        agent = PolicyGradientAgent(state_dim=8, action_dim=4)
        state_vec = np.ones(8)
        probs = agent.get_action_probs(state_vec)
        assert probs.shape == (4,)
        assert np.sum(probs) == pytest.approx(1.0)
        assert np.all(probs >= 0.0)

    def test_policy_update(self):
        agent = PolicyGradientAgent(state_dim=8, action_dim=4, lr=0.1)
        states = [np.ones(8), np.ones(8) * 0.5]
        actions = [1, 2]
        rewards = [0.0, 1.0]
        grad_norm = agent.update(states, actions, rewards)
        assert grad_norm >= 0.0


class TestGRPORunner:
    """测试 DeepSeek-R1 式 GRPO 算法引擎"""

    def test_group_advantage_normalization(self):
        rewards = [1.0, 0.0, 0.0, 1.0]
        adv = GRPORunner.compute_group_advantages(rewards)
        assert adv.shape == (4,)
        assert np.mean(adv) == pytest.approx(0.0, abs=1e-5)
        assert adv[0] > 0.0
        assert adv[1] < 0.0

    def test_r1_reasoning_simulation(self):
        res = GRPORunner.simulate_r1_reasoning_evolution(n_iterations=10)
        assert len(res["iterations"]) == 10
        assert len(res["cot_lengths"]) == 10
        assert len(res["accuracies"]) == 10
        assert len(res["aha_counts"]) == 10
        assert len(res["cases"]) == 3
        # 思考链 Token 长度应显著随轮次增长
        assert res["cot_lengths"][-1] > res["cot_lengths"][0]
        assert GRPORunner.simulate_r1_reasoning_evolution(10, seed=7) == (
            GRPORunner.simulate_r1_reasoning_evolution(10, seed=7)
        )

    def test_r1_simulation_rejects_nonpositive_length(self):
        with pytest.raises(ValueError):
            GRPORunner.simulate_r1_reasoning_evolution(0)
