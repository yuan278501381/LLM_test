# Copyright (c) 2026 Yy1 (yuan278501381) | MIT License
"""
tests/test_evaluation.py - 评估基准框架 (Harness) 与指标单元测试
"""

import numpy as np
import pytest

from nn_core.evaluation import (
    BenchmarkQuestion,
    BenchmarkTask,
    EvaluationHarness,
    compute_accuracy,
    compute_f1,
    compute_perplexity,
    get_mini_gsm8k,
    get_mini_hellaswag,
    get_mini_mmlu,
    get_mini_safety,
)


def test_compute_perplexity():
    """测试困惑度 PPL 计算逻辑"""
    # 均匀分布 100 分类 log_probs = -log(100) ≈ -4.605
    vocab_size = 100
    uniform_log_p = np.full((50,), -np.log(vocab_size))
    ppl = compute_perplexity(uniform_log_p)
    assert ppl == pytest.approx(100.0, rel=1e-3)

    # 确定性分布 log_prob = 0.0 -> PPL = 1.0 (完美预测)
    perfect_log_p = np.zeros(20)
    assert compute_perplexity(perfect_log_p) == pytest.approx(1.0, rel=1e-5)


def test_compute_accuracy_and_f1():
    """测试准确率与 F1 指标边界"""
    p = [0, 1, 2, 3]
    y = [0, 1, 2, 3]
    assert compute_accuracy(p, y) == 1.0
    assert compute_f1(p, y) == 1.0

    p_bad = [1, 0, 3, 2]
    assert compute_accuracy(p_bad, y) == 0.0

    p_half = [0, 1, 0, 0]
    assert compute_accuracy(p_half, y) == 0.5


def test_benchmark_tasks_structure():
    """测试四大预置题库题量与答案索引范围"""
    tasks = [get_mini_mmlu(), get_mini_hellaswag(), get_mini_gsm8k(), get_mini_safety()]
    for t in tasks:
        assert len(t.questions) >= 8
        for q in t.questions:
            assert len(q.choices) == 4
            assert 0 <= q.answer_idx <= 3
            assert len(q.question) > 0
            assert len(q.category) > 0


def test_evaluation_harness_execution():
    """测试 EvaluationHarness 自动化调度与雷达数据生成"""
    task = get_mini_mmlu()
    harness = EvaluationHarness(tasks=[task])

    # 模拟 100% 答对
    def perfect_model(question: str, choices: list[str]) -> int:
        for q in task.questions:
            if q.question == question:
                return q.answer_idx
        return 0

    scores = harness.run_all(perfect_model)
    assert scores[task.name] == 100.0

    radar_data = EvaluationHarness.generate_radar_data(scores)
    assert len(radar_data["theta"]) == 2  # 1 个任务 + 1 个闭环点
    assert radar_data["r"][0] == 100.0
