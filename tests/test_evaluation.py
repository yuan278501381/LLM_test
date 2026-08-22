# Copyright (c) 2026 Yy1 (yuan278501381) | MIT License
"""
tests/test_evaluation.py - 评估基准框架 (Harness) 与指标单元测试
"""

import numpy as np
import pytest
from sklearn.metrics import accuracy_score, f1_score

from nn_core.evaluation import (
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

    masked = np.array([-np.log(2.0), -np.log(8.0), -np.log(4.0)])
    assert compute_perplexity(masked, mask=np.array([True, False, True])) == pytest.approx(
        np.sqrt(8.0)
    )


@pytest.mark.parametrize(
    "values,mask",
    [
        (np.array([]), None),
        (np.array([np.nan]), None),
        (np.array([0.1]), None),
        (np.array([-1.0]), np.array([False])),
        (np.array([-1.0]), np.array([True, False])),
    ],
)
def test_compute_perplexity_rejects_invalid_contract(values, mask):
    with pytest.raises(ValueError):
        compute_perplexity(values, mask=mask)


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

    with pytest.raises(ValueError):
        compute_accuracy([0], [0, 1])
    with pytest.raises(ValueError):
        compute_f1(np.array([[0, 1]]), np.array([[0, 1]]))


@pytest.mark.parametrize(
    "labels,predictions",
    [
        ([0, 0, 1, 1], [0, 1, 1, 1]),
        ([0, 0, 2, 2], [0, 1, 2, 1]),
        ([2, 2, 2], [2, 1, 1]),
    ],
)
def test_accuracy_and_macro_f1_match_sklearn(labels, predictions):
    assert compute_accuracy(predictions, labels) == pytest.approx(
        accuracy_score(labels, predictions)
    )
    all_labels = np.unique(np.concatenate([labels, predictions]))
    assert compute_f1(predictions, labels) == pytest.approx(
        f1_score(labels, predictions, labels=all_labels, average="macro", zero_division=0)
    )


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


def test_evaluation_harness_honors_metric_and_rejects_unknown_metric():
    from nn_core.evaluation import BenchmarkQuestion, BenchmarkTask

    questions = [
        BenchmarkQuestion("q0", ["a", "b"], 0, "demo"),
        BenchmarkQuestion("q1", ["a", "b"], 0, "demo"),
        BenchmarkQuestion("q2", ["a", "b"], 1, "demo"),
    ]
    predictions = {"q0": 0, "q1": 1, "q2": 1}

    def predict(question, _choices):
        return predictions[question]

    f1_task = BenchmarkTask("f1-demo", "metric dispatch", questions, metric="f1")
    expected = compute_f1([0, 1, 1], [0, 0, 1]) * 100.0
    assert EvaluationHarness([f1_task]).run_task(f1_task, predict) == pytest.approx(expected)

    invalid = BenchmarkTask("bad", "invalid metric", questions, metric="auc")
    with pytest.raises(ValueError, match="未知评估指标"):
        EvaluationHarness([invalid]).run_task(invalid, predict)


def test_teaching_task_names_do_not_claim_canonical_dataset_membership():
    tasks = [get_mini_mmlu(), get_mini_hellaswag(), get_mini_gsm8k()]
    for task in tasks:
        assert "-style 教学题" in task.name
        assert "非正式" in task.description
