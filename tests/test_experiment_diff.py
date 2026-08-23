# Copyright (c) 2026 Yy1 (yuan278501381) | MIT License
"""
tests/test_experiment_diff.py - 双实验消融差分对比套件单元测试
"""

from dashboard.components.experiment_diff import (
    clear_baseline,
    get_baseline,
    save_baseline,
)


def test_experiment_diff_baseline_lifecycle():
    """验证基准实验的保存、读取与清除生命周期"""
    module_id = "test_mod_01"
    name = "Test Run A"
    params = {"lr": 0.01, "batch_size": 32}
    metrics = {"loss": 0.25, "accuracy": 0.92}
    history = [0.8, 0.5, 0.25]

    save_baseline(module_id, name, params, metrics, history)
    loaded = get_baseline(module_id)

    assert loaded is not None
    assert loaded["name"] == name
    assert loaded["params"]["lr"] == 0.01
    assert loaded["metrics"]["loss"] == 0.25
    assert len(loaded["loss_history"]) == 3

    clear_baseline(module_id)
    assert get_baseline(module_id) is None
