# Copyright (c) 2026 Yy1 (yuan278501381) | MIT License
"""训练回调的行为契约：快照隔离、早停状态机与可审计日志。"""

import json
from pathlib import Path

import numpy as np
import pytest

from nn_core.callbacks import EarlyStopping, ExperimentLogger, TrainingHistory


class _SnapshotModel:
    def __init__(self) -> None:
        self.weight = np.array([1.0])

    def get_snapshot(self) -> dict:
        return {"weight": self.weight.copy()}


def test_training_history_interval_and_deep_snapshot_contract():
    model = _SnapshotModel()
    history = TrainingHistory(snapshot_interval=2)
    logs = {"loss": 1.0}
    history.on_epoch_end(0, model, logs)
    model.weight[0] = 9.0
    logs["loss"] = 0.5
    history.on_epoch_end(1, model, logs)
    history.on_epoch_end(2, model, logs)

    assert [snapshot["epoch"] for snapshot in history.snapshots] == [0, 2]
    assert history.snapshots[0]["weight"][0] == 1.0
    assert history.snapshots[0]["logs"] == {"loss": 1.0}
    assert "interval=2" in repr(history)
    with pytest.raises(ValueError):
        TrainingHistory(snapshot_interval=0)


def test_early_stopping_improvement_patience_missing_loss_and_reset():
    callback = EarlyStopping(patience=2, min_delta=0.1)
    assert callback.on_epoch_end(0, None, {"loss": 1.0}) is False
    assert callback.on_epoch_end(1, None, {"loss": 0.95}) is False
    assert callback.on_epoch_end(2, None, {"loss": 0.94}) is True
    callback.reset()
    assert np.isinf(callback.best_loss) and callback.counter == 0
    assert callback.on_epoch_end(3, None, {}) is False
    assert "patience=2" in repr(callback)
    with pytest.raises(ValueError):
        EarlyStopping(patience=0)


def test_early_stopping_restores_best_sequential_like_parameters():
    class Layer:
        weights = np.array([1.0])
        biases = np.array([0.0])

    class Model:
        def __init__(self):
            self._layers = [Layer()]

    model = Model()
    callback = EarlyStopping(patience=1, monitor="val_loss", restore_best_weights=True)
    assert callback.on_epoch_end(0, model, {"val_loss": 0.5}) is False
    model._layers[0].weights[:] = 99.0
    model._layers[0].biases[:] = 99.0
    assert callback.on_epoch_end(1, model, {"val_loss": 0.6}) is True
    np.testing.assert_array_equal(model._layers[0].weights, [1.0])
    np.testing.assert_array_equal(model._layers[0].biases, [0.0])


def test_experiment_logger_round_trips_numpy_without_mutating_values(tmp_path: Path):
    logger = ExperimentLogger(str(tmp_path))
    path = Path(
        logger.log_experiment(
            {"array": np.array([1, 2]), "integer": np.int64(3)},
            {"score": np.float64(0.5), "passed": np.bool_(True)},
        )
    )
    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["params"] == {"array": [1, 2], "integer": 3}
    assert record["results"] == {"score": 0.5, "passed": True}
    assert len(record["trace_id"]) == 8
    assert record["timestamp"]
    assert "ExperimentLogger" in repr(logger)
