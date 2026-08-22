# Copyright (c) 2026 Yy1 (yuan278501381) | MIT License
"""Sequential 训练循环的数据、随机性、尾批与验证集契约。"""

import numpy as np
import pytest

from nn_core.model import Sequential


class RecordingLayer:
    def __init__(self) -> None:
        self.batches: list[np.ndarray] = []

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        if training:
            self.batches.append(x.copy())
        return x[:, :1]

    def backward(self, dout: np.ndarray) -> np.ndarray:
        return dout


class SquaredLoss:
    def forward(self, prediction: np.ndarray, target: np.ndarray) -> float:
        self.prediction = prediction
        self.target = target
        return float(np.mean((prediction - target) ** 2))

    def backward(self) -> np.ndarray:
        return 2 * (self.prediction - self.target) / self.prediction.size


class NoOpOptimizer:
    def step(self, _layers) -> None:
        return None


def _trained_batch_order(seed: int) -> tuple[list[int], dict[str, list[float]]]:
    layer = RecordingLayer()
    model = Sequential().add(layer)
    X = np.arange(10, dtype=float).reshape(5, 2)
    y = np.zeros((5, 1))
    history = model.train(
        X,
        y,
        epochs=1,
        batch_size=2,
        loss_fn=SquaredLoss(),
        optimizer=NoOpOptimizer(),
        X_val=X[:2],
        y_val=y[:2],
        seed=seed,
    )
    return [len(batch) for batch in layer.batches], history


def test_train_covers_tail_batch_validation_and_seed_without_global_rng():
    np.random.seed(31415)
    expected_next_global = np.random.random()
    np.random.seed(31415)
    batches_a, history_a = _trained_batch_order(8)
    batches_b, history_b = _trained_batch_order(8)

    assert batches_a == [2, 2, 1]
    assert batches_b == batches_a
    assert history_a == history_b
    assert len(history_a["val_loss"]) == len(history_a["loss"]) == 1
    assert np.random.random() == expected_next_global


@pytest.mark.parametrize(
    "X,y,kwargs",
    [
        (np.empty((0, 2)), np.empty((0, 1)), {}),
        (np.ones((2, 2)), np.ones((3, 1)), {}),
        (np.ones((2, 2)), np.ones((2, 1)), {"epochs": 0}),
        (np.ones((2, 2)), np.ones((2, 1)), {"X_val": np.ones((1, 2))}),
        (
            np.ones((2, 2)),
            np.ones((2, 1)),
            {"seed": 1, "rng": np.random.default_rng(1)},
        ),
    ],
)
def test_train_rejects_invalid_contracts(X, y, kwargs):
    model = Sequential().add(RecordingLayer())
    with pytest.raises(ValueError):
        model.train(X, y, loss_fn=SquaredLoss(), optimizer=NoOpOptimizer(), **kwargs)


def test_callback_can_stop_and_snapshot_is_deep_copy():
    class StopImmediately:
        def on_epoch_end(self, _epoch, _model, _logs) -> bool:
            return True

    class StatefulLayer(RecordingLayer):
        def __init__(self) -> None:
            super().__init__()
            self.weights = np.array([[2.0]])
            self.biases = np.array([1.0])
            self.grad_weights = np.array([[0.5]])
            self.grad_biases = np.array([0.25])
            self.input_cache = np.array([[3.0]])
            self.output_cache = np.array([[7.0]])

    layer = StatefulLayer()
    model = Sequential().add(layer)
    history = model.train(
        np.ones((2, 1)),
        np.zeros((2, 1)),
        epochs=5,
        loss_fn=SquaredLoss(),
        optimizer=NoOpOptimizer(),
        callbacks=[StopImmediately()],
        seed=1,
    )
    assert len(history["loss"]) == 1
    snapshot = model.get_snapshot()
    assert snapshot["layers"][0]["type"] == "StatefulLayer"
    assert "StatefulLayer" in repr(model)
    layer.weights[0, 0] = 999.0
    assert snapshot["layers"][0]["weights"][0, 0] == 2.0


def test_multiclass_count_correct_uses_argmax():
    model = Sequential()
    predictions = np.array([[0.1, 0.8, 0.1], [0.7, 0.2, 0.1]])
    labels = np.array([[0, 1, 0], [0, 0, 1]])
    assert model._count_correct(predictions, labels) == 1
