# Copyright (c) 2026 Yy1 (yuan278501381) | MIT License
"""
nn_core.callbacks - 训练回调模块

提供训练过程中的事件钩子，用于记录快照、早停、实验日志等。

支持的回调:
    - TrainingHistory: 按间隔记录模型快照
    - EarlyStopping: 损失不再下降时自动停止
    - ExperimentLogger: 将实验参数和结果序列化为 JSON
"""

import json
import logging
import os
import uuid
from datetime import datetime
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class TrainingHistory:
    """
    训练历史记录器 — 按间隔记录模型的完整快照。

    每隔 snapshot_interval 个 epoch，调用 model.get_snapshot() 保存
    当前权重、梯度、激活值的深拷贝，供可视化回放使用。

    Args:
        snapshot_interval: 快照间隔（每 N 个 epoch 记录一次），默认 1
    """

    def __init__(self, snapshot_interval: int = 1) -> None:
        if snapshot_interval < 1:
            raise ValueError(f"snapshot_interval 必须 >= 1，收到: {snapshot_interval}")
        self.snapshot_interval = snapshot_interval
        self.snapshots: list[dict[str, Any]] = []
        logger.debug("TrainingHistory: 快照间隔=%d", snapshot_interval)

    def on_epoch_end(self, epoch: int, model: Any, logs: dict) -> None:
        """
        Epoch 结束时的回调。

        Args:
            epoch: 当前 epoch 编号（0-indexed）
            model: Sequential 模型实例
            logs: 当前 epoch 的指标（loss, accuracy 等）
        """
        if epoch % self.snapshot_interval == 0:
            snapshot = model.get_snapshot()
            snapshot["epoch"] = epoch
            snapshot["logs"] = logs.copy()
            self.snapshots.append(snapshot)
            logger.debug(
                "TrainingHistory: 已记录 epoch %d 的快照（累计 %d 份）",
                epoch,
                len(self.snapshots),
            )

    def __repr__(self) -> str:
        return f"TrainingHistory(interval={self.snapshot_interval}, 已记录={len(self.snapshots)})"


class EarlyStopping:
    """
    早停回调 — 当损失不再改善时自动停止训练。

    机制: 持续监控训练损失，如果连续 patience 个 epoch 的损失
    都没有比历史最优值改善超过 min_delta，则触发早停。

    Args:
        patience: 最大容忍不改善的 epoch 数量，默认 10
        min_delta: 最小改善阈值，默认 1e-4
    """

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 1e-4,
        *,
        monitor: str = "loss",
        restore_best_weights: bool = True,
    ) -> None:
        if patience < 1:
            raise ValueError(f"patience 必须 >= 1，收到: {patience}")
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.restore_best_weights = restore_best_weights
        self.best_loss: float = float("inf")
        self.counter: int = 0
        self._best_parameters: list[dict[str, np.ndarray]] | None = None
        logger.debug(
            "EarlyStopping: patience=%d, min_delta=%.6f",
            patience,
            min_delta,
        )

    def on_epoch_end(self, epoch: int, model: Any, logs: dict) -> bool:
        """
        Epoch 结束时检查是否应该早停。

        Args:
            epoch: 当前 epoch 编号
            model: Sequential 模型实例
            logs: 当前 epoch 的指标

        Returns:
            True = 应该停止训练，False = 继续
        """
        current_loss = logs.get(self.monitor, float("inf"))

        if current_loss < self.best_loss - self.min_delta:
            # 损失有显著改善，重置计数器
            self.best_loss = current_loss
            self.counter = 0
            self._best_parameters = self._capture_parameters(model)
            return False

        # 损失没有改善，递增计数器
        self.counter += 1
        if self.counter >= self.patience:
            if self.restore_best_weights:
                self._restore_parameters(model)
            logger.info(
                "EarlyStopping 触发: 连续 %d 个 epoch 无改善 (best_loss=%.6f, current=%.6f)",
                self.patience,
                self.best_loss,
                current_loss,
            )
            return True
        return False

    def reset(self) -> None:
        """重置早停状态（用于重新训练）"""
        self.best_loss = float("inf")
        self.counter = 0
        self._best_parameters = None

    @staticmethod
    def _capture_parameters(model: Any) -> list[dict[str, np.ndarray]] | None:
        """复制 Sequential-like 层参数；无参数模型返回 None。"""
        layers = getattr(model, "_layers", None)
        if layers is None:
            return None
        captured: list[dict[str, np.ndarray]] = []
        for layer in layers:
            params: dict[str, np.ndarray] = {}
            for name in ("weights", "biases"):
                value = getattr(layer, name, None)
                if isinstance(value, np.ndarray):
                    params[name] = value.copy()
            captured.append(params)
        return captured

    def _restore_parameters(self, model: Any) -> None:
        """在触发早停时恢复最佳 epoch 的权重和偏置。"""
        layers = getattr(model, "_layers", None)
        if layers is None or self._best_parameters is None:
            return
        for layer, params in zip(layers, self._best_parameters, strict=True):
            for name, value in params.items():
                setattr(layer, name, value.copy())

    def __repr__(self) -> str:
        return (
            f"EarlyStopping(patience={self.patience}, "
            f"counter={self.counter}, best={self.best_loss:.6f})"
        )


class _NumpyEncoder(json.JSONEncoder):
    """自定义 JSON 编码器，处理 NumPy 类型序列化"""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


class ExperimentLogger:
    """
    实验日志记录器 — 将实验参数和结果序列化为 JSON 文件。

    每次实验生成一个独立的 JSON 文件，包含完整的超参数配置、
    训练结果和时间戳，方便后续实验对比和追溯。

    Args:
        log_dir: 日志输出目录，默认 'logs/'
    """

    def __init__(self, log_dir: str = "logs/") -> None:
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        logger.debug("ExperimentLogger: log_dir=%s", log_dir)

    def log_experiment(self, params: dict[str, Any], results: dict[str, Any]) -> str:
        """
        记录一次实验。

        Args:
            params: 实验超参数（学习率、层数、优化器等）
            results: 实验结果（最终 loss、accuracy、训练时长等）

        Returns:
            生成的日志文件路径
        """
        tid = uuid.uuid4().hex[:8]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"experiment_{timestamp}_{tid}.json"
        filepath = os.path.join(self.log_dir, filename)

        record: dict[str, Any] = {
            "trace_id": tid,
            "timestamp": datetime.now().isoformat(),
            "params": params,
            "results": results,
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(record, f, cls=_NumpyEncoder, ensure_ascii=False, indent=2)

        logger.info("[%s] 实验日志已保存: %s", tid, filepath)
        return filepath

    def __repr__(self) -> str:
        return f"ExperimentLogger(dir='{self.log_dir}')"
