# Copyright (c) 2026 Yy1 (yuan278501381) | MIT License
"""
nn_core.model - Sequential 模型容器

将多个层（Dense、Activation、Dropout）按顺序堆叠成完整的神经网络，
提供训练循环、推理、快照等全生命周期管理。

典型用法:
    model = Sequential()
    model.add(Dense(2, 16, initializer='he'))
    model.add(ReLU())
    model.add(Dense(16, 1))
    model.add(Sigmoid())

    history = model.train(X, y, epochs=100, batch_size=32,
                          loss_fn=BinaryCrossEntropy(), optimizer=Adam())
"""

import inspect
import logging
import time
import uuid
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class Sequential:
    """
    顺序模型容器 — 按添加顺序依次执行前向/反向传播。

    设计原则:
        - 层的组装通过 add() 链式调用
        - forward/backward 自动处理不同层类型的接口差异
        - get_snapshot() 提供完整的内部状态快照，供可视化使用
        - train() 封装完整的训练循环（mini-batch SGD）
    """

    def __init__(self) -> None:
        self._layers: list[Any] = []
        self._tid = uuid.uuid4().hex[:8]
        logger.info("[%s] Sequential 模型已创建", self._tid)

    # ------------------------------------------------------------------
    # 模型构建
    # ------------------------------------------------------------------
    def add(self, layer: Any) -> "Sequential":
        """
        添加一个层到模型末尾。支持链式调用。

        Args:
            layer: Dense、Activation 或 Dropout 实例

        Returns:
            self（支持链式调用）
        """
        self._layers.append(layer)
        logger.debug("[%s] 添加层: %s (总层数=%d)", self._tid, layer, len(self._layers))
        return self

    @property
    def layers(self) -> list[Any]:
        """获取所有层的列表"""
        return self._layers

    # ------------------------------------------------------------------
    # 前向传播
    # ------------------------------------------------------------------
    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """
        逐层执行前向传播。

        自动识别不同层类型的接口:
            - Dense / Dropout: 传入 (x, training)
            - Activation: 仅传入 (x,)

        Args:
            x: 输入数据，shape (batch_size, n_features)
            training: 是否为训练模式

        Returns:
            模型输出
        """
        for layer in self._layers:
            sig = inspect.signature(layer.forward)
            if "training" in sig.parameters:
                x = layer.forward(x, training=training)
            else:
                x = layer.forward(x)
        return x

    # ------------------------------------------------------------------
    # 反向传播
    # ------------------------------------------------------------------
    def backward(self, dloss: np.ndarray) -> None:
        """
        逆序逐层执行反向传播。

        Args:
            dloss: 损失函数对模型输出的梯度
        """
        for layer in reversed(self._layers):
            dloss = layer.backward(dloss)

    # ------------------------------------------------------------------
    # 推理
    # ------------------------------------------------------------------
    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        推理模式前向传播（Dropout 不生效等）。

        Args:
            x: 输入数据

        Returns:
            模型预测输出
        """
        return self.forward(x, training=False)

    # ------------------------------------------------------------------
    # 训练循环
    # ------------------------------------------------------------------
    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 100,
        batch_size: int | None = None,
        loss_fn: Any = None,
        optimizer: Any = None,
        callbacks: list[Any] | None = None,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        verbose: bool = False,
        **kwargs: Any,
    ) -> dict[str, list[float]]:
        """
        执行完整的训练循环。

        训练流程（每个 epoch）:
            1. 打乱数据
            2. 按 batch_size 切分 mini-batch (若 batch_size 为 None 则执行全量 Batch GD)
            3. 每个 batch: forward → loss → backward → optimizer.step
            4. 记录 epoch 级别的平均指标
            5. 可选: 计算验证集指标
            6. 调用回调函数

        Args:
            X: 训练数据，shape (n_samples, n_features)
            y: 训练标签，shape (n_samples, n_outputs)
            epochs: 训练轮数
            batch_size: 批大小（None 或 <=0 表示全量批次）
            loss_fn: 损失函数实例
            optimizer: 优化器实例
            callbacks: 回调列表，可选
            X_val: 验证数据，可选
            y_val: 验证标签，可选
            verbose: 是否输出详细日志
            **kwargs: 额外扩展参数

        Returns:
            history 字典: {
                'loss': [...],
                'accuracy': [...],
                'val_loss': [...],
                'val_accuracy': [...]
            }
        """
        tid = uuid.uuid4().hex[:8]
        n_samples = X.shape[0]
        actual_batch_size = (
            n_samples if (batch_size is None or batch_size <= 0) else min(batch_size, n_samples)
        )
        callbacks = callbacks or []

        history: dict[str, list[float]] = {
            "loss": [],
            "accuracy": [],
            "val_loss": [],
            "val_accuracy": [],
        }

        logger.info(
            "[%s] 开始训练: epochs=%d, batch_size=%d (actual=%d), 样本数=%d, 优化器=%s",
            tid,
            epochs,
            batch_size if batch_size is not None else -1,
            actual_batch_size,
            n_samples,
            optimizer,
        )

        t_start = time.perf_counter()

        for epoch in range(epochs):
            # ---- 1. 打乱数据 ----
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            # ---- 2. Mini-batch 训练 ----
            epoch_losses: list[float] = []
            epoch_correct = 0
            epoch_total = 0

            for start in range(0, n_samples, actual_batch_size):
                end = min(start + actual_batch_size, n_samples)
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]
                batch_n = end - start

                # 前向传播
                y_pred = self.forward(X_batch, training=True)

                # 计算损失
                loss_val = loss_fn.forward(y_pred, y_batch)
                epoch_losses.append(loss_val)

                # 反向传播
                dloss = loss_fn.backward()
                self.backward(dloss)

                # 参数更新
                optimizer.step(self._layers)

                # 计算准确率
                epoch_correct += self._count_correct(y_pred, y_batch)
                epoch_total += batch_n

            # ---- 3. Epoch 级别指标 ----
            avg_loss = float(np.mean(epoch_losses))
            accuracy = epoch_correct / epoch_total if epoch_total > 0 else 0.0
            history["loss"].append(avg_loss)
            history["accuracy"].append(accuracy)

            # ---- 4. 验证集指标 ----
            if X_val is not None and y_val is not None:
                val_pred = self.predict(X_val)
                val_loss = float(loss_fn.forward(val_pred, y_val))
                val_correct = self._count_correct(val_pred, y_val)
                val_accuracy = val_correct / X_val.shape[0]
                history["val_loss"].append(val_loss)
                history["val_accuracy"].append(val_accuracy)

            # ---- 5. 回调 ----
            logs = {
                "loss": avg_loss,
                "accuracy": accuracy,
            }
            if X_val is not None:
                logs["val_loss"] = history["val_loss"][-1]
                logs["val_accuracy"] = history["val_accuracy"][-1]

            should_stop = False
            for cb in callbacks:
                result = cb.on_epoch_end(epoch, self, logs)
                if result is True:
                    should_stop = True

            # ---- 6. 日志输出（每 10% 的 epoch 输出一次）----
            log_interval = max(1, epochs // 10)
            if epoch % log_interval == 0 or epoch == epochs - 1:
                val_info = ""
                if X_val is not None:
                    val_info = (
                        f", val_loss={history['val_loss'][-1]:.4f}"
                        f", val_acc={history['val_accuracy'][-1]:.4f}"
                    )
                logger.info(
                    "[%s] Epoch %d/%d — loss=%.4f, acc=%.4f%s",
                    tid,
                    epoch + 1,
                    epochs,
                    avg_loss,
                    accuracy,
                    val_info,
                )

            if should_stop:
                logger.info("[%s] EarlyStopping 在 epoch %d 触发", tid, epoch + 1)
                break

        elapsed = time.perf_counter() - t_start
        logger.info(
            "[%s] 训练完成: 最终 loss=%.4f, acc=%.4f, 耗时=%.2fs",
            tid,
            history["loss"][-1],
            history["accuracy"][-1],
            elapsed,
        )

        return history

    # ------------------------------------------------------------------
    # 模型快照（供可视化使用）
    # ------------------------------------------------------------------
    def get_snapshot(self) -> dict[str, Any]:
        """
        获取模型所有层的完整快照。

        返回每层的权重、梯度、输入/输出缓存的深拷贝，
        确保后续训练不会覆盖快照数据。

        Returns:
            {
                'layers': [
                    {
                        'name': str,           # 如 'Dense_0'
                        'type': str,           # 如 'Dense'
                        'weights': ndarray | None,
                        'biases': ndarray | None,
                        'grad_weights': ndarray | None,
                        'grad_biases': ndarray | None,
                        'input': ndarray | None,
                        'output': ndarray | None,
                    },
                    ...
                ]
            }
        """
        snapshot: dict[str, Any] = {"layers": []}

        for i, layer in enumerate(self._layers):
            info: dict[str, Any] = {
                "name": f"{type(layer).__name__}_{i}",
                "type": type(layer).__name__,
                "weights": None,
                "biases": None,
                "grad_weights": None,
                "grad_biases": None,
                "input": None,
                "output": None,
            }

            # 复制数值数据（深拷贝防止后续训练覆盖）
            for attr, key in [
                ("weights", "weights"),
                ("biases", "biases"),
                ("grad_weights", "grad_weights"),
                ("grad_biases", "grad_biases"),
                ("input_cache", "input"),
                ("output_cache", "output"),
            ]:
                val = getattr(layer, attr, None)
                if val is not None:
                    info[key] = val.copy()

            snapshot["layers"].append(info)

        return snapshot

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------
    def _count_correct(self, y_pred: np.ndarray, y_true: np.ndarray) -> int:
        """
        计算预测正确的样本数。

        自动区分:
            - 二分类 (y.shape[1] == 1): 以 0.5 为阈值
            - 多分类 (y.shape[1] > 1): 比较 argmax
        """
        if y_true.shape[1] == 1:
            # 二分类
            predicted = (y_pred >= 0.5).astype(int)
            return int(np.sum(predicted == y_true.astype(int)))
        else:
            # 多分类
            return int(np.sum(np.argmax(y_pred, axis=1) == np.argmax(y_true, axis=1)))

    def __repr__(self) -> str:
        lines = [f"Sequential(共 {len(self._layers)} 层):"]
        for i, layer in enumerate(self._layers):
            lines.append(f"  [{i}] {layer}")
        return "\n".join(lines)
