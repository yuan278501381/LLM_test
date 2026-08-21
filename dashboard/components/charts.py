# Copyright (c) 2026 Yy1 (yuan278501381) | MIT License
"""
dashboard.components.charts - Plotly 可视化图表工厂

封装所有 Plotly 图表的生成逻辑，提供统一的样式和交互体验。
所有图表使用暗色主题，中文标签，完整的交互功能。
"""

from typing import Any

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ---------------------------------------------------------------------------
# 全局样式配置
# ---------------------------------------------------------------------------
_THEME = "plotly_dark"
_COLORS = {
    "primary": "#00D4FF",
    "secondary": "#FF6B6B",
    "accent": "#51CF66",
    "warning": "#FFD43B",
    "class_0": "#00D4FF",
    "class_1": "#FF6B6B",
    "class_2": "#51CF66",
    "class_3": "#FFD43B",
}
_CLASS_COLORS = ["#00D4FF", "#FF6B6B", "#51CF66", "#FFD43B", "#B197FC"]


def _apply_theme(fig: go.Figure) -> go.Figure:
    """统一应用暗色主题"""
    fig.update_layout(
        template=_THEME,
        font=dict(family="system-ui, sans-serif", size=12),
        margin=dict(l=40, r=20, t=50, b=40),
        hoverlabel=dict(font_size=12),
    )
    return fig


# ---------------------------------------------------------------------------
# 决策边界
# ---------------------------------------------------------------------------
def plot_decision_boundary(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    resolution: int = 100,
    title: str = "🗺️ 决策边界",
) -> go.Figure:
    """
    绘制模型在 2D 空间上的决策边界。

    原理: 在特征空间中生成均匀网格，用模型预测每个网格点的类别，
    用等高线图绘制分类区域，再叠加训练数据散点。

    Args:
        model: Sequential 模型实例
        X: 训练数据，shape (n, 2)
        y: 标签，shape (n, 1) 或 (n, c)
        resolution: 网格密度
        title: 图表标题
    """
    # 生成 2D 网格
    margin = 0.2
    x_min, x_max = X[:, 0].min() - margin, X[:, 0].max() + margin
    y_min, y_max = X[:, 1].min() - margin, X[:, 1].max() + margin

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, resolution),
        np.linspace(y_min, y_max, resolution),
    )
    grid = np.c_[xx.ravel(), yy.ravel()]

    # 模型预测
    preds = model.predict(grid)
    if preds.shape[1] == 1:
        zz = preds.reshape(xx.shape)
    else:
        zz = np.argmax(preds, axis=1).reshape(xx.shape).astype(float)

    fig = go.Figure()

    # 等高线填充
    fig.add_trace(go.Contour(
        x=np.linspace(x_min, x_max, resolution),
        y=np.linspace(y_min, y_max, resolution),
        z=zz,
        colorscale=[[0, "rgba(0,212,255,0.3)"], [1, "rgba(255,107,107,0.3)"]],
        showscale=False,
        contours=dict(showlines=True, coloring="fill"),
        line=dict(width=1, color="white"),
        hoverinfo="skip",
    ))

    # 训练数据散点
    if y.shape[1] == 1:
        labels = y.ravel()
    else:
        labels = np.argmax(y, axis=1)

    for cls_idx in sorted(set(labels.astype(int))):
        mask = labels == cls_idx
        fig.add_trace(go.Scatter(
            x=X[mask, 0], y=X[mask, 1],
            mode="markers",
            name=f"类别 {cls_idx}",
            marker=dict(
                size=6, color=_CLASS_COLORS[cls_idx % len(_CLASS_COLORS)],
                line=dict(width=1, color="white"),
            ),
        ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        xaxis_title="特征 x₁",
        yaxis_title="特征 x₂",
        showlegend=True,
        legend=dict(orientation="h", y=-0.15),
        height=450,
    )
    return _apply_theme(fig)


# ---------------------------------------------------------------------------
# 损失曲线
# ---------------------------------------------------------------------------
def plot_loss_curve(
    history: dict[str, list[float]],
    title: str = "📈 训练曲线",
) -> go.Figure:
    """
    绘制 Loss 和 Accuracy 的训练曲线（双 Y 轴）。

    Args:
        history: 训练历史字典 {'loss': [...], 'accuracy': [...], ...}
        title: 图表标题
    """
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    epochs = list(range(1, len(history["loss"]) + 1))

    # Loss 曲线
    fig.add_trace(
        go.Scatter(
            x=epochs, y=history["loss"],
            name="训练 Loss", mode="lines",
            line=dict(color=_COLORS["secondary"], width=2),
        ),
        secondary_y=False,
    )

    # Accuracy 曲线
    if history.get("accuracy"):
        fig.add_trace(
            go.Scatter(
                x=epochs, y=history["accuracy"],
                name="训练 Accuracy", mode="lines",
                line=dict(color=_COLORS["accent"], width=2),
            ),
            secondary_y=True,
        )

    # 验证集曲线
    if history.get("val_loss"):
        fig.add_trace(
            go.Scatter(
                x=epochs, y=history["val_loss"],
                name="验证 Loss", mode="lines",
                line=dict(color=_COLORS["secondary"], width=2, dash="dash"),
            ),
            secondary_y=False,
        )
    if history.get("val_accuracy"):
        fig.add_trace(
            go.Scatter(
                x=epochs, y=history["val_accuracy"],
                name="验证 Accuracy", mode="lines",
                line=dict(color=_COLORS["accent"], width=2, dash="dash"),
            ),
            secondary_y=True,
        )

    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        legend=dict(orientation="h", y=-0.15),
        height=400,
    )
    fig.update_yaxes(title_text="Loss", secondary_y=False)
    fig.update_yaxes(title_text="Accuracy", secondary_y=True, range=[0, 1.05])
    fig.update_xaxes(title_text="Epoch")
    return _apply_theme(fig)


# ---------------------------------------------------------------------------
# 权重 / 梯度直方图
# ---------------------------------------------------------------------------
def plot_weight_histograms(
    snapshot: dict,
    title: str = "📊 权重分布",
) -> go.Figure:
    """
    绘制每个 Dense 层的权重分布直方图。

    Args:
        snapshot: model.get_snapshot() 的返回值
        title: 图表标题
    """
    dense_layers = [
        info for info in snapshot["layers"]
        if info["type"] == "Dense" and info["weights"] is not None
    ]

    if not dense_layers:
        fig = go.Figure()
        fig.add_annotation(text="暂无权重数据", xref="paper", yref="paper", x=0.5, y=0.5)
        return _apply_theme(fig)

    n = len(dense_layers)
    fig = make_subplots(rows=1, cols=n, subplot_titles=[l["name"] for l in dense_layers])

    for i, info in enumerate(dense_layers):
        weights = info["weights"].ravel()
        fig.add_trace(
            go.Histogram(
                x=weights, nbinsx=50,
                marker_color=_CLASS_COLORS[i % len(_CLASS_COLORS)],
                opacity=0.8, name=info["name"],
                showlegend=False,
            ),
            row=1, col=i + 1,
        )

    fig.update_layout(title=dict(text=title, font=dict(size=16)), height=350)
    return _apply_theme(fig)


def plot_gradient_histograms(
    snapshot: dict,
    title: str = "🔥 梯度分布",
) -> go.Figure:
    """绘制每个 Dense 层的梯度分布直方图"""
    dense_layers = [
        info for info in snapshot["layers"]
        if info["type"] == "Dense" and info["grad_weights"] is not None
    ]

    if not dense_layers:
        fig = go.Figure()
        fig.add_annotation(text="暂无梯度数据", xref="paper", yref="paper", x=0.5, y=0.5)
        return _apply_theme(fig)

    n = len(dense_layers)
    fig = make_subplots(rows=1, cols=n, subplot_titles=[l["name"] for l in dense_layers])

    for i, info in enumerate(dense_layers):
        grads = info["grad_weights"].ravel()
        fig.add_trace(
            go.Histogram(
                x=grads, nbinsx=50,
                marker_color=_CLASS_COLORS[i % len(_CLASS_COLORS)],
                opacity=0.8, name=info["name"],
                showlegend=False,
            ),
            row=1, col=i + 1,
        )

    fig.update_layout(title=dict(text=title, font=dict(size=16)), height=350)
    return _apply_theme(fig)


# ---------------------------------------------------------------------------
# 激活值热力图
# ---------------------------------------------------------------------------
def plot_activation_heatmap(
    snapshot: dict,
    layer_idx: int = 0,
    title: str = "⚡ 激活值热力图",
) -> go.Figure:
    """
    绘制指定层输出的热力图。

    Args:
        snapshot: 模型快照
        layer_idx: 目标层索引
        title: 图表标题
    """
    layers = snapshot["layers"]
    if layer_idx >= len(layers) or layers[layer_idx]["output"] is None:
        fig = go.Figure()
        fig.add_annotation(text="暂无激活数据", xref="paper", yref="paper", x=0.5, y=0.5)
        return _apply_theme(fig)

    output = layers[layer_idx]["output"]
    # 最多展示前 100 个样本
    output = output[:100]

    fig = go.Figure(data=go.Heatmap(
        z=output,
        colorscale="RdBu_r",
        zmid=0,
        colorbar=dict(title="激活值"),
    ))

    fig.update_layout(
        title=dict(text=f"{title} — {layers[layer_idx]['name']}", font=dict(size=16)),
        xaxis_title="神经元索引",
        yaxis_title="样本索引",
        height=400,
    )
    return _apply_theme(fig)


# ---------------------------------------------------------------------------
# 多优化器对比
# ---------------------------------------------------------------------------
def plot_multi_loss_curves(
    histories: dict[str, dict[str, list[float]]],
    title: str = "⚙️ 优化器对比",
) -> go.Figure:
    """
    在同一图表上绘制多个优化器的 Loss 曲线。

    Args:
        histories: {优化器名称: history_dict}
        title: 图表标题
    """
    fig = go.Figure()

    for i, (name, history) in enumerate(histories.items()):
        epochs = list(range(1, len(history["loss"]) + 1))
        fig.add_trace(go.Scatter(
            x=epochs, y=history["loss"],
            name=name, mode="lines",
            line=dict(color=_CLASS_COLORS[i % len(_CLASS_COLORS)], width=2.5),
        ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        xaxis_title="Epoch",
        yaxis_title="Loss",
        legend=dict(orientation="h", y=-0.15),
        height=400,
    )
    return _apply_theme(fig)


# ---------------------------------------------------------------------------
# 权重变化轨迹
# ---------------------------------------------------------------------------
def plot_weight_trajectory(
    snapshots: list[dict],
    layer_idx: int = 0,
    title: str = "🛤️ 权重变化轨迹",
) -> go.Figure:
    """
    绘制指定层的权重在训练过程中的变化轨迹。

    取前两个权重元素作为 2D 坐标，绘制参数空间中的移动路径。

    Args:
        snapshots: TrainingHistory.snapshots 列表
        layer_idx: 目标 Dense 层索引
        title: 图表标题
    """
    # 找到目标 Dense 层在 snapshot 中的位置
    w_values: list[tuple[float, float]] = []

    for snap in snapshots:
        dense_count = 0
        for info in snap["layers"]:
            if info["type"] == "Dense":
                if dense_count == layer_idx and info["weights"] is not None:
                    w = info["weights"].ravel()
                    if len(w) >= 2:
                        w_values.append((float(w[0]), float(w[1])))
                    break
                dense_count += 1

    if not w_values:
        fig = go.Figure()
        fig.add_annotation(text="暂无轨迹数据", xref="paper", yref="paper", x=0.5, y=0.5)
        return _apply_theme(fig)

    xs, ys = zip(*w_values)

    fig = go.Figure()

    # 轨迹线
    fig.add_trace(go.Scatter(
        x=list(xs), y=list(ys),
        mode="lines+markers",
        marker=dict(
            size=4, color=list(range(len(xs))),
            colorscale="Viridis", showscale=True,
            colorbar=dict(title="Epoch"),
        ),
        line=dict(color="rgba(255,255,255,0.3)", width=1),
        name="权重轨迹",
    ))

    # 起点和终点
    fig.add_trace(go.Scatter(
        x=[xs[0]], y=[ys[0]],
        mode="markers", name="起点",
        marker=dict(size=12, color=_COLORS["accent"], symbol="star"),
    ))
    fig.add_trace(go.Scatter(
        x=[xs[-1]], y=[ys[-1]],
        mode="markers", name="终点",
        marker=dict(size=12, color=_COLORS["secondary"], symbol="diamond"),
    ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        xaxis_title="W[0]",
        yaxis_title="W[1]",
        legend=dict(orientation="h", y=-0.15),
        height=400,
    )
    return _apply_theme(fig)
