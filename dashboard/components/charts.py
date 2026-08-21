# Copyright (c) 2026 Yy1 (yuan278501381) | MIT License
"""
dashboard.components.charts - 世界级 Plotly 可视化图表引擎 (极简极客风格)

封装所有 Plotly 图表的生成逻辑，提供极简深空霓虹主题与交互体验（杜绝一切 Emoji）。
- 连续平滑概率场决策边界（支持探针样本点高亮）
- 发光 Loss 曲线与最小损失标注
- 多层梯度与权重流形直方图
- 神经元逐层激活强度热力图
- 优化器多轨竞速轨迹图
"""

from typing import Any

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ---------------------------------------------------------------------------
# 全局霓虹视觉调色板
# ---------------------------------------------------------------------------
CYBER_PALETTE = {
    "bg_plot": "rgba(8, 12, 20, 0.4)",
    "bg_paper": "rgba(0, 0, 0, 0)",
    "grid": "rgba(255, 255, 255, 0.05)",
    "zero_line": "rgba(255, 255, 255, 0.15)",
    "font_color": "#e2e8f0",
    "font_muted": "#94a3b8",
    "primary": "#38bdf8",     # 霓虹青蓝
    "secondary": "#fb7185",   # 珊瑚烈焰
    "accent": "#34d399",      # 电光极光绿
    "warning": "#fbbf24",     # 琥珀金
    "purple": "#c084fc",      # 量子紫
    "classes": ["#38bdf8", "#fb7185", "#34d399", "#fbbf24", "#c084fc"],
    "optimizers": {
        "SGD": "#94a3b8",
        "Momentum": "#38bdf8",
        "RMSProp": "#fbbf24",
        "Adam": "#34d399",
    },
}


def _apply_cyber_theme(fig: go.Figure, title: str | None = None) -> go.Figure:
    """统一注入赛博深空图表布局属性"""
    layout_update: dict[str, Any] = {
        "template": "plotly_dark",
        "plot_bgcolor": CYBER_PALETTE["bg_plot"],
        "paper_bgcolor": CYBER_PALETTE["bg_paper"],
        "font": dict(
            family="Plus Jakarta Sans, -apple-system, Segoe UI, sans-serif",
            size=12,
            color=CYBER_PALETTE["font_color"],
        ),
        "margin": dict(l=35, r=25, t=50 if title else 25, b=35),
        "hoverlabel": dict(
            bgcolor="rgba(15, 23, 42, 0.95)",
            bordercolor="rgba(56, 189, 248, 0.4)",
            font=dict(family="JetBrains Mono, monospace", size=12, color="#ffffff"),
        ),
        "xaxis": dict(
            gridcolor=CYBER_PALETTE["grid"],
            zerolinecolor=CYBER_PALETTE["zero_line"],
            tickfont=dict(size=10, color=CYBER_PALETTE["font_muted"]),
        ),
        "yaxis": dict(
            gridcolor=CYBER_PALETTE["grid"],
            zerolinecolor=CYBER_PALETTE["zero_line"],
            tickfont=dict(size=10, color=CYBER_PALETTE["font_muted"]),
        ),
    }

    if title:
        layout_update["title"] = dict(
            text=f"<b>{title}</b>",
            font=dict(size=14, color="#f8fafc"),
            x=0.02,
            y=0.96,
        )

    fig.update_layout(**layout_update)
    return fig


# ---------------------------------------------------------------------------
# 1. 决策边界 (Decision Boundary)
# ---------------------------------------------------------------------------
def plot_decision_boundary(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    probe_point: tuple[float, float] | None = None,
    resolution: int = 100,
    title: str = "DECISION MANIFOLD // 连续概率决策流形",
) -> go.Figure:
    """
    绘制模型在 2D 空间的高清连续概率场决策边界。
    支持叠加交互式探针点 (Probe Point)。
    """
    margin = 0.3
    x_min, x_max = X[:, 0].min() - margin, X[:, 0].max() + margin
    y_min, y_max = X[:, 1].min() - margin, X[:, 1].max() + margin

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, resolution),
        np.linspace(y_min, y_max, resolution),
    )
    grid = np.c_[xx.ravel(), yy.ravel()]

    preds = model.predict(grid)
    if preds.shape[1] == 1:
        zz = preds.reshape(xx.shape)
        colorscale = [
            [0.0, "rgba(56, 189, 248, 0.45)"],   # 类别 0 (青蓝)
            [0.5, "rgba(15, 23, 42, 0.15)"],     # 决策临界线 (深空)
            [1.0, "rgba(251, 113, 133, 0.45)"],  # 类别 1 (珊瑚红)
        ]
    else:
        zz = np.argmax(preds, axis=1).reshape(xx.shape).astype(float)
        colorscale = "Viridis"

    fig = go.Figure()

    # 连续概率等高面
    fig.add_trace(go.Contour(
        x=np.linspace(x_min, x_max, resolution),
        y=np.linspace(y_min, y_max, resolution),
        z=zz,
        colorscale=colorscale,
        showscale=False,
        contours=dict(showlines=True, coloring="fill"),
        line=dict(width=1, color="rgba(255,255,255,0.15)"),
        hoverinfo="skip",
    ))

    # 训练数据散点
    labels = y.ravel() if y.shape[1] == 1 else np.argmax(y, axis=1)
    unique_classes = sorted(set(labels.astype(int)))

    for cls_idx in unique_classes:
        mask = labels == cls_idx
        color = CYBER_PALETTE["classes"][cls_idx % len(CYBER_PALETTE["classes"])]
        fig.add_trace(go.Scatter(
            x=X[mask, 0],
            y=X[mask, 1],
            mode="markers",
            name=f"Class {cls_idx}",
            marker=dict(
                size=7,
                color=color,
                line=dict(width=1.5, color="rgba(255,255,255,0.85)"),
                opacity=0.9,
            ),
            hovertemplate=(
                f"<b>Class {cls_idx}</b><br>"
                + "x₁: %{x:.3f}<br>"
                + "x₂: %{y:.3f}<extra></extra>"
            ),
        ))

    # 动态探针样本点 (如果存在)
    if probe_point is not None:
        px, py = probe_point
        fig.add_trace(go.Scatter(
            x=[px],
            y=[py],
            mode="markers+text",
            name="PROBE POINT",
            text=["PROBE"],
            textposition="top center",
            textfont=dict(color="#fbbf24", size=10, family="JetBrains Mono"),
            marker=dict(
                size=14,
                color="#fbbf24",
                symbol="cross",
                line=dict(width=2.0, color="#ffffff"),
            ),
            hovertemplate="<b>PROBE POINT // 活性探针点</b><br>x₁: %{x:.3f}<br>x₂: %{y:.3f}<extra></extra>",
        ))

    fig.update_layout(
        xaxis_title="Feature x₁",
        yaxis_title="Feature x₂",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(15,23,42,0.6)",
            bordercolor="rgba(255,255,255,0.1)",
            borderwidth=1,
        ),
    )

    return _apply_cyber_theme(fig, title)


# ---------------------------------------------------------------------------
# 2. 损失与准确率曲线 (Loss & Accuracy)
# ---------------------------------------------------------------------------
def plot_loss_curve(
    history: dict[str, list[float]],
    title: str = "TRAINING DYNAMICS // 损失与准确率收敛",
) -> go.Figure:
    """绘制带极光填充与双指标监控的训练收敛图"""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("LOSS CONVERGENCE", "ACCURACY PROGRESSION"),
        horizontal_spacing=0.12,
    )

    epochs = list(range(1, len(history.get("loss", [])) + 1))

    # 1. 损失曲线
    if "loss" in history and history["loss"]:
        losses = history["loss"]
        min_idx = int(np.argmin(losses))
        fig.add_trace(
            go.Scatter(
                x=epochs, y=losses,
                mode="lines",
                name="Train Loss",
                line=dict(color=CYBER_PALETTE["primary"], width=2.5),
                fill="tozeroy",
                fillcolor="rgba(56, 189, 248, 0.08)",
                hovertemplate="Epoch %{x}: Loss = %{y:.4f}<extra></extra>",
            ),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=[epochs[min_idx]], y=[losses[min_idx]],
                mode="markers",
                name="Min Loss",
                marker=dict(size=8, color=CYBER_PALETTE["accent"], symbol="diamond"),
                hovertemplate=f"Min Loss: {losses[min_idx]:.4f} (Epoch {epochs[min_idx]})<extra></extra>",
                showlegend=False,
            ),
            row=1, col=1,
        )

    # 2. 准确率曲线
    if "accuracy" in history and history["accuracy"]:
        accs = history["accuracy"]
        fig.add_trace(
            go.Scatter(
                x=epochs, y=accs,
                mode="lines",
                name="Train Acc",
                line=dict(color=CYBER_PALETTE["accent"], width=2.5),
                fill="tozeroy",
                fillcolor="rgba(52, 211, 153, 0.08)",
                hovertemplate="Epoch %{x}: Acc = %{y:.2%}<extra></extra>",
            ),
            row=1, col=2,
        )

    fig.update_xaxes(title_text="Epoch", row=1, col=1, gridcolor=CYBER_PALETTE["grid"])
    fig.update_xaxes(title_text="Epoch", row=1, col=2, gridcolor=CYBER_PALETTE["grid"])
    fig.update_yaxes(title_text="Loss", row=1, col=1, gridcolor=CYBER_PALETTE["grid"])
    fig.update_yaxes(title_text="Accuracy", row=1, col=2, gridcolor=CYBER_PALETTE["grid"], range=[0, 1.05])

    return _apply_cyber_theme(fig, title)


# ---------------------------------------------------------------------------
# 3. 逐层激活热力图 (Activation Heatmap)
# ---------------------------------------------------------------------------
def plot_activation_heatmap(
    activations: list[np.ndarray],
    title: str = "ACTIVATION HEATMAP // 逐层神经元激活分布",
) -> go.Figure:
    """绘制各层神经元输出的激活热力矩阵"""
    n_layers = len(activations)
    fig = make_subplots(
        rows=1, cols=n_layers,
        subplot_titles=[f"Layer {i+1} (dim={act.shape[1]})" for i, act in enumerate(activations)],
        horizontal_spacing=0.06,
    )

    for idx, act in enumerate(activations):
        sample_act = act[:30] if act.shape[0] > 30 else act
        fig.add_trace(
            go.Heatmap(
                z=sample_act,
                colorscale="Viridis",
                showscale=(idx == n_layers - 1),
                colorbar=dict(title=dict(text="Activation", side="right")),
                hovertemplate="Sample: %{y}<br>Neuron: %{x}<br>Value: %{z:.3f}<extra></extra>",
            ),
            row=1, col=idx + 1,
        )
        fig.update_xaxes(title_text="Neuron", row=1, col=idx + 1, gridcolor="rgba(255,255,255,0.03)")
        fig.update_yaxes(title_text="Sample" if idx == 0 else "", row=1, col=idx + 1, gridcolor="rgba(255,255,255,0.03)")

    return _apply_cyber_theme(fig, title)


# ---------------------------------------------------------------------------
# 4. 梯度流形直方图 (Gradient Histogram)
# ---------------------------------------------------------------------------
def plot_gradient_histograms(
    gradients: list[np.ndarray],
    layer_names: list[str],
    title: str = "GRADIENT FLOW // 反向传播梯度流分布",
) -> go.Figure:
    """绘制各层权重的梯度分布直方图"""
    fig = go.Figure()

    colors = [CYBER_PALETTE["primary"], CYBER_PALETTE["purple"], CYBER_PALETTE["warning"], CYBER_PALETTE["secondary"]]

    for idx, (grad, name) in enumerate(zip(gradients, layer_names, strict=False)):
        vals = grad.ravel()
        color = colors[idx % len(colors)]
        fig.add_trace(go.Histogram(
            x=vals,
            name=name,
            opacity=0.65,
            marker_color=color,
            nbinsx=40,
            hovertemplate=f"<b>{name}</b><br>Gradient Range: %{{x:.4f}}<br>Count: %{{y}}<extra></extra>",
        ))

    fig.update_layout(
        barmode="overlay",
        xaxis_title="Gradient ∂L/∂W",
        yaxis_title="Frequency",
        legend=dict(orientation="h", y=1.08, x=1, xanchor="right"),
    )

    return _apply_cyber_theme(fig, title)


# ---------------------------------------------------------------------------
# 5. 权重分布直方图 (Weight Histogram)
# ---------------------------------------------------------------------------
def plot_weight_histograms(
    weights: list[np.ndarray],
    layer_names: list[str],
    title: str = "WEIGHT SPECTRUM // 各层权重参数分布",
) -> go.Figure:
    """绘制各层权重的参数分布"""
    fig = go.Figure()
    colors = [CYBER_PALETTE["accent"], CYBER_PALETTE["primary"], CYBER_PALETTE["purple"], CYBER_PALETTE["warning"]]

    for idx, (w, name) in enumerate(zip(weights, layer_names, strict=False)):
        vals = w.ravel()
        color = colors[idx % len(colors)]
        fig.add_trace(go.Histogram(
            x=vals,
            name=name,
            opacity=0.65,
            marker_color=color,
            nbinsx=40,
            hovertemplate=f"<b>{name}</b><br>Weight Value: %{{x:.4f}}<br>Count: %{{y}}<extra></extra>",
        ))

    fig.update_layout(
        barmode="overlay",
        xaxis_title="Weight Parameter W",
        yaxis_title="Frequency",
        legend=dict(orientation="h", y=1.08, x=1, xanchor="right"),
    )

    return _apply_cyber_theme(fig, title)


# ---------------------------------------------------------------------------
# 6. 多优化器竞速对比 (Multi-Optimizer Curves)
# ---------------------------------------------------------------------------
def plot_multi_loss_curves(
    histories: dict[str, dict[str, list[float]]],
    title: str = "OPTIMIZER BENCHMARK // 优化器多轨收敛对比",
) -> go.Figure:
    """绘制多种优化器同屏收敛速度对比"""
    fig = go.Figure()

    for name, hist in histories.items():
        losses = hist.get("loss", [])
        if not losses:
            continue
        epochs = list(range(1, len(losses) + 1))
        color = CYBER_PALETTE["optimizers"].get(name, CYBER_PALETTE["primary"])

        fig.add_trace(go.Scatter(
            x=epochs,
            y=losses,
            mode="lines",
            name=name,
            line=dict(color=color, width=2.5),
            hovertemplate=f"<b>{name}</b><br>Epoch: %{{x}}<br>Loss: %{{y:.4f}}<extra></extra>",
        ))

    fig.update_layout(
        xaxis_title="Epoch",
        yaxis_title="Loss (Log Scale)",
        yaxis_type="log",
        legend=dict(
            orientation="h",
            y=1.08,
            x=1,
            xanchor="right",
            bgcolor="rgba(15,23,42,0.6)",
            bordercolor="rgba(255,255,255,0.1)",
        ),
    )

    return _apply_cyber_theme(fig, title)


# ---------------------------------------------------------------------------
# 7. 权重优化空间轨迹 (Weight Trajectory)
# ---------------------------------------------------------------------------
def plot_weight_trajectory(
    trajectory: list[np.ndarray],
    title: str = "PARAMETER TRAJECTORY // 参数空间梯度搜索路径",
) -> go.Figure:
    """绘制 2D 参数空间中的优化轨迹"""
    fig = go.Figure()

    if len(trajectory) > 0 and trajectory[0].size >= 2:
        w1_vals = [float(w.ravel()[0]) for w in trajectory]
        w2_vals = [float(w.ravel()[1]) for w in trajectory]

        fig.add_trace(go.Scatter(
            x=w1_vals, y=w2_vals,
            mode="lines+markers",
            name="Path",
            line=dict(color=CYBER_PALETTE["primary"], width=2.5),
            marker=dict(
                size=5,
                color=list(range(len(w1_vals))),
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title=dict(text="Step")),
            ),
            hovertemplate="Step %{marker.color}: w₁=%{x:.3f}, w₂=%{y:.3f}<extra></extra>",
        ))

        fig.add_trace(go.Scatter(
            x=[w1_vals[0]], y=[w2_vals[0]],
            mode="markers+text",
            name="Origin",
            text=["START"],
            textposition="bottom right",
            marker=dict(size=11, color=CYBER_PALETTE["secondary"], symbol="circle"),
        ))
        fig.add_trace(go.Scatter(
            x=[w1_vals[-1]], y=[w2_vals[-1]],
            mode="markers+text",
            name="Optimal",
            text=["FINAL"],
            textposition="top right",
            marker=dict(size=12, color=CYBER_PALETTE["accent"], symbol="diamond"),
        ))

    fig.update_layout(
        xaxis_title="Parameter w₁",
        yaxis_title="Parameter w₂",
    )

    return _apply_cyber_theme(fig, title)
