# Copyright (c) 2026 Yy1 (yuan278501381) | MIT License
"""
dashboard.components.charts - 世界级 Plotly 可视化图表引擎 (现代亮色极客风格 · 无重叠排版)

提供高对比度、清晰透亮的 Plotly 亮色图表系统（plotly_white）：
- 连续平滑概率场决策边界（支持探针样本点高亮）
- 发光 Loss 曲线与最小损失标注
- 多层梯度与权重流形直方图
- 神经元逐层激活强度热力图
- 优化器多轨竞速轨迹图
- 参数空间梯度寻优轨迹（严格防图例与色条重叠布局）
"""

from typing import Any

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ---------------------------------------------------------------------------
# 全局亮色视觉调色板 (Light Mode Palette)
# ---------------------------------------------------------------------------
LIGHT_PALETTE = {
    "bg_plot": "#ffffff",
    "bg_paper": "rgba(0, 0, 0, 0)",
    "grid": "rgba(15, 23, 42, 0.06)",
    "zero_line": "rgba(15, 23, 42, 0.12)",
    "font_color": "#0f172a",
    "font_muted": "#64748b",
    "primary": "#1d4ed8",     # 纯正皇家蓝
    "secondary": "#be123c",   # 玫瑰红
    "accent": "#047857",      # 森林翡翠绿
    "warning": "#b45309",     # 琥珀深橙
    "purple": "#6d28d9",      # 紫罗兰
    "classes": ["#1d4ed8", "#be123c", "#047857", "#b45309", "#6d28d9"],
    "optimizers": {
        "SGD": "#64748b",
        "Momentum": "#1d4ed8",
        "RMSProp": "#b45309",
        "Adam": "#047857",
    },
}


def _apply_light_theme(fig: go.Figure, title: str | None = None) -> go.Figure:
    """统一注入现代极简亮色图表布局属性"""
    layout_update: dict[str, Any] = {
        "template": "plotly_white",
        "plot_bgcolor": LIGHT_PALETTE["bg_plot"],
        "paper_bgcolor": LIGHT_PALETTE["bg_paper"],
        "font": dict(
            family="Plus Jakarta Sans, -apple-system, Segoe UI, sans-serif",
            size=12,
            color=LIGHT_PALETTE["font_color"],
        ),
        "margin": dict(l=40, r=40, t=55 if title else 30, b=40),
        "hoverlabel": dict(
            bgcolor="#ffffff",
            bordercolor="#cbd5e1",
            font=dict(family="JetBrains Mono, monospace", size=12, color="#0f172a"),
        ),
        "xaxis": dict(
            gridcolor=LIGHT_PALETTE["grid"],
            zerolinecolor=LIGHT_PALETTE["zero_line"],
            tickfont=dict(size=10, color=LIGHT_PALETTE["font_muted"]),
        ),
        "yaxis": dict(
            gridcolor=LIGHT_PALETTE["grid"],
            zerolinecolor=LIGHT_PALETTE["zero_line"],
            tickfont=dict(size=10, color=LIGHT_PALETTE["font_muted"]),
        ),
    }

    if title:
        layout_update["title"] = dict(
            text=f"<b>{title}</b>",
            font=dict(size=13, color="#0f172a"),
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
    """绘制模型在 2D 空间的亮色连续概率场决策边界"""
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
            [0.0, "rgba(29, 78, 216, 0.22)"],   # 类别 0 (蓝)
            [0.5, "rgba(241, 245, 249, 0.5)"],   # 决策临界线
            [1.0, "rgba(190, 18, 60, 0.22)"],    # 类别 1 (红)
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
        line=dict(width=1, color="rgba(15,23,42,0.1)"),
        hoverinfo="skip",
    ))

    # 训练数据散点
    labels = y.ravel() if y.shape[1] == 1 else np.argmax(y, axis=1)
    unique_classes = sorted(set(labels.astype(int)))

    for cls_idx in unique_classes:
        mask = labels == cls_idx
        color = LIGHT_PALETTE["classes"][cls_idx % len(LIGHT_PALETTE["classes"])]
        fig.add_trace(go.Scatter(
            x=X[mask, 0],
            y=X[mask, 1],
            mode="markers",
            name=f"Class {cls_idx}",
            marker=dict(
                size=8,
                color=color,
                line=dict(width=1.5, color="#ffffff"),
                opacity=0.9,
            ),
            hovertemplate=(
                f"<b>Class {cls_idx}</b><br>"
                + "x₁: %{x:.3f}<br>"
                + "x₂: %{y:.3f}<extra></extra>"
            ),
        ))

    # 动态探针样本点
    if probe_point is not None:
        px, py = probe_point
        fig.add_trace(go.Scatter(
            x=[px],
            y=[py],
            mode="markers+text",
            name="PROBE POINT",
            text=["PROBE"],
            textposition="top center",
            textfont=dict(color="#b45309", size=10, family="JetBrains Mono"),
            marker=dict(
                size=15,
                color="#b45309",
                symbol="cross",
                line=dict(width=2.5, color="#ffffff"),
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
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="#e2e8f0",
            borderwidth=1,
        ),
    )

    return _apply_light_theme(fig, title)


# ---------------------------------------------------------------------------
# 2. 损失与准确率曲线 (Loss & Accuracy)
# ---------------------------------------------------------------------------
def plot_loss_curve(
    history: dict[str, list[float]],
    title: str = "TRAINING DYNAMICS // 损失与准确率收敛",
) -> go.Figure:
    """绘制亮色训练收敛图"""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("LOSS CONVERGENCE // 损失下降", "ACCURACY PROGRESSION // 准确率提升"),
        horizontal_spacing=0.12,
    )

    epochs = list(range(1, len(history.get("loss", [])) + 1))

    if "loss" in history and history["loss"]:
        losses = history["loss"]
        min_idx = int(np.argmin(losses))
        fig.add_trace(
            go.Scatter(
                x=epochs, y=losses,
                mode="lines",
                name="Train Loss",
                line=dict(color=LIGHT_PALETTE["primary"], width=2.5),
                fill="tozeroy",
                fillcolor="rgba(29, 78, 216, 0.06)",
                hovertemplate="Epoch %{x}: Loss = %{y:.4f}<extra></extra>",
            ),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=[epochs[min_idx]], y=[losses[min_idx]],
                mode="markers",
                name="Min Loss",
                marker=dict(size=8, color=LIGHT_PALETTE["accent"], symbol="diamond"),
                hovertemplate=f"Min Loss: {losses[min_idx]:.4f} (Epoch {epochs[min_idx]})<extra></extra>",
                showlegend=False,
            ),
            row=1, col=1,
        )

    if "accuracy" in history and history["accuracy"]:
        accs = history["accuracy"]
        fig.add_trace(
            go.Scatter(
                x=epochs, y=accs,
                mode="lines",
                name="Train Acc",
                line=dict(color=LIGHT_PALETTE["accent"], width=2.5),
                fill="tozeroy",
                fillcolor="rgba(4, 120, 87, 0.06)",
                hovertemplate="Epoch %{x}: Acc = %{y:.2%}<extra></extra>",
            ),
            row=1, col=2,
        )

    fig.update_xaxes(title_text="Epoch", row=1, col=1, gridcolor=LIGHT_PALETTE["grid"])
    fig.update_xaxes(title_text="Epoch", row=1, col=2, gridcolor=LIGHT_PALETTE["grid"])
    fig.update_yaxes(title_text="Loss", row=1, col=1, gridcolor=LIGHT_PALETTE["grid"])
    fig.update_yaxes(title_text="Accuracy", row=1, col=2, gridcolor=LIGHT_PALETTE["grid"], range=[0, 1.05])

    return _apply_light_theme(fig, title)


# ---------------------------------------------------------------------------
# 3. 逐层激活热力图 (Activation Heatmap)
# ---------------------------------------------------------------------------
def plot_activation_heatmap(
    activations: list[np.ndarray],
    title: str = "ACTIVATION HEATMAP // 逐层神经元激活分布",
) -> go.Figure:
    """绘制亮色各层神经元激活热力矩阵"""
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
                colorscale="Blues",
                showscale=(idx == n_layers - 1),
                colorbar=dict(
                    title=dict(text="Activation", side="right", font=dict(size=10, color="#0f172a")),
                    x=1.02,
                    thickness=12,
                    len=0.85,
                    y=0.45,
                ),
                hovertemplate="Sample: %{y}<br>Neuron: %{x}<br>Value: %{z:.3f}<extra></extra>",
            ),
            row=1, col=idx + 1,
        )
        fig.update_xaxes(title_text="Neuron", row=1, col=idx + 1, gridcolor=LIGHT_PALETTE["grid"])
        fig.update_yaxes(title_text="Sample" if idx == 0 else "", row=1, col=idx + 1, gridcolor=LIGHT_PALETTE["grid"])

    return _apply_light_theme(fig, title)


# ---------------------------------------------------------------------------
# 4. 梯度流形直方图 (Gradient Histogram)
# ---------------------------------------------------------------------------
def plot_gradient_histograms(
    gradients: list[np.ndarray],
    layer_names: list[str],
    title: str = "GRADIENT FLOW // 反向传播梯度流分布",
) -> go.Figure:
    """绘制各层梯度的亮色直方图"""
    fig = go.Figure()

    colors = [LIGHT_PALETTE["primary"], LIGHT_PALETTE["purple"], LIGHT_PALETTE["warning"], LIGHT_PALETTE["secondary"]]

    for idx, (grad, name) in enumerate(zip(gradients, layer_names, strict=False)):
        vals = grad.ravel()
        color = colors[idx % len(colors)]
        fig.add_trace(go.Histogram(
            x=vals,
            name=name,
            opacity=0.6,
            marker_color=color,
            nbinsx=40,
            hovertemplate=f"<b>{name}</b><br>Gradient: %{{x:.4f}}<br>Count: %{{y}}<extra></extra>",
        ))

    fig.update_layout(
        barmode="overlay",
        xaxis_title="Gradient ∂L/∂W",
        yaxis_title="Frequency",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="#e2e8f0",
            borderwidth=1,
        ),
    )

    return _apply_light_theme(fig, title)


# ---------------------------------------------------------------------------
# 5. 权重分布直方图 (Weight Histogram)
# ---------------------------------------------------------------------------
def plot_weight_histograms(
    weights: list[np.ndarray],
    layer_names: list[str],
    title: str = "WEIGHT SPECTRUM // 各层权重参数分布",
) -> go.Figure:
    """绘制各层权重的亮色直方图"""
    fig = go.Figure()
    colors = [LIGHT_PALETTE["accent"], LIGHT_PALETTE["primary"], LIGHT_PALETTE["purple"], LIGHT_PALETTE["warning"]]

    for idx, (w, name) in enumerate(zip(weights, layer_names, strict=False)):
        vals = w.ravel()
        color = colors[idx % len(colors)]
        fig.add_trace(go.Histogram(
            x=vals,
            name=name,
            opacity=0.6,
            marker_color=color,
            nbinsx=40,
            hovertemplate=f"<b>{name}</b><br>Weight: %{{x:.4f}}<br>Count: %{{y}}<extra></extra>",
        ))

    fig.update_layout(
        barmode="overlay",
        xaxis_title="Weight Parameter W",
        yaxis_title="Frequency",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="#e2e8f0",
            borderwidth=1,
        ),
    )

    return _apply_light_theme(fig, title)


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
        clean_key = name.split(" ")[0]
        color = LIGHT_PALETTE["optimizers"].get(clean_key, LIGHT_PALETTE["primary"])

        fig.add_trace(go.Scatter(
            x=epochs,
            y=losses,
            mode="lines",
            name=name,
            line=dict(color=color, width=2.5),
            hovertemplate=f"<b>{name}</b><br>Epoch: %{{x}}<br>Loss: %{{y:.4f}}<extra></extra>",
        ))

    fig.update_layout(
        xaxis_title="Epoch 训练轮次",
        yaxis_title="Loss (Log 对数刻度)",
        yaxis_type="log",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="#e2e8f0",
            borderwidth=1,
        ),
    )

    return _apply_light_theme(fig, title)


# ---------------------------------------------------------------------------
# 7. 权重优化空间轨迹 (Weight Trajectory - 彻底防重叠排版)
# ---------------------------------------------------------------------------
def plot_weight_trajectory(
    trajectory: list[np.ndarray],
    title: str = "PARAMETER TRAJECTORY // 参数空间梯度搜索路径",
) -> go.Figure:
    """绘制 2D 参数空间中的亮色优化轨迹 (严格隔离图例与颜色条)"""
    fig = go.Figure()

    if len(trajectory) > 0 and trajectory[0].size >= 2:
        w1_vals = [float(w.ravel()[0]) for w in trajectory]
        w2_vals = [float(w.ravel()[1]) for w in trajectory]

        fig.add_trace(go.Scatter(
            x=w1_vals, y=w2_vals,
            mode="lines+markers",
            name="Path (搜索路径)",
            line=dict(color=LIGHT_PALETTE["primary"], width=2.5),
            marker=dict(
                size=5,
                color=list(range(len(w1_vals))),
                colorscale="Blues",
                showscale=True,
                colorbar=dict(
                    title=dict(text="Step // 步数", font=dict(size=11, color="#0f172a")),
                    x=1.03,
                    thickness=14,
                    len=0.85,
                    y=0.45,
                    tickfont=dict(size=10, color="#64748b"),
                ),
            ),
            hovertemplate="Step %{marker.color}: w₁=%{x:.3f}, w₂=%{y:.3f}<extra></extra>",
        ))

        fig.add_trace(go.Scatter(
            x=[w1_vals[0]], y=[w2_vals[0]],
            mode="markers+text",
            name="Start (初始点)",
            text=["START"],
            textposition="bottom right",
            textfont=dict(color="#0f172a", family="JetBrains Mono", size=10),
            marker=dict(size=11, color=LIGHT_PALETTE["secondary"], symbol="circle"),
        ))
        fig.add_trace(go.Scatter(
            x=[w1_vals[-1]], y=[w2_vals[-1]],
            mode="markers+text",
            name="Optimal (当前极优点)",
            text=["FINAL"],
            textposition="top right",
            textfont=dict(color="#0f172a", family="JetBrains Mono", size=10),
            marker=dict(size=12, color=LIGHT_PALETTE["accent"], symbol="diamond"),
        ))

    fig.update_layout(
        xaxis_title="Parameter w₁ (权重参数 1)",
        yaxis_title="Parameter w₂ (权重参数 2)",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="#e2e8f0",
            borderwidth=1,
        ),
        margin=dict(l=40, r=70, t=55, b=40),
    )

    return _apply_light_theme(fig, title)
