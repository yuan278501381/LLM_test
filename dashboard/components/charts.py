# Copyright (c) 2026 Yy1 (yuan278501381) | MIT License
"""
dashboard.components.charts - 世界级 Plotly 可视化图表引擎 (2026 工业级设计标准 · 零重叠防碰撞)

提供高对比度、清晰透亮、无重叠排版的 Plotly 亮色图表系统（plotly_white）：
- 连续平滑概率场决策边界（支持探针样本点发光环）
- 贝塞尔平滑 Loss 曲线与最小损失标注
- 多层梯度与权重流形直方图
- 优化器多轨竞速对比
- 权重参数空间寻优轨迹 (严格防图例与色条重叠)
- 3D 词嵌入空间与语义平行四边形矢量流形
- 序列记忆衰减与注意力热力矩阵 (严格防顶部 X 轴与标题重叠)
- 下一词概率水平柱状图
"""

from typing import Any

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ---------------------------------------------------------------------------
# 全局亮色视觉调色板 (Light Mode Palette - 2026 Linear / Stripe 风格)
# ---------------------------------------------------------------------------
LIGHT_PALETTE = {
    "bg_plot": "#ffffff",
    "bg_paper": "rgba(0, 0, 0, 0)",
    "grid": "rgba(15, 23, 42, 0.05)",
    "zero_line": "rgba(15, 23, 42, 0.08)",
    "font_color": "#0f172a",
    "font_muted": "#64748b",
    "primary": "#1d4ed8",     # 纯正皇家蓝
    "secondary": "#be123c",   # 玫瑰红
    "accent": "#047857",      # 翡翠绿
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
    """统一注入现代极简亮色图表布局属性 (彻底杜绝文字与图例重叠)"""
    layout_update: dict[str, Any] = {
        "template": "plotly_white",
        "plot_bgcolor": LIGHT_PALETTE["bg_plot"],
        "paper_bgcolor": LIGHT_PALETTE["bg_paper"],
        "font": dict(
            family="JetBrains Mono, -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif",
            size=11,
            color=LIGHT_PALETTE["font_color"],
        ),
        "margin": dict(l=45, r=45, t=50 if title else 25, b=45),
        "hoverlabel": dict(
            bgcolor="#ffffff",
            bordercolor="#cbd5e1",
            font=dict(family="JetBrains Mono, monospace", size=11, color="#0f172a"),
        ),
        "xaxis": dict(
            gridcolor=LIGHT_PALETTE["grid"],
            zerolinecolor=LIGHT_PALETTE["zero_line"],
            tickfont=dict(size=10, color=LIGHT_PALETTE["font_muted"]),
            linecolor="rgba(15, 23, 42, 0.1)",
        ),
        "yaxis": dict(
            gridcolor=LIGHT_PALETTE["grid"],
            zerolinecolor=LIGHT_PALETTE["zero_line"],
            tickfont=dict(size=10, color=LIGHT_PALETTE["font_muted"]),
            linecolor="rgba(15, 23, 42, 0.1)",
        ),
    }

    if title:
        layout_update["title"] = dict(
            text=f"<b>{title}</b>",
            font=dict(size=12, color="#0f172a", family="JetBrains Mono"),
            x=0.01,
            y=0.98,
            xanchor="left",
            yanchor="top",
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
    title: str | None = None,
) -> go.Figure:
    """绘制模型在 2D 空间的亮色连续概率场决策边界 (图例置于底部，防碰撞)"""
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
            [0.5, "rgba(241, 245, 249, 0.6)"],   # 决策临界线
            [1.0, "rgba(190, 18, 60, 0.22)"],    # 类别 1 (红)
        ]
    else:
        zz = np.argmax(preds, axis=1).reshape(xx.shape).astype(float)
        colorscale = "Viridis"

    fig = go.Figure()

    # 连续概率等高面 (平滑淡色概率背景场)
    fig.add_trace(go.Contour(
        x=np.linspace(x_min, x_max, resolution),
        y=np.linspace(y_min, y_max, resolution),
        z=zz,
        colorscale=colorscale,
        showscale=False,
        contours=dict(showlines=False, coloring="fill"),
        hoverinfo="skip",
        showlegend=False,
    ))

    # 显式绘制唯一的决策分界线 (Decision Boundary: 概率临界线 P=0.5 / 直线方程 z=0)
    if preds.shape[1] == 1:
        fig.add_trace(go.Contour(
            x=np.linspace(x_min, x_max, resolution),
            y=np.linspace(y_min, y_max, resolution),
            z=zz,
            contours=dict(
                start=0.5,
                end=0.5,
                size=0,
                coloring="none",
                showlabels=False,
            ),
            line=dict(width=3.5, color="#0f172a", dash="solid"),
            name="决策分界线 (Line: P=0.5)",
            showlegend=True,
            hovertemplate="<b>[DECISION LINE] 决策分界线 (Decision Line: P=0.5)</b><extra></extra>",
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
            textfont=dict(color="#b45309", size=10, family="JetBrains Mono", weight="bold"),
            marker=dict(
                size=14,
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
            yanchor="top",
            y=-0.18,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(255,255,255,0.9)",
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
    title: str | None = None,
) -> go.Figure:
    """绘制亮色平滑训练收敛图 (严格防多列重叠)"""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("损失收敛 (Loss)", "准确率 (Accuracy)"),
        horizontal_spacing=0.14,
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
                line=dict(color=LIGHT_PALETTE["primary"], width=2.5, shape="spline", smoothing=1.1),
                fill="tozeroy",
                fillcolor="rgba(29, 78, 216, 0.05)",
                hovertemplate="轮数 %{x}: 损失 Loss = %{y:.4f}<extra></extra>",
            ),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=[epochs[min_idx]], y=[losses[min_idx]],
                mode="markers",
                name="Min Loss",
                marker=dict(size=8, color=LIGHT_PALETTE["accent"], symbol="diamond"),
                hovertemplate=f"最低损失: {losses[min_idx]:.4f} (第 {epochs[min_idx]} 轮)<extra></extra>",
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
                line=dict(color=LIGHT_PALETTE["accent"], width=2.5, shape="spline", smoothing=1.1),
                fill="tozeroy",
                fillcolor="rgba(4, 120, 87, 0.05)",
                hovertemplate="轮数 %{x}: 准确率 Acc = %{y:.2%}<extra></extra>",
            ),
            row=1, col=2,
        )

    fig.update_xaxes(title_text="训练轮数 (Epoch)", row=1, col=1, gridcolor=LIGHT_PALETTE["grid"])
    fig.update_xaxes(title_text="训练轮数 (Epoch)", row=1, col=2, gridcolor=LIGHT_PALETTE["grid"])
    fig.update_yaxes(title_text="损失误差 (Loss)", row=1, col=1, gridcolor=LIGHT_PALETTE["grid"])
    fig.update_yaxes(title_text="准确率 (Accuracy)", row=1, col=2, gridcolor=LIGHT_PALETTE["grid"], range=[0, 1.05])

    fig.update_layout(
        showlegend=False,
    )

    return _apply_light_theme(fig, title)


# ---------------------------------------------------------------------------
# 3. 逐层激活热力图 (Activation Heatmap)
# ---------------------------------------------------------------------------
def plot_activation_heatmap(
    activations: list[np.ndarray],
    title: str | None = None,
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
                    y=0.5,
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
    title: str | None = None,
) -> go.Figure:
    """绘制各层梯度的亮色直方图 (图例置于底部，防遮挡)"""
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
            yanchor="top",
            y=-0.2,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(255,255,255,0.9)",
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
    title: str | None = None,
) -> go.Figure:
    """绘制各层权重的亮色直方图 (图例置于底部)"""
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
            yanchor="top",
            y=-0.2,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(255,255,255,0.9)",
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
    title: str | None = None,
) -> go.Figure:
    """绘制多种优化器同屏收敛速度对比 (图例置于底部居中)"""
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
            line=dict(color=color, width=2.5, shape="spline", smoothing=1.1),
            hovertemplate=f"<b>{name}</b><br>Epoch: %{{x}}<br>Loss: %{{y:.4f}}<extra></extra>",
        ))

    fig.update_layout(
        xaxis_title="Epoch 训练轮次",
        yaxis_title="Loss (Log 对数刻度)",
        yaxis_type="log",
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.2,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="#e2e8f0",
            borderwidth=1,
        ),
    )

    return _apply_light_theme(fig, title)


# ---------------------------------------------------------------------------
# 7. 权重优化空间轨迹 (Weight Trajectory - 彻底杜绝文字重叠)
# ---------------------------------------------------------------------------
def plot_weight_trajectory(
    trajectory: list[np.ndarray],
    title: str | None = None,
) -> go.Figure:
    """绘制 2D 参数空间中的优化轨迹 (图例置于底部，Colorbar 独立靠右，零重叠)"""
    fig = go.Figure()

    if len(trajectory) > 0 and trajectory[0].size >= 2:
        w1_vals = [float(w.ravel()[0]) for w in trajectory]
        w2_vals = [float(w.ravel()[1]) for w in trajectory]

        fig.add_trace(go.Scatter(
            x=w1_vals, y=w2_vals,
            mode="lines+markers",
            name="Search Path (寻优轨迹)",
            line=dict(color=LIGHT_PALETTE["primary"], width=2.5),
            marker=dict(
                size=5,
                color=list(range(len(w1_vals))),
                colorscale="Blues",
                showscale=True,
                colorbar=dict(
                    title=dict(text="Step 步数", font=dict(size=10, color="#0f172a")),
                    x=1.02,
                    thickness=12,
                    len=0.8,
                    y=0.45,
                    tickfont=dict(size=9, color="#64748b"),
                ),
            ),
            hovertemplate="Step %{marker.color}: w₁=%{x:.3f}, w₂=%{y:.3f}<extra></extra>",
        ))

        fig.add_trace(go.Scatter(
            x=[w1_vals[0]], y=[w2_vals[0]],
            mode="markers+text",
            name="Start (起点)",
            text=["START"],
            textposition="bottom right",
            textfont=dict(color="#be123c", family="JetBrains Mono", size=10, weight="bold"),
            marker=dict(size=11, color=LIGHT_PALETTE["secondary"], symbol="circle"),
        ))
        fig.add_trace(go.Scatter(
            x=[w1_vals[-1]], y=[w2_vals[-1]],
            mode="markers+text",
            name="Optimal (极优点)",
            text=["FINAL"],
            textposition="top right",
            textfont=dict(color="#047857", family="JetBrains Mono", size=10, weight="bold"),
            marker=dict(size=12, color=LIGHT_PALETTE["accent"], symbol="diamond"),
        ))

    fig.update_layout(
        xaxis_title="Parameter w₁",
        yaxis_title="Parameter w₂",
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.22,
            xanchor="center",
            x=0.45,
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="#e2e8f0",
            borderwidth=1,
        ),
        margin=dict(l=45, r=80, t=50 if title else 20, b=55),
    )

    return _apply_light_theme(fig, title)


# ---------------------------------------------------------------------------
# 8. 词嵌入空间 (Word Embedding Space)
# ---------------------------------------------------------------------------
def plot_embedding_space(
    words: list[str],
    vectors: np.ndarray,
    highlight_words: list[str] | None = None,
    arithmetic: dict | None = None,
    title: str | None = None,
) -> go.Figure:
    """绘制 3D 词嵌入流形空间 (支持高亮与几何平行四边形)"""
    from sklearn.decomposition import PCA

    # 降维到 3D
    if vectors.shape[1] > 3:
        pca = PCA(n_components=3)
        vecs_3d = pca.fit_transform(vectors)
    elif vectors.shape[1] == 2:
        vecs_3d = np.c_[vectors, np.zeros(vectors.shape[0])]
    else:
        vecs_3d = vectors

    fig = go.Figure()

    # 背景词
    mask_bg = np.ones(len(words), dtype=bool)
    if highlight_words:
        for i, w in enumerate(words):
            if w in highlight_words or (arithmetic and w in arithmetic.values()):
                mask_bg[i] = False

    if np.any(mask_bg):
        fig.add_trace(go.Scatter3d(
            x=vecs_3d[mask_bg, 0],
            y=vecs_3d[mask_bg, 1],
            z=vecs_3d[mask_bg, 2],
            mode="markers",
            name="背景词 (Vocabulary)",
            text=np.array(words)[mask_bg],
            marker=dict(size=4, color="rgba(100, 116, 139, 0.4)"),
            hovertemplate="Word: %{text}<extra></extra>",
        ))

    # 高亮词
    if highlight_words:
        for hw in highlight_words:
            if hw in words:
                idx = words.index(hw)
                fig.add_trace(go.Scatter3d(
                    x=[vecs_3d[idx, 0]],
                    y=[vecs_3d[idx, 1]],
                    z=[vecs_3d[idx, 2]],
                    mode="markers+text",
                    name=hw,
                    text=[hw],
                    textposition="top center",
                    textfont=dict(size=11, color=LIGHT_PALETTE["primary"], family="JetBrains Mono", weight="bold"),
                    marker=dict(size=8, color=LIGHT_PALETTE["primary"]),
                    hovertemplate="Word: %{text}<extra></extra>",
                ))

    # 算术矢量平行四边形 A - B + C = Result
    if arithmetic:
        A, B, C, R = arithmetic.get("A"), arithmetic.get("B"), arithmetic.get("C"), arithmetic.get("Result")
        if all(w in words for w in [A, B, C, R]):
            idx_a, idx_b, idx_c, idx_r = words.index(A), words.index(B), words.index(C), words.index(R)
            
            # 画虚线 B -> A 和 C -> R
            fig.add_trace(go.Scatter3d(
                x=[vecs_3d[idx_b, 0], vecs_3d[idx_a, 0]],
                y=[vecs_3d[idx_b, 1], vecs_3d[idx_a, 1]],
                z=[vecs_3d[idx_b, 2], vecs_3d[idx_a, 2]],
                mode="lines",
                name=f"{B} -> {A}",
                line=dict(color=LIGHT_PALETTE["accent"], width=4, dash="dash"),
            ))
            fig.add_trace(go.Scatter3d(
                x=[vecs_3d[idx_c, 0], vecs_3d[idx_r, 0]],
                y=[vecs_3d[idx_c, 1], vecs_3d[idx_r, 1]],
                z=[vecs_3d[idx_c, 2], vecs_3d[idx_r, 2]],
                mode="lines",
                name=f"{C} -> {R}",
                line=dict(color=LIGHT_PALETTE["accent"], width=4, dash="dash"),
            ))
            
            # 标注关键 4 词
            for w, idx, color in zip([A, B, C, R], [idx_a, idx_b, idx_c, idx_r], [LIGHT_PALETTE["primary"]]*3 + [LIGHT_PALETTE["secondary"]]):
                fig.add_trace(go.Scatter3d(
                    x=[vecs_3d[idx, 0]],
                    y=[vecs_3d[idx, 1]],
                    z=[vecs_3d[idx, 2]],
                    mode="markers+text",
                    name=w,
                    text=[w],
                    textposition="top center",
                    textfont=dict(size=12, color=color, family="JetBrains Mono", weight="bold"),
                    marker=dict(size=9, color=color),
                ))

    fig.update_layout(
        scene=dict(
            xaxis_title="Dim 1",
            yaxis_title="Dim 2",
            zaxis_title="Dim 3",
            xaxis=dict(showbackground=False, gridcolor=LIGHT_PALETTE["grid"]),
            yaxis=dict(showbackground=False, gridcolor=LIGHT_PALETTE["grid"]),
            zaxis=dict(showbackground=False, gridcolor=LIGHT_PALETTE["grid"]),
        ),
        margin=dict(l=0, r=0, t=30 if title else 0, b=0),
        showlegend=False,
    )
    return _apply_light_theme(fig, title)


# ---------------------------------------------------------------------------
# 9. 记忆衰减热力图 (Memory Decay Heatmap)
# ---------------------------------------------------------------------------
def plot_memory_decay_heatmap(
    hidden_states: list[np.ndarray],
    tokens: list[str],
    title: str | None = None,
) -> go.Figure:
    """绘制 RNN 序列记忆强度的衰减热力图 (带微白间隔网格，彻底防重叠)"""
    n_steps = len(hidden_states)
    
    memory_matrix = np.zeros((n_steps, n_steps))
    for i in range(n_steps):
        for j in range(i + 1):
            v_i = hidden_states[i].ravel()
            v_j = hidden_states[j].ravel()
            norm_i = np.linalg.norm(v_i) + 1e-8
            norm_j = np.linalg.norm(v_j) + 1e-8
            memory_matrix[i, j] = np.abs(np.dot(v_i, v_j) / (norm_i * norm_j))

    memory_matrix[np.triu_indices(n_steps, 1)] = np.nan

    fig = go.Figure(data=go.Heatmap(
        z=memory_matrix,
        x=tokens,
        y=tokens,
        colorscale="Blues",
        showscale=True,
        xgap=2,
        ygap=2,
        colorbar=dict(
            title=dict(text="Memory", font=dict(size=10, color="#0f172a")),
            thickness=12,
            len=0.85,
            x=1.02,
        ),
        hovertemplate="当前词 (Query): <b>%{y}</b><br>历史词 (Key): <b>%{x}</b><br>记忆强度: <b>%{z:.3f}</b><extra></extra>",
    ))
    
    fig.update_layout(
        xaxis=dict(
            side="bottom",
            title=dict(text="历史词汇 (History Keys)", font=dict(size=11, color="#64748b")),
            tickangle=-25 if len(tokens) > 6 else 0,
        ),
        yaxis=dict(
            autorange="reversed",
            title=dict(text="当前步 (Current Query)", font=dict(size=11, color="#64748b")),
        ),
        margin=dict(l=60, r=60, t=50 if title else 25, b=50),
    )
    return _apply_light_theme(fig, title)


# ---------------------------------------------------------------------------
# 10. 注意力机制热力图 (Attention Heatmap - 彻底杜绝顶部文字挤压与重叠)
# ---------------------------------------------------------------------------
def plot_attention_heatmap_nlp(
    attention_weights: np.ndarray,
    tokens_x: list[str],
    tokens_y: list[str],
    title: str | None = None,
) -> go.Figure:
    """
    绘制 N×N 注意力权重热力图 (2026 工业级防重叠排版)。
    
    关键防撞车设计：
    - X 轴刻度置于顶部方便阅读，但免除顶部冗长大标题文字撞车
    - 垂直预留充足 margin，防止 title 与顶部词汇重合
    - 丰富的 Hover 悬停气泡完整呈现 Query -> Key 关系与精准百分比
    """
    fig = go.Figure(data=go.Heatmap(
        z=attention_weights,
        x=tokens_x,
        y=tokens_y,
        colorscale="Blues",
        showscale=True,
        xgap=2,
        ygap=2,
        colorbar=dict(
            title=dict(text="Weight", font=dict(size=10, color="#0f172a")),
            thickness=11,
            len=0.85,
            x=1.02,
            y=0.48,
        ),
        zmin=0.0,
        zmax=1.0,
        hovertemplate="Query (生成词): <b>%{y}</b><br>Key (关注词): <b>%{x}</b><br>Attention 权重: <b>%{z:.2%}</b><extra></extra>",
    ))

    # 根据序列长度自适应刻度倾斜角度
    tick_angle = -30 if len(tokens_x) > 6 else 0

    fig.update_layout(
        xaxis=dict(
            side="top",
            tickangle=tick_angle,
            tickfont=dict(size=10, family="JetBrains Mono", color="#334155"),
            # 顶部不放长标题，完全避免与图表主标题和刻度挤在一起
            title=None,
        ),
        yaxis=dict(
            autorange="reversed",
            tickfont=dict(size=10, family="JetBrains Mono", color="#334155"),
            title=dict(text="Queries", font=dict(size=10, color="#64748b")),
        ),
        # 顶部留足空间放 title 和顶部词汇刻度
        margin=dict(l=60, r=60, t=65 if title else 45, b=30),
    )
    return _apply_light_theme(fig, title)


# ---------------------------------------------------------------------------
# 11. 下一词概率柱状图 (Token Probabilities)
# ---------------------------------------------------------------------------
def plot_token_probabilities(
    token_probs: np.ndarray,
    vocab: list[str],
    top_k: int = 15,
    title: str | None = None,
) -> go.Figure:
    """水平柱状图展示下一个 Token 的概率分布 (圆角与清晰外显标签)"""
    top_indices = np.argsort(token_probs)[-top_k:]
    top_probs = token_probs[top_indices]
    top_words = [vocab[i] for i in top_indices]

    fig = go.Figure(go.Bar(
        x=top_probs,
        y=top_words,
        orientation='h',
        marker=dict(
            color=top_probs,
            colorscale="Blues",
            line=dict(width=1, color="rgba(15,23,42,0.1)"),
        ),
        text=[f"{p:.1%}" for p in top_probs],
        textposition="outside",
        textfont=dict(family="JetBrains Mono", size=10, color="#0f172a"),
        hovertemplate="Token: <b>%{y}</b><br>Probability: %{x:.4f}<extra></extra>",
    ))

    fig.update_layout(
        xaxis_title="Probability (置信概率)",
        yaxis_title="Token",
        xaxis=dict(range=[0, min(1.0, float(max(top_probs)) * 1.25)]),
        margin=dict(l=55, r=55, t=50 if title else 25, b=45),
    )
    return _apply_light_theme(fig, title)
