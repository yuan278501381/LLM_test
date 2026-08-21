# Copyright (c) 2026 Yy1 (yuan278501381) | MIT License
"""
dashboard.components.network_viz - 网络拓扑可视化

使用 Plotly 绘制神经网络结构图：圆圈表示神经元，线条连接表示权重，
线条粗细正比于权重绝对值，颜色表示正负（蓝色正、红色负）。
"""


import numpy as np
import plotly.graph_objects as go


def plot_network_topology(
    layer_sizes: list[int],
    weights: list[np.ndarray] | None = None,
    title: str = "🧬 网络结构",
) -> go.Figure:
    """
    绘制神经网络拓扑图。

    Args:
        layer_sizes: 每层的神经元数量列表，如 [2, 8, 4, 1]
        weights: 各层权重矩阵列表（可选），用于控制连线粗细和颜色
        title: 图表标题

    Returns:
        Plotly Figure
    """
    n_layers = len(layer_sizes)
    max_neurons = max(layer_sizes)

    # 限制显示的神经元数量，太多时截断
    display_limit = 12

    fig = go.Figure()

    # 计算每层的 (x, y) 坐标
    layer_positions: list[list[tuple[float, float]]] = []
    x_spacing = 1.0

    for layer_idx, n_neurons in enumerate(layer_sizes):
        x = layer_idx * x_spacing
        show_n = min(n_neurons, display_limit)
        y_offset = (max_neurons - show_n) / 2
        positions = [(x, y_offset + i) for i in range(show_n)]
        layer_positions.append(positions)

    # ---- 绘制连接线 ----
    for layer_idx in range(n_layers - 1):
        positions_from = layer_positions[layer_idx]
        positions_to = layer_positions[layer_idx + 1]

        for i, (x1, y1) in enumerate(positions_from):
            for j, (x2, y2) in enumerate(positions_to):
                # 计算线条属性
                if weights and layer_idx < len(weights):
                    w = weights[layer_idx]
                    if i < w.shape[0] and j < w.shape[1]:
                        w_val = float(w[i, j])
                        width = min(max(abs(w_val) * 3, 0.3), 4)
                        color = (
                            f"rgba(0,212,255,{min(abs(w_val), 0.8)})"
                            if w_val >= 0
                            else f"rgba(255,107,107,{min(abs(w_val), 0.8)})"
                        )
                    else:
                        width, color = 0.5, "rgba(128,128,128,0.2)"
                else:
                    width, color = 0.5, "rgba(128,128,128,0.3)"

                fig.add_trace(go.Scatter(
                    x=[x1, x2], y=[y1, y2],
                    mode="lines",
                    line=dict(width=width, color=color),
                    hoverinfo="skip",
                    showlegend=False,
                ))

    # ---- 绘制神经元 ----
    layer_labels = ["输入层"] + [f"隐藏层 {i}" for i in range(1, n_layers - 1)] + ["输出层"]

    for layer_idx, positions in enumerate(layer_positions):
        xs = [p[0] for p in positions]
        ys = [p[1] for p in positions]
        actual_n = layer_sizes[layer_idx]
        show_n = len(positions)

        labels = [
            f"{layer_labels[layer_idx]}<br>神经元 {i+1}/{actual_n}"
            for i in range(show_n)
        ]

        fig.add_trace(go.Scatter(
            x=xs, y=ys,
            mode="markers+text",
            marker=dict(
                size=24,
                color="#1a1a2e",
                line=dict(width=2, color="#00D4FF"),
            ),
            text=[str(i + 1) for i in range(show_n)],
            textfont=dict(size=8, color="white"),
            hovertext=labels,
            hoverinfo="text",
            showlegend=False,
        ))

        # 如果有截断，显示省略号
        if actual_n > display_limit:
            fig.add_annotation(
                x=layer_idx * x_spacing,
                y=positions[-1][1] + 0.5,
                text=f"... (+{actual_n - display_limit})",
                showarrow=False,
                font=dict(size=10, color="gray"),
            )

    # ---- 层标签 ----
    for layer_idx in range(n_layers):
        fig.add_annotation(
            x=layer_idx * x_spacing,
            y=-1.5,
            text=f"<b>{layer_labels[layer_idx]}</b><br>({layer_sizes[layer_idx]} 个)",
            showarrow=False,
            font=dict(size=11, color="white"),
        )

    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        template="plotly_dark",
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        margin=dict(l=20, r=20, t=50, b=60),
        height=450,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )

    return fig
