# Copyright (c) 2026 Yy1 (yuan278501381) | MIT License
"""
dashboard.components.network_viz - 动态神经元拓扑与活性探针可视化引擎 (现代亮色风格)

使用 Plotly 构建现代极简亮色神经网络拓扑图：
- 节点状态响应「单样本活性探针」：根据实际激活值 $a^{[l]}$ 实时高对比变色
- 突触连线粗细与颜色映射权重大小与极性（正向蓝、负向红）
- 截断保护与超大层优雅折叠
- 亮色背景下清晰可辨
"""

import numpy as np
import plotly.graph_objects as go


def plot_network_topology(
    layer_sizes: list[int],
    weights: list[np.ndarray] | None = None,
    neuron_activations: list[np.ndarray] | None = None,
    title: str = "TOPOLOGY & PROBE // 神经网络拓扑与动态活性探针",
) -> go.Figure:
    """
    绘制亮色神经网络拓扑图与神经元激活状态。

    Args:
        layer_sizes: 每层的神经元数量列表，如 [2, 8, 4, 1]
        weights: 各层权重矩阵列表（可选），控制突触粗细与极性
        neuron_activations: 各层在当前探针点下的激活向量列表（可选），控制神经元发光状态
        title: 图表标题

    Returns:
        Plotly Figure
    """
    n_layers = len(layer_sizes)
    max_neurons = max(layer_sizes)
    display_limit = 10

    fig = go.Figure()

    layer_positions: list[list[tuple[float, float]]] = []
    x_spacing = 1.4

    for layer_idx, n_neurons in enumerate(layer_sizes):
        x = layer_idx * x_spacing
        show_n = min(n_neurons, display_limit)
        y_offset = (max_neurons - show_n) / 2.0
        positions = [(x, y_offset + i) for i in range(show_n)]
        layer_positions.append(positions)

    # -------------------------------------------------------------------------
    # 1. 绘制突触连接线 (Synaptic Connections)
    # -------------------------------------------------------------------------
    for layer_idx in range(n_layers - 1):
        positions_from = layer_positions[layer_idx]
        positions_to = layer_positions[layer_idx + 1]

        for i, (x1, y1) in enumerate(positions_from):
            for j, (x2, y2) in enumerate(positions_to):
                if weights and layer_idx < len(weights):
                    w = weights[layer_idx]
                    if i < w.shape[0] and j < w.shape[1]:
                        w_val = float(w[i, j])
                        width = float(np.clip(abs(w_val) * 2.8, 0.5, 4.5))
                        alpha = float(np.clip(abs(w_val) / 2.0, 0.2, 0.9))
                        color = (
                            f"rgba(37, 99, 235, {alpha})"
                            if w_val >= 0
                            else f"rgba(225, 29, 72, {alpha})"
                        )
                    else:
                        width, color = 0.5, "rgba(203, 213, 225, 0.3)"
                else:
                    width, color = 0.6, "rgba(203, 213, 225, 0.4)"

                fig.add_trace(go.Scatter(
                    x=[x1, x2], y=[y1, y2],
                    mode="lines",
                    line=dict(width=width, color=color),
                    hoverinfo="skip",
                    showlegend=False,
                ))

    # -------------------------------------------------------------------------
    # 2. 绘制神经元节点与活性探针 (Neuron Activation Nodes)
    # -------------------------------------------------------------------------
    layer_labels = ["INPUT", *[f"HIDDEN #{i}" for i in range(1, n_layers - 1)], "OUTPUT"]

    for layer_idx, positions in enumerate(layer_positions):
        xs = [p[0] for p in positions]
        ys = [p[1] for p in positions]
        actual_n = layer_sizes[layer_idx]
        show_n = len(positions)

        layer_acts = None
        if neuron_activations and layer_idx < len(neuron_activations):
            layer_acts = neuron_activations[layer_idx].ravel()

        node_colors = []
        node_border_colors = []
        node_text_colors = []
        node_sizes = []
        hover_texts = []

        for i in range(show_n):
            if layer_acts is not None and i < len(layer_acts):
                act_val = float(layer_acts[i])
                if act_val > 0.5:
                    node_colors.append("#2563eb")
                    node_border_colors.append("#1d4ed8")
                    node_text_colors.append("#ffffff")
                    node_sizes.append(28)
                    status = "[EXCITED // 强兴奋]"
                elif act_val > 0.05:
                    node_colors.append("#059669")
                    node_border_colors.append("#047857")
                    node_text_colors.append("#ffffff")
                    node_sizes.append(26)
                    status = "[ACTIVE // 适度激活]"
                elif act_val < -0.05:
                    node_colors.append("#e11d48")
                    node_border_colors.append("#be123c")
                    node_text_colors.append("#ffffff")
                    node_sizes.append(26)
                    status = "[INHIBITED // 负向抑制]"
                else:
                    node_colors.append("#f1f5f9")
                    node_border_colors.append("#cbd5e1")
                    node_text_colors.append("#64748b")
                    node_sizes.append(22)
                    status = "[DORMANT // 休眠]"

                hover_texts.append(
                    f"<b>{layer_labels[layer_idx]} · Node #{i+1}</b><br>"
                    + f"Activation: <code>a={act_val:.4f}</code><br>"
                    + f"Status: {status}"
                )
            else:
                node_colors.append("#ffffff")
                node_border_colors.append("#2563eb" if layer_idx == 0 or layer_idx == n_layers - 1 else "#7c3aed")
                node_text_colors.append("#0f172a")
                node_sizes.append(24)
                hover_texts.append(f"<b>{layer_labels[layer_idx]} · Node #{i+1}/{actual_n}</b>")

        fig.add_trace(go.Scatter(
            x=xs, y=ys,
            mode="markers+text",
            marker=dict(
                size=node_sizes,
                color=node_colors,
                line=dict(width=2, color=node_border_colors),
            ),
            text=[str(i + 1) for i in range(show_n)],
            textfont=dict(size=9, color=node_text_colors if any(c == "#ffffff" for c in node_text_colors) else "#0f172a", family="JetBrains Mono"),
            hovertext=hover_texts,
            hoverinfo="text",
            showlegend=False,
        ))

        if actual_n > display_limit:
            fig.add_annotation(
                x=layer_idx * x_spacing,
                y=positions[-1][1] + 0.6,
                text=f"... (+{actual_n - display_limit} hidden)",
                showarrow=False,
                font=dict(size=10, color="#64748b"),
            )

    # -------------------------------------------------------------------------
    # 3. 底部层名与维数标签
    # -------------------------------------------------------------------------
    for layer_idx in range(n_layers):
        fig.add_annotation(
            x=layer_idx * x_spacing,
            y=-0.8,
            text=f"<b>{layer_labels[layer_idx]}</b><br><span style='color:#64748b; font-size:10px;'>dim={layer_sizes[layer_idx]}</span>",
            showarrow=False,
            font=dict(size=11, color="#0f172a"),
        )

    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b>",
            font=dict(size=14, color="#0f172a"),
            x=0.02,
            y=0.96,
        ),
        template="plotly_white",
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        margin=dict(l=20, r=20, t=50, b=50),
        height=420,
        plot_bgcolor="#ffffff",
        paper_bgcolor="rgba(0,0,0,0)",
    )

    return fig
