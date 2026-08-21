# Copyright (c) 2026 Yy1 (yuan278501381) | MIT License
"""
dashboard.components.network_viz - 动态神经元拓扑与活性探针可视化引擎

使用 Plotly 构建世界级交互式神经网络拓扑图：
- 节点状态响应「单样本活性探针」：根据实际激活值 $a^{[l]}$ 实时发光/渐变着色
- 突触连线粗细与颜色映射权重大小与正负（正向青蓝、负向烈焰红）
- 截断保护与超大层优雅折叠
"""

from typing import Any

import numpy as np
import plotly.graph_objects as go


def plot_network_topology(
    layer_sizes: list[int],
    weights: list[np.ndarray] | None = None,
    neuron_activations: list[np.ndarray] | None = None,
    title: str = "🧬 神经网络微观拓扑 · 动态活性探针",
) -> go.Figure:
    """
    绘制世界级神经网络拓扑图与神经元激活状态。

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
    display_limit = 10  # 单层最多展示神经元个数，超出优雅折叠

    fig = go.Figure()

    # 计算每层神经元的几何坐标
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
                        width = float(np.clip(abs(w_val) * 2.8, 0.4, 4.5))
                        alpha = float(np.clip(abs(w_val) / 2.0, 0.15, 0.85))
                        color = (
                            f"rgba(56, 189, 248, {alpha})"
                            if w_val >= 0
                            else f"rgba(251, 113, 133, {alpha})"
                        )
                    else:
                        width, color = 0.5, "rgba(148, 163, 184, 0.15)"
                else:
                    width, color = 0.6, "rgba(148, 163, 184, 0.2)"

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
    layer_labels = ["输入层"] + [f"隐藏层 {i}" for i in range(1, n_layers - 1)] + ["输出层"]

    for layer_idx, positions in enumerate(layer_positions):
        xs = [p[0] for p in positions]
        ys = [p[1] for p in positions]
        actual_n = layer_sizes[layer_idx]
        show_n = len(positions)

        # 获取该层探针激活值（如果存在）
        layer_acts = None
        if neuron_activations and layer_idx < len(neuron_activations):
            layer_acts = neuron_activations[layer_idx].ravel()

        node_colors = []
        node_border_colors = []
        node_sizes = []
        hover_texts = []

        for i in range(show_n):
            if layer_acts is not None and i < len(layer_acts):
                act_val = float(layer_acts[i])
                # 根据激活值强度映射节点颜色
                if act_val > 0.5:
                    node_colors.append("rgba(56, 189, 248, 0.9)")    # 强兴奋
                    node_border_colors.append("#38bdf8")
                    node_sizes.append(28)
                    status = "🟢 强兴奋 (Highly Active)"
                elif act_val > 0.05:
                    node_colors.append("rgba(52, 211, 153, 0.75)")   # 中等活跃
                    node_border_colors.append("#34d399")
                    node_sizes.append(26)
                    status = "🟡 适度激活 (Active)"
                elif act_val < -0.05:
                    node_colors.append("rgba(251, 113, 133, 0.75)")  # 负向激活
                    node_border_colors.append("#fb7185")
                    node_sizes.append(26)
                    status = "🔴 负向抑制 (Inhibited)"
                else:
                    node_colors.append("rgba(30, 41, 59, 0.6)")      # 休眠/零
                    node_border_colors.append("rgba(148, 163, 184, 0.4)")
                    node_sizes.append(22)
                    status = "⚪ 休眠/死神经元 (Dormant)"

                hover_texts.append(
                    f"<b>{layer_labels[layer_idx]} · 神经元 #{i+1}</b><br>"
                    + f"实时激活值 a = <code>{act_val:.4f}</code><br>"
                    + f"状态判定: {status}"
                )
            else:
                # 默认非探针状态下的现代科技感节点
                node_colors.append("rgba(15, 23, 42, 0.85)")
                node_border_colors.append("#38bdf8" if layer_idx == 0 or layer_idx == n_layers - 1 else "#818cf8")
                node_sizes.append(24)
                hover_texts.append(f"<b>{layer_labels[layer_idx]} · 神经元 #{i+1}/{actual_n}</b>")

        fig.add_trace(go.Scatter(
            x=xs, y=ys,
            mode="markers+text",
            marker=dict(
                size=node_sizes,
                color=node_colors,
                line=dict(width=2, color=node_border_colors),
            ),
            text=[str(i + 1) for i in range(show_n)],
            textfont=dict(size=9, color="#f8fafc", family="JetBrains Mono"),
            hovertext=hover_texts,
            hoverinfo="text",
            showlegend=False,
        ))

        # 超限折叠标注
        if actual_n > display_limit:
            fig.add_annotation(
                x=layer_idx * x_spacing,
                y=positions[-1][1] + 0.6,
                text=f"... (+{actual_n - display_limit} 个隐藏)",
                showarrow=False,
                font=dict(size=10, color="#94a3b8"),
            )

    # -------------------------------------------------------------------------
    # 3. 底部层名与维数标签
    # -------------------------------------------------------------------------
    for layer_idx in range(n_layers):
        fig.add_annotation(
            x=layer_idx * x_spacing,
            y=-0.8,
            text=f"<b>{layer_labels[layer_idx]}</b><br><span style='color:#94a3b8;'>dim={layer_sizes[layer_idx]}</span>",
            showarrow=False,
            font=dict(size=11, color="#f8fafc"),
        )

    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b>",
            font=dict(size=15, color="#f8fafc"),
            x=0.02,
            y=0.96,
        ),
        template="plotly_dark",
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        margin=dict(l=20, r=20, t=50, b=50),
        height=420,
        plot_bgcolor="rgba(8, 12, 20, 0.4)",
        paper_bgcolor="rgba(0,0,0,0)",
    )

    return fig
