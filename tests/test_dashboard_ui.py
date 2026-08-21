# Copyright (c) 2026 Yy1 (yuan278501381) | MIT License
"""
tests.test_dashboard_ui - 工业级前端 UI 与可视化图表自动化测试套件

集成在 CI/CD 流水线中，确保所有图表工厂、矢量图标、拓扑网络与 HTML 渲染逻辑 100% 稳健无退化。
"""

import numpy as np
import plotly.graph_objects as go

from dashboard.components.charts import (
    plot_activation_heatmap,
    plot_decision_boundary,
    plot_gradient_histograms,
    plot_loss_curve,
    plot_multi_loss_curves,
    plot_weight_histograms,
    plot_weight_trajectory,
)
from dashboard.components.network_viz import plot_network_topology
from dashboard.styles.icons import svg_icon
from dashboard.styles.theme import (
    render_metric_card,
)
from dashboard.utils.state import ACTIVATION_MAP, get_dataset
from nn_core.layers import Dense
from nn_core.model import Sequential


class TestSvgIconEngine:
    """测试矢量 SVG 图标引擎"""

    def test_all_registered_icons(self):
        icon_names = [
            "cpu",
            "layers",
            "activity",
            "crosshair",
            "zap",
            "sliders",
            "database",
            "trending-up",
            "trending-down",
            "shield",
            "refresh",
            "play",
            "step",
            "terminal",
            "target",
            "box",
            "award",
            "git-commit",
            "gauge",
        ]
        for name in icon_names:
            svg = svg_icon(name, size=20, color="#2563eb")
            assert "<svg" in svg
            assert "</svg>" in svg
            assert 'width="20"' in svg
            assert "#2563eb" in svg

    def test_fallback_icon(self):
        svg = svg_icon("non_existent_icon_xyz")
        assert "<svg" in svg


class TestThemeHtmlIntegrity:
    """测试 HTML 渲染无缩进冲突"""

    def test_metric_card_html(self):
        card_html = render_metric_card(
            "ACCURACY", "99.5%", delta="OPTIMAL", delta_type="positive", icon_name="target"
        )
        assert "telemetry-card" in card_html
        assert "99.5%" in card_html
        assert "ACCURACY" in card_html
        # 确保没有以4个以上空格开头的危险换行，防止 Markdown 代码块逃逸
        lines = card_html.split("\n")
        for line in lines:
            if line.strip():
                assert not line.startswith("    "), f"Detected hazardous leading indent: {line}"


class TestChartFactory:
    """测试 Plotly 图表生成与主题属性"""

    def test_plot_decision_boundary(self):
        X, y = get_dataset("moons", 100, 0.1, 42)
        model = Sequential()
        model.add(Dense(2, 4))
        model.add(ACTIVATION_MAP["ReLU"]())
        model.add(Dense(4, 1))
        model.add(ACTIVATION_MAP["Sigmoid"]())

        fig = plot_decision_boundary(model, X, y, probe_point=(0.5, 0.5), resolution=30)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 2  # 等高面 + 散点 + 探针

    def test_plot_loss_curve(self):
        history = {
            "loss": [0.8, 0.6, 0.4, 0.2],
            "accuracy": [0.5, 0.7, 0.85, 0.95],
        }
        fig = plot_loss_curve(history)
        assert isinstance(fig, go.Figure)

    def test_plot_activation_heatmap(self):
        acts = [np.random.rand(20, 8), np.random.rand(20, 4)]
        fig = plot_activation_heatmap(acts)
        assert isinstance(fig, go.Figure)

    def test_plot_gradient_histograms(self):
        grads = [np.random.randn(2, 8), np.random.randn(8, 1)]
        names = ["Dense #1", "Dense #2"]
        fig = plot_gradient_histograms(grads, names)
        assert isinstance(fig, go.Figure)

    def test_plot_weight_histograms(self):
        weights = [np.random.randn(2, 8), np.random.randn(8, 1)]
        names = ["Dense #1", "Dense #2"]
        fig = plot_weight_histograms(weights, names)
        assert isinstance(fig, go.Figure)

    def test_plot_multi_loss_curves(self):
        histories = {
            "SGD": {"loss": [0.9, 0.8, 0.7]},
            "Adam": {"loss": [0.9, 0.4, 0.1]},
        }
        fig = plot_multi_loss_curves(histories)
        assert isinstance(fig, go.Figure)

    def test_plot_weight_trajectory(self):
        traj = [np.array([[0.1, 0.2]]), np.array([[0.3, 0.4]]), np.array([[0.5, 0.6]])]
        fig = plot_weight_trajectory(traj)
        assert isinstance(fig, go.Figure)


class TestNetworkTopologyViz:
    """测试网络拓扑与活性探针渲染"""

    def test_plot_network_topology_basic(self):
        fig = plot_network_topology(layer_sizes=[2, 8, 4, 1])
        assert isinstance(fig, go.Figure)

    def test_plot_network_topology_with_probe(self):
        weights = [np.random.randn(2, 4), np.random.randn(4, 1)]
        activations = [
            np.array([[0.5, -0.2]]),
            np.array([[0.9, 0.1, -0.8, 0.0]]),
            np.array([[0.85]]),
        ]
        fig = plot_network_topology(
            layer_sizes=[2, 4, 1],
            weights=weights,
            neuron_activations=activations,
        )
        assert isinstance(fig, go.Figure)
