# Copyright (c) 2026 Yy1 (yuan278501381) | MIT License
"""
tests.test_dashboard_ui - 工业级前端 UI、图表工厂与 E2E 页面自动化测试套件

集成在 CI/CD 流水线中，确保：
1. 所有数据集类型 (moons, circles, xor, spiral, blobs) 100% 格式与维度正确
2. 所有经典预设 (PRESETS) 均能端到端全链路成功构建与收敛训练
3. Streamlit 所有主页面与子页面 (AppTest E2E) 运行 0 异常、0 崩溃
4. 矢量图标、拓扑探针与 HTML 渲染无任何代码块解析冲突
"""

import os

import numpy as np
import plotly.graph_objects as go
from streamlit.testing.v1 import AppTest

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
from dashboard.components.param_panel import PRESETS
from dashboard.styles.icons import svg_icon
from dashboard.styles.theme import render_metric_card
from dashboard.utils.state import ACTIVATION_MAP, OPTIMIZER_MAP, build_model, get_dataset
from nn_core.layers import Dense
from nn_core.losses import BinaryCrossEntropy
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
            svg = svg_icon(name, size=20, color="#1d4ed8")
            assert "<svg" in svg
            assert "</svg>" in svg
            assert 'width="20"' in svg
            assert "#1d4ed8" in svg

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
        lines = card_html.split("\n")
        for line in lines:
            if line.strip():
                assert not line.startswith("    "), f"Detected hazardous leading indent: {line}"


class TestAllDatasetsGeneration:
    """全量数据集生成矩阵校验"""

    def test_all_dataset_types_and_dimensions(self):
        dataset_types = [
            "moons",
            "circles",
            "xor",
            "spiral",
            "blobs",
            "Moons (双月分布)",
            "Circles (同心圆)",
            "XOR (正交分布)",
            "Spiral (双螺旋)",
            "Blobs (高斯聚类)",
        ]
        for dt in dataset_types:
            X, y = get_dataset(dt, n_samples=100, noise=0.1, random_state=42)
            assert X.shape == (100, 2), f"Dataset {dt} X shape mismatch: {X.shape}"
            assert y.shape == (100, 1), f"Dataset {dt} y shape mismatch: {y.shape}"
            assert not np.isnan(X).any(), f"NaN found in X for {dt}"
            assert not np.isnan(y).any(), f"NaN found in y for {dt}"


class TestAllPresetsExecution:
    """测试所有一键预设场景的端到端可执行性"""

    def test_all_presets_train_successfully(self):
        for preset_name, p in PRESETS.items():
            if preset_name == "自定义配置 (Custom)":
                continue
            X, y = get_dataset(p["dataset"], n_samples=100, noise=p["noise"], random_state=42)

            model = Sequential()
            prev_dim = 2
            for n_neurons in p["neurons"]:
                model.add(Dense(prev_dim, n_neurons, initializer=p["initializer"]))
                model.add(ACTIVATION_MAP[p["activation"]]())
                prev_dim = n_neurons

            model.add(Dense(prev_dim, 1, initializer=p["initializer"]))
            model.add(ACTIVATION_MAP["Sigmoid"]())

            loss_fn = BinaryCrossEntropy()
            opt = OPTIMIZER_MAP[p["optimizer"]](learning_rate=p["lr"])

            hist = model.train(
                X, y, loss_fn=loss_fn, optimizer=opt, epochs=5, batch_size=32, verbose=False
            )
            assert "loss" in hist
            assert len(hist["loss"]) == 5
            assert not np.isnan(hist["loss"][-1])


class TestChartFactory:
    """测试 Plotly 图表生成与主题属性"""

    def test_plot_decision_boundary(self):
        X, y = get_dataset("moons", 100, 0.1, 42)
        model = build_model(2, 1, [4], activation="ReLU", output_activation="Sigmoid")

        fig = plot_decision_boundary(model, X, y, probe_point=(0.5, 0.5), resolution=30)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 2

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


class TestStreamlitAppE2E:
    """Streamlit 全站 AppTest E2E 自动化端到端测试"""

    def test_app_landing_page_e2e(self):
        app_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "dashboard", "app.py")
        )
        at = AppTest.from_file(app_path, default_timeout=20).run()
        assert not at.exception, f"app.py runtime error: {at.exception}"

    def test_page1_perceptron_e2e(self):
        page1_path = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__), "..", "dashboard", "pages", "1_单神经元感知器.py"
            )
        )
        at = AppTest.from_file(page1_path, default_timeout=20).run()
        assert not at.exception, f"Page 1 runtime error: {at.exception}"

    def test_page2_deep_network_e2e(self):
        page2_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "dashboard", "pages", "2_多层网络.py")
        )
        at = AppTest.from_file(page2_path, default_timeout=20).run()
        assert not at.exception, f"Page 2 runtime error: {at.exception}"

    def test_page3_optimizer_arena_e2e(self):
        page3_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "dashboard", "pages", "3_优化器对比.py")
        )
        at = AppTest.from_file(page3_path, default_timeout=20).run()
        assert not at.exception, f"Page 3 runtime error: {at.exception}"

    def test_page4_parameter_lab_e2e(self):
        page4_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "dashboard", "pages", "4_参数实验室.py")
        )
        at = AppTest.from_file(page4_path, default_timeout=20).run()
        assert not at.exception, f"Page 4 runtime error: {at.exception}"
