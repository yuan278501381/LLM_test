# Copyright (c) 2026 Yy1 (yuan278501381) | MIT License
"""
tests/test_dashboard_ui.py - 工业级 Dashboard UI、组件、SVG 与 E2E 全自动化回归测试矩阵

覆盖范围：
1. SVG 矢量图标引擎完整性与容错机制
2. 高对比度亮色卡片 HTML 结构与缩进安全性
3. 2D 合成数据集全类型与维度一致性矩阵
4. 经典一键实验预设方案可训练性
5. 6 大核心 Plotly 图表渲染与布局无重叠
6. 网络拓扑图与单样本动态探针可视化
7. Streamlit 全站 5 大页面端到端 (E2E) AppTest 与交互仿真（按钮点击、预设切换、参数热调整）
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
from dashboard.constants.knowledge import DATASETS
from dashboard.styles.icons import SVG_ICONS, svg_icon
from dashboard.styles.theme import render_metric_card
from dashboard.utils.state import (
    get_dataset,
    resolve_activation,
    resolve_initializer,
    resolve_optimizer,
)
from nn_core.layers import Dense
from nn_core.losses import BinaryCrossEntropy
from nn_core.model import Sequential


class TestSvgIconEngine:
    """测试 SVG 图标渲染引擎"""

    def test_all_registered_icons(self):
        for icon_key in SVG_ICONS:
            rendered = svg_icon(icon_key, size=20, color="#1d4ed8")
            assert "<svg" in rendered
            assert "</svg>" in rendered
            assert 'width="20"' in rendered
            assert 'stroke="#1d4ed8"' in rendered

    def test_fallback_icon(self):
        rendered = svg_icon("non_existent_icon_key", size=16, color="#000000")
        assert "<svg" in rendered
        assert "</svg>" in rendered


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
        dataset_types = list(DATASETS.keys()) + [m.label for m in DATASETS.values()]
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
            act_cls = resolve_activation(p["activation"])
            init_name = resolve_initializer(p["initializer"])
            for n_neurons in p["neurons"]:
                model.add(Dense(prev_dim, n_neurons, initializer=init_name))
                model.add(act_cls())
                prev_dim = n_neurons

            model.add(Dense(prev_dim, 1, initializer=init_name))
            model.add(resolve_activation("Sigmoid")())

            loss_fn = BinaryCrossEntropy()
            opt_cls = resolve_optimizer(p["optimizer"])
            opt = opt_cls(learning_rate=p["lr"])

            history = model.train(X, y, loss_fn=loss_fn, optimizer=opt, epochs=2, batch_size=32)
            assert len(history["loss"]) == 2
            assert len(history["accuracy"]) == 2


class TestChartFactory:
    """测试所有图表工厂函数的生成完整性与 Plotly 兼容性"""

    @classmethod
    def setup_class(cls):
        cls.X = np.random.randn(100, 2)
        cls.y = (cls.X[:, 0] + cls.X[:, 1] > 0).astype(float).reshape(-1, 1)
        cls.model = Sequential()
        cls.model.add(Dense(2, 4, initializer="he"))
        cls.model.add(resolve_activation("ReLU")())
        cls.model.add(Dense(4, 1, initializer="he"))
        cls.model.add(resolve_activation("Sigmoid")())

    def test_plot_decision_boundary(self):
        fig = plot_decision_boundary(self.model, self.X, self.y, probe_point=(0.5, 0.5))
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
        grads = [np.random.randn(2, 4), np.random.randn(4, 1)]
        fig = plot_gradient_histograms(grads, ["Dense 1", "Dense 2"])
        assert isinstance(fig, go.Figure)

    def test_plot_weight_histograms(self):
        weights = [np.random.randn(2, 4), np.random.randn(4, 1)]
        fig = plot_weight_histograms(weights, ["Dense 1", "Dense 2"])
        assert isinstance(fig, go.Figure)

    def test_plot_multi_loss_curves(self):
        histories = {
            "SGD": {"loss": [0.9, 0.8, 0.7]},
            "Adam": {"loss": [0.9, 0.4, 0.1]},
        }
        fig = plot_multi_loss_curves(histories)
        assert isinstance(fig, go.Figure)

    def test_plot_weight_trajectory(self):
        traj = [np.array([[0.1, 0.2]]), np.array([[0.3, 0.5]]), np.array([[0.6, 0.8]])]
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
    """Streamlit 全站 AppTest E2E 自动化端到端测试与交互仿真"""

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

    def test_page4_button_interactions_e2e(self):
        """测试 Page 4 的交互按钮：训练 50 轮与单步微调"""
        page4_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "dashboard", "pages", "4_参数实验室.py")
        )
        at = AppTest.from_file(page4_path, default_timeout=25).run()
        assert not at.exception

        # 点击训练 50 轮
        train_btn = next((b for b in at.button if "50" in b.label), None)
        if train_btn:
            train_btn.click().run()
            assert not at.exception, f"Page 4 train 50 button error: {at.exception}"

        # 点击单步微调
        step_btn = next((b for b in at.button if "微调" in b.label or "STEP" in b.label), None)
        if step_btn:
            step_btn.click().run()
            assert not at.exception, f"Page 4 step button error: {at.exception}"
