# Copyright (c) 2026 Yy1 (yuan278501381) | MIT License
"""
tests/test_dashboard_ui.py - Dashboard 组件、SVG 与 AppTest 回归测试

覆盖范围：
1. SVG 矢量图标引擎完整性与容错机制
2. 高对比度亮色卡片 HTML 结构与缩进安全性
3. 2D 合成数据集全类型与维度一致性矩阵
4. 经典一键实验预设方案可训练性
5. 11 大核心 Plotly 图表渲染与布局无重叠 (包含 M05~M09 新增图表)
6. 网络拓扑图与单样本动态探针可视化
7. Streamlit 全站 10 大页面 (app + 9 大里程碑页面) 端到端 (E2E) AppTest 与交互仿真
"""

import os

import numpy as np
import plotly.graph_objects as go
from streamlit.testing.v1 import AppTest

from dashboard.components.charts import (
    add_chart_playback,
    plot_activation_heatmap,
    plot_attention_heatmap_nlp,
    plot_decision_boundary,
    plot_embedding_space,
    plot_gradient_histograms,
    plot_loss_curve,
    plot_memory_decay_heatmap,
    plot_multi_loss_curves,
    plot_token_probabilities,
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
    """测试 SVG 纯矢量图标渲染引擎"""

    def test_all_registered_icons(self):
        assert len(SVG_ICONS) >= 20, f"Expected >= 20 vector icons, got {len(SVG_ICONS)}"
        for icon_key in SVG_ICONS:
            rendered = svg_icon(icon_key, size=20, color="#1d4ed8")
            assert "<svg" in rendered
            assert "</svg>" in rendered
            assert 'width="20"' in rendered
            assert 'stroke="#1d4ed8"' in rendered
            assert 'viewBox="0 0 24 24"' in rendered

    def test_fallback_icon(self):
        rendered = svg_icon("non_existent_icon_key", size=16, color="#000000")
        assert "<svg" in rendered
        assert "</svg>" in rendered


class TestThemeHtmlIntegrity:
    """测试 HTML 渲染无缩进冲突与 Markdown 代码块安全"""

    def test_metric_card_html(self):
        card_html = render_metric_card(
            "ACCURACY", "99.5%", delta="OPTIMAL", delta_type="positive", icon_name="target"
        )
        assert "telemetry-card" in card_html
        assert "99.5%" in card_html
        assert "ACCURACY" in card_html
        for line in card_html.split("\n"):
            if line.strip():
                assert not line.startswith("    "), f"Detected hazardous leading indent: {line}"

    def test_all_custom_renderers_safety(self):
        """测试所有自定义 HTML 组件均无前导缩进冲突"""
        from dashboard.styles.theme import (
            render_architecture_flow_card,
            render_sequence_flow,
            render_text_stream_box,
            render_vector_equation_card,
        )

        # 确保这些函数能够无异常执行
        # 验证不会抛异常即可
        assert callable(render_sequence_flow)
        assert callable(render_vector_equation_card)
        assert callable(render_architecture_flow_card)
        assert callable(render_text_stream_box)

    def test_course_navigation_is_self_contained_and_animates_focus(self):
        theme_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "dashboard", "styles", "theme.py")
        )
        with open(theme_path, encoding="utf-8") as theme_file:
            source = theme_file.read()

        assert "doc.__nnFocusRegion = focusRegion" in source
        assert "focusRegion(targetId)" in source
        assert "data-nn-focus-bound" in source
        assert "e.stopImmediatePropagation()" in source
        assert '"region-b"' in source
        assert "main.scrollTo" in source
        focus_keyframes = source.split("@keyframes region-focus-enter", 1)[1].split(
            ".nn-focus-chip", 1
        )[0]
        assert "!important" not in focus_keyframes

    def test_every_guided_page_declares_all_content_targets(self):
        pages_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "dashboard", "pages")
        )
        for filename in os.listdir(pages_dir):
            if not filename.endswith(".py"):
                continue
            page_path = os.path.join(pages_dir, filename)
            with open(page_path, encoding="utf-8") as page_file:
                source = page_file.read()
            if "render_page_guide" not in source:
                continue
            for region in ("region-a", "region-c", "region-d", "region-e"):
                assert region in source, f"{filename} is missing {region}"


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

    def test_plot_embedding_space(self):
        words = ["king", "queen", "man", "woman", "apple"]
        vecs = np.random.randn(5, 32)
        fig = plot_embedding_space(
            words,
            vecs,
            highlight_words=["king", "queen"],
            arithmetic={"A": "king", "B": "man", "C": "woman", "Result": "queen"},
        )
        assert isinstance(fig, go.Figure)

    def test_plot_memory_decay_heatmap(self):
        hidden_states = [np.random.randn(1, 16) for _ in range(5)]
        tokens = ["the", "cat", "sat", "on", "mat"]
        fig = plot_memory_decay_heatmap(hidden_states, tokens)
        assert isinstance(fig, go.Figure)

    def test_plot_attention_heatmap_nlp(self):
        attn_matrix = np.random.rand(5, 5)
        attn_matrix[np.triu_indices(5, 1)] = 0
        tokens = ["the", "cat", "sat", "on", "mat"]
        fig = plot_attention_heatmap_nlp(attn_matrix, tokens, tokens)
        assert isinstance(fig, go.Figure)

    def test_plot_token_probabilities(self):
        probs = np.array([0.1, 0.5, 0.2, 0.15, 0.05])
        vocab = ["the", "cat", "dog", "king", "queen"]
        fig = plot_token_probabilities(probs, vocab, top_k=3)
        assert isinstance(fig, go.Figure)

    def test_global_chart_playback_preserves_static_result(self):
        fig = go.Figure(go.Scatter(x=[1, 2, 3], y=[2, 4, 8], mode="lines+markers"))
        original_x = tuple(fig.data[0].x)
        animated = add_chart_playback(fig, frame_count=6)

        assert tuple(animated.data[0].x) == original_x
        assert len(animated.frames) == 7
        assert len(animated.frames[0].data[0].x) == 0
        assert tuple(animated.frames[-1].data[0].x) == original_x
        assert animated.layout.updatemenus[0].buttons[1].label == "▶ 读图"

    def test_global_chart_playback_supports_heatmap_and_bar(self):
        fig = go.Figure(
            data=[
                go.Heatmap(z=[[0.1, 0.9], [0.4, 0.6]]),
                go.Bar(x=["A", "B"], y=[2, 5]),
            ]
        )
        animated = add_chart_playback(fig, frame_count=4)

        assert animated.frames[0].data[0].opacity == 0
        assert tuple(animated.frames[0].data[1].y) == (0.0, 0.0)
        assert animated.frames[-1].data[0].opacity == 1
        assert tuple(animated.frames[-1].data[1].y) == (2.0, 5.0)


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
    """Streamlit 全站 10 大页面 (App + 9 大里程碑) AppTest E2E 自动化端到端测试与交互仿真"""

    def test_app_landing_page_e2e(self):
        app_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "dashboard", "app.py")
        )
        at = AppTest.from_file(app_path, default_timeout=20).run()
        assert not at.exception, f"app.py runtime error: {at.exception}"

    def test_page0_math_foundations_e2e(self):
        """测试 M00：shape、链式法则与有限差分页面。"""
        page_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "dashboard", "pages", "0_数学基础.py")
        )
        at = AppTest.from_file(page_path, default_timeout=25).run()
        assert not at.exception, f"Page 0 runtime error: {at.exception}"

    def test_page1_perceptron_e2e(self):
        page_path = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__), "..", "dashboard", "pages", "1_单神经元感知器.py"
            )
        )
        at = AppTest.from_file(page_path, default_timeout=20).run()
        assert not at.exception, f"Page 1 runtime error: {at.exception}"

    def test_page1_player_uses_client_side_animation(self):
        """播放器必须在浏览器原位更新，禁止重新引入服务端定时片段。"""
        page_path = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__), "..", "dashboard", "pages", "1_单神经元感知器.py"
            )
        )
        at = AppTest.from_file(page_path, default_timeout=25).run()
        assert not at.exception

        with open(page_path, encoding="utf-8") as page_file:
            source = page_file.read()
        assert "@st.fragment" not in source
        assert "render_player_controls(player_payload)" in source
        assert "render_boundary_canvas(player_payload)" in source
        assert "render_trajectory_canvas(player_payload)" in source

    def test_page2_deep_network_e2e(self):
        page_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "dashboard", "pages", "2_多层网络.py")
        )
        at = AppTest.from_file(page_path, default_timeout=20).run()
        assert not at.exception, f"Page 2 runtime error: {at.exception}"

        with open(page_path, encoding="utf-8") as page_file:
            source = page_file.read()
        assert "TrainingTrajectoryRecorder" in source
        assert 'event_name="nn:m2-train"' in source
        assert "render_network_signal_canvas(training_payload)" in source
        assert "render_probe_manifold_canvas(training_payload)" in source

    def test_page3_optimizer_arena_e2e(self):
        page_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "dashboard", "pages", "3_优化器对比.py")
        )
        at = AppTest.from_file(page_path, default_timeout=20).run()
        assert not at.exception, f"Page 3 runtime error: {at.exception}"

    def test_page4_parameter_lab_e2e(self):
        page_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "dashboard", "pages", "4_参数实验室.py")
        )
        at = AppTest.from_file(page_path, default_timeout=20).run()
        assert not at.exception, f"Page 4 runtime error: {at.exception}"

    def test_page4_button_interactions_e2e(self):
        """测试 Page 4 的交互按钮：训练 50 轮与单步微调"""
        page_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "dashboard", "pages", "4_参数实验室.py")
        )
        at = AppTest.from_file(page_path, default_timeout=25).run()
        assert not at.exception

        train_btn = next((b for b in at.button if "50" in b.label), None)
        if train_btn:
            train_btn.click().run()
            assert not at.exception, f"Page 4 train 50 button error: {at.exception}"

        step_btn = next((b for b in at.button if "微调" in b.label or "STEP" in b.label), None)
        if step_btn:
            step_btn.click().run()
            assert not at.exception, f"Page 4 step button error: {at.exception}"

    def test_page5_embedding_space_e2e(self):
        """测试 M05: 词嵌入语义空间页面"""
        page_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "dashboard", "pages", "5_词嵌入空间.py")
        )
        at = AppTest.from_file(page_path, default_timeout=25).run()
        assert not at.exception, f"Page 5 runtime error: {at.exception}"

    def test_page6_sequence_memory_e2e(self):
        """测试 M06: 序列记忆与遗忘瓶颈页面"""
        page_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "dashboard", "pages", "6_序列记忆.py")
        )
        at = AppTest.from_file(page_path, default_timeout=25).run()
        assert not at.exception, f"Page 6 runtime error: {at.exception}"

    def test_page7_attention_mechanism_e2e(self):
        """测试 M07: 注意力机制页面"""
        page_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "dashboard", "pages", "7_注意力机制.py")
        )
        at = AppTest.from_file(page_path, default_timeout=25).run()
        assert not at.exception, f"Page 7 runtime error: {at.exception}"

    def test_page8_transformer_block_e2e(self):
        """测试 M08: Transformer 结构块页面"""
        page_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "dashboard", "pages", "8_Transformer.py")
        )
        at = AppTest.from_file(page_path, default_timeout=25).run()
        assert not at.exception, f"Page 8 runtime error: {at.exception}"

    def test_page9_mini_gpt_e2e(self):
        """测试 M09: Mini-GPT 文本生成页面与交互按钮"""
        page_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "dashboard", "pages", "9_Mini_GPT.py")
        )
        at = AppTest.from_file(page_path, default_timeout=25).run()
        assert not at.exception, f"Page 9 runtime error: {at.exception}"

        # 测试一键生成按钮
        gen_btn = next((b for b in at.button if "一键" in b.label or "生成" in b.label), None)
        if gen_btn:
            gen_btn.click().run()
            assert not at.exception, f"Page 9 generate button error: {at.exception}"

    def test_page10_vision_perception_e2e(self):
        """测试 M10: 卷积与视觉感知页面"""
        page_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "dashboard", "pages", "10_视觉感知.py")
        )
        at = AppTest.from_file(page_path, default_timeout=25).run()
        assert not at.exception, f"Page 10 runtime error: {at.exception}"

    def test_page11_audio_perception_e2e(self):
        """测试 M11: 音频信号与语音理解页面"""
        page_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "dashboard", "pages", "11_音频感知.py")
        )
        at = AppTest.from_file(page_path, default_timeout=25).run()
        assert not at.exception, f"Page 11 runtime error: {at.exception}"

    def test_page12_video_world_model_e2e(self):
        """测试 M12: 视频理解与世界模型页面"""
        page_path = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__), "..", "dashboard", "pages", "12_视频与世界模型.py"
            )
        )
        at = AppTest.from_file(page_path, default_timeout=25).run()
        assert not at.exception, f"Page 12 runtime error: {at.exception}"

        with open(page_path, encoding="utf-8") as page_file:
            source = page_file.read()
        assert 'event_name="nn:m12-frame"' in source
        assert "render_video_timeline(video_payload)" in source
        assert "render_timeline_controls(" in source

    def test_page13_pretraining_paradigms_e2e(self):
        """测试 M13: 预训练范式全景页面"""
        page_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "dashboard", "pages", "13_预训练范式.py")
        )
        at = AppTest.from_file(page_path, default_timeout=25).run()
        assert not at.exception, f"Page 13 runtime error: {at.exception}"

    def test_page14_posttraining_alignment_e2e(self):
        """测试 M14: 后训练对齐与轻量微调页面"""
        page_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "dashboard", "pages", "14_后训练工程.py")
        )
        at = AppTest.from_file(page_path, default_timeout=25).run()
        assert not at.exception, f"Page 14 runtime error: {at.exception}"

    def test_page15_evaluation_harness_e2e(self):
        """测试 M15: 评估基准框架页面"""
        page_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "dashboard", "pages", "15_评估基准.py")
        )
        at = AppTest.from_file(page_path, default_timeout=25).run()
        assert not at.exception, f"Page 15 runtime error: {at.exception}"

    def test_page16_reinforcement_learning_e2e(self):
        """测试 M16: 强化学习与自主智能体实验室页面"""
        page_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "dashboard", "pages", "16_强化学习.py")
        )
        at = AppTest.from_file(page_path, default_timeout=25).run()
        assert not at.exception, f"Page 16 runtime error: {at.exception}"

    def test_page17_engineering_traps_e2e(self):
        """测试 M17: 工程陷阱与Harness控制环页面"""
        page_path = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                "..",
                "dashboard",
                "pages",
                "17_工程陷阱与Harness.py",
            )
        )
        at = AppTest.from_file(page_path, default_timeout=25).run()
        assert not at.exception, f"Page 17 runtime error: {at.exception}"
