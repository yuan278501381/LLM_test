# Copyright (c) 2026 Yy1 (yuan278501381) | MIT License
"""
tests/test_devops_api_contracts.py - 公共基础组件库符号契约守卫 (DevOps Gate 3)

检查关键公共模块的函数存在性和代表性签名；具体行为由组件测试覆盖。
"""

import inspect

from dashboard.components import param_panel, pedagogy
from dashboard.styles import theme


def test_param_panel_api_contract():
    """验证 dashboard.components.param_panel 公开接口契约"""
    required_functions = [
        "render_presets_selector",
        "render_dataset_selector",
        "render_network_params",
        "render_training_params",
        "render_probe_point_selector",
        "render_deep_dive_card",
        "render_regularization_params",
    ]
    for fn_name in required_functions:
        assert hasattr(param_panel, fn_name), f"param_panel 缺少必需的公共接口: {fn_name}"
        assert callable(getattr(param_panel, fn_name)), f"param_panel.{fn_name} 必须为可调用函数"


def test_theme_api_contract():
    """验证 dashboard.styles.theme 公开接口契约"""
    required_functions = [
        "apply_custom_theme",
        "render_hero_header",
        "render_section_heading",
        "render_metric_card",
        "render_floating_hud_navigator",
        "render_interactive_region_header",
        "render_live_param_status_bar",
        "render_page_guide",
        "render_formula_breakdown_card",
        "render_preset_badge",
        "anchor_badge",
    ]
    for fn_name in required_functions:
        assert hasattr(theme, fn_name), f"theme 缺少必需的公共接口: {fn_name}"
        assert callable(getattr(theme, fn_name)), f"theme.{fn_name} 必须为可调用函数"


def test_pedagogy_api_contract():
    """验证 dashboard.components.pedagogy 公开接口契约"""
    required_functions = [
        "render_lesson_evidence",
        "render_core_result_evidence",
        "render_result_evidence",
        "render_lesson_contract",
        "render_formative_quiz",
        "evidence_badge",
    ]
    for fn_name in required_functions:
        assert hasattr(pedagogy, fn_name), f"pedagogy 缺少必需的公共接口: {fn_name}"
        assert callable(getattr(pedagogy, fn_name)), f"pedagogy.{fn_name} 必须为可调用函数"

    expected_parameters = {
        "render_lesson_evidence": ["lesson_id", "show_contract"],
        "render_core_result_evidence": ["lesson_id"],
        "render_result_evidence": ["lesson_id", "result_id"],
        "render_lesson_contract": ["lesson_id"],
        "render_formative_quiz": ["lesson_id"],
        "evidence_badge": ["level"],
    }
    for function_name, parameter_names in expected_parameters.items():
        function = getattr(pedagogy, function_name)
        assert list(inspect.signature(function).parameters) == parameter_names


def test_navigation_api_signatures_are_stable():
    assert list(inspect.signature(theme.anchor_badge).parameters) == [
        "text",
        "color_type",
        "target_id",
    ]
    assert list(inspect.signature(theme.render_page_guide).parameters) == [
        "title",
        "plain_intro",
        "hyperparams_desc",
        "telemetry_desc",
        "experiments",
        "blueprint_sections",
        "guide_region_id",
    ]
