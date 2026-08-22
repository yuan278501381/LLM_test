# Copyright (c) 2026 Yy1 (yuan278501381) | MIT License
"""
tests/test_devops_api_contracts.py - 公共基础组件库符号契约守卫 (DevOps Gate 3)

严格锁定关键公共模块的公开 API 导出清单。
杜绝在重构或性能优化时不慎误删、遗漏或重命名公共函数，确保全站契约高度稳定。
"""

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
        "evidence_badge",
    ]
    for fn_name in required_functions:
        assert hasattr(pedagogy, fn_name), f"pedagogy 缺少必需的公共接口: {fn_name}"
        assert callable(getattr(pedagogy, fn_name)), f"pedagogy.{fn_name} 必须为可调用函数"
