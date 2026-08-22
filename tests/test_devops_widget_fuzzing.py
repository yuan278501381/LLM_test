# Copyright (c) 2026 Yy1 (yuan278501381) | MIT License
"""
tests/test_devops_widget_fuzzing.py - UI 控件代表值与边界值采样测试

本文件抽样检查各页面控件的代表值和边界值。它不是状态空间穷举，
也不能单独证明所有控件组合都无异常。
"""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_PAGES_DIR = _PROJECT_ROOT / "dashboard" / "pages"


def get_all_pages() -> list[Path]:
    return sorted(_PAGES_DIR.glob("*.py"))


@pytest.mark.parametrize("page_path", get_all_pages(), ids=lambda p: p.name)
def test_page_widget_representative_states_start_without_exception(page_path: Path):
    """抽样验证页面在代表性控件状态下能完成一次 Streamlit 重运行。"""
    at = AppTest.from_file(str(page_path), default_timeout=30).run()
    assert not at.exception, f"初始加载失败: {[e.message for e in at.exception]}"

    # 1. 测试首个和末尾 Slider 的极值响应 (Fresh AppTest 避免动态增删控件的陈旧引用)
    if at.sidebar.slider:
        at_min = AppTest.from_file(str(page_path), default_timeout=30).run()
        at_min.sidebar.slider[0].set_value(at_min.sidebar.slider[0].min).run()
        assert not at_min.exception, f"滑块设为最小值崩溃: {[e.message for e in at_min.exception]}"

        at_max = AppTest.from_file(str(page_path), default_timeout=30).run()
        at_max.sidebar.slider[-1].set_value(at_max.sidebar.slider[-1].max).run()
        assert not at_max.exception, f"滑块设为最大值崩溃: {[e.message for e in at_max.exception]}"

    # 2. 遍历并测试 Number Inputs
    if at.sidebar.number_input:
        at_num = AppTest.from_file(str(page_path), default_timeout=30).run()
        num_in = at_num.sidebar.number_input[0]
        if num_in.value is not None:
            num_in.set_value(num_in.value).run()
            assert not at_num.exception, (
                f"NumberInput {num_in.label} 触发崩溃: {[e.message for e in at_num.exception]}"
            )

    # 3. 抽样离散选择控件的末项；这只覆盖单控件变化，不宣称组合穷举。
    for widget_name in ("selectbox", "select_slider"):
        widgets = getattr(at.sidebar, widget_name)
        if widgets:
            sample = AppTest.from_file(str(page_path), default_timeout=30).run()
            widget = getattr(sample.sidebar, widget_name)[0]
            if widget.options:
                widget.set_value(widget.options[-1]).run()
                assert not sample.exception, (
                    f"{widget_name} {widget.label} 末项触发异常: "
                    f"{[e.message for e in sample.exception]}"
                )

    # Streamlit AppTest 无法把带 format_func 的 radio 展示文本反解为原始对象；
    # 此处验证选项可见性，真实切换由浏览器门禁覆盖。
    for radio in at.sidebar.radio:
        assert len(radio.options) >= 1
        assert radio.value is not None

    # 4. multiselect 分别覆盖空选择和全部选择。
    if at.sidebar.multiselect:
        for selected in ([], list(at.sidebar.multiselect[0].options)):
            sample = AppTest.from_file(str(page_path), default_timeout=30).run()
            widget = sample.sidebar.multiselect[0]
            widget.set_value(selected).run()
            assert not sample.exception, (
                f"multiselect {widget.label} 触发异常: {[e.message for e in sample.exception]}"
            )
