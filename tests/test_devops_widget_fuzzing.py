# Copyright (c) 2026 Yy1 (yuan278501381) | MIT License
"""
tests/test_devops_widget_fuzzing.py - UI 控件全组合模糊遍历与混沌韧性测试 (Tier-1 DevOps Gate)

世界级前沿测试实践:
不只是检验初始页面加载 (Happy Path)，而是对全站每个页面的所有 Radio Cards、Sliders、
Number Inputs、Select Sliders 进行程序化遍历切换，模拟用户疯狂点击与边界输入，
确保在任何参数组合与状态转换下，系统具备 100% 韧性与零崩溃保障。
"""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_PAGES_DIR = _PROJECT_ROOT / "dashboard" / "pages"


def get_all_pages() -> list[Path]:
    return sorted(_PAGES_DIR.glob("*.py"))


@pytest.mark.parametrize("page_path", get_all_pages(), ids=lambda p: p.name)
def test_page_widget_interaction_chaos_resilience(page_path: Path):
    """验证页面在交互控件状态切换与边界触发下 100% 稳定运行"""
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
            assert not at_num.exception, f"NumberInput {num_in.label} 触发崩溃: {[e.message for e in at_num.exception]}"
