# Copyright (c) 2026 Yy1 (yuan278501381) | MIT License
"""
tests/test_devops_smoke_pages.py - 全站多页面无死角动态自动发现冒烟测试 (DevOps Gate 2)

自动扫描 `dashboard/pages/*.py` 和 `dashboard/app.py`，
利用 Streamlit 官方 AppTest 无头沙盒框架对全站每一个页面执行端到端仿真渲染。
无论项目如何扩展（新增页面、重命名页面），无需人工维护测试列表，自动 100% 覆盖。
一旦有任何页面产生运行时异常、未捕获错误或组件崩溃，毫秒级熔断拦截！
"""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_PAGES_DIR = _PROJECT_ROOT / "dashboard" / "pages"
_APP_PY = _PROJECT_ROOT / "dashboard" / "app.py"


def get_all_page_scripts() -> list[Path]:
    """动态获取全部 Streamlit 入口与页面脚本"""
    scripts = [_APP_PY]
    pages = sorted(_PAGES_DIR.glob("*.py"))
    scripts.extend(pages)
    return scripts


@pytest.mark.parametrize("page_script", get_all_page_scripts(), ids=lambda p: p.name)
def test_page_renders_with_zero_exceptions(page_script: Path):
    """DevOps Gate 2: 页面必须 100% 渲染成功且无任何未捕获异常"""
    assert page_script.exists(), f"页面脚本不存在: {page_script}"

    at = AppTest.from_file(str(page_script), default_timeout=25).run()

    if at.exception:
        err_details = "\n".join([f"  • {e.message}" for e in at.exception])
        pytest.fail(
            f"页面 {page_script.name} 运行时抛出 {len(at.exception)} 项未捕获异常:\n{err_details}"
        )
