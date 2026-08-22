# Copyright (c) 2026 Yy1 (yuan278501381) | MIT License
"""
tests/test_devops_smoke_pages.py - 全站多页面无死角动态自动发现冒烟测试 (DevOps Gate 2)

自动扫描 `dashboard/pages/*.py` 和 `dashboard/app.py`，
利用 Streamlit 官方 AppTest 无头沙盒框架对全站每一个页面执行端到端仿真渲染。
测试按当前页面目录自动收集并验证启动无未捕获异常；它不证明内容或交互正确。
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
def test_page_starts_without_uncaught_exceptions(page_script: Path):
    """DevOps Gate 2: 当前收集到的页面应启动且无未捕获异常。"""
    assert page_script.exists(), f"页面脚本不存在: {page_script}"

    at = AppTest.from_file(str(page_script), default_timeout=25).run()

    if at.exception:
        err_details = "\n".join([f"  • {e.message}" for e in at.exception])
        pytest.fail(
            f"页面 {page_script.name} 运行时抛出 {len(at.exception)} 项未捕获异常:\n{err_details}"
        )
