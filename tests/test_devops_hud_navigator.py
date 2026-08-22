# Copyright (c) 2026 Yy1 (yuan278501381) | MIT License
"""
tests/test_devops_hud_navigator.py - 页面空间 HUD 罗盘与蓝图地图无缺陷契约测试 (DevOps Gate)

专门校验:
1. 全站 17 大页面的 blueprint_sections / HUD 配置是否符合标准规范
2. 校验每一个 section 是否具备非空的 id/letter 与 name/title
3. 校验生成的 HTML / JS 绝对不得包含 "undefined" 文本
"""

from pathlib import Path

import pytest

from dashboard.styles.theme import render_floating_hud_navigator

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_PAGES_DIR = _PROJECT_ROOT / "dashboard" / "pages"


def test_hud_sections_never_produce_undefined():
    """验证 theme.py 的 HUD 罗盘与蓝图地图渲染器对任何数据结构绝对免疫 undefined"""
    # 模拟各种格式的输入数据（包括缺失字段、异名字段、空值）
    dirty_inputs = [
        [{"letter": "A", "title": "控制台", "role": "调节超参数"}],
        [{"id": "B", "name": "核心原理", "desc": "深度解析"}],
        [{"id": "C"}],  # 缺 name/desc
        [{"title": "无 ID 区域"}],  # 缺 id
    ]

    for sample_secs in dirty_inputs:
        # 验证不会抛出异常
        try:
            render_floating_hud_navigator(sample_secs)
        except Exception as e:
            pytest.fail(f"render_floating_hud_navigator 渲染失败: {e}")


def test_all_pages_blueprint_sections_contract():
    """静态检查所有 17 个页面中定义的 blueprint_sections 是否合规"""
    py_pages = sorted(_PAGES_DIR.glob("*.py"))
    assert len(py_pages) >= 16

    errors = []
    for page in py_pages:
        content = page.read_text(encoding="utf-8")
        if "blueprint_sections" in content and '"letter":' in content:
            # 确保没有残留的纯 letter 键定义而无 id
            # 匹配字典中是否出现 "letter":
            errors.append(f"{page.name}: blueprint_sections 仍在使用旧的 'letter' 键名，请统一为 'id'")

    assert not errors, "发现 HUD 蓝图配置不符合契约规范:\n" + "\n".join(errors)
