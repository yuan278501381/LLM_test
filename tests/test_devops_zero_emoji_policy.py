# Copyright (c) 2026 Yy1 (yuan278501381) | MIT License
"""
tests/test_devops_zero_emoji_policy.py - 全局 0-Emoji 物理硬门禁校验 (DevOps Gate)

根据用户全局规范 RULE[user_global]：
代码库、文档字符串、UI 标签、图表标题及注释中严禁出现任何 Emoji 表情字符。
统一采用专业技术标签或 100% 矢量 SVG 图标替代。
本测试遍历全站所有 Python 源码文件，0.1 秒内严密监控并杜绝任何 Emoji 字符渗入。
"""

import re
from pathlib import Path

# Match standard emojis
EMOJI_REGEX = re.compile(
    r'[\U0001F300-\U0001F9FF]|[\U0001FA00-\U0001FAFF]|[\u2600-\u26FF]|[\u2700-\u27BF]|[\u231A-\u231B]|[\u23E9-\u23EC]|[\u23F0]|[\u23F3]'
)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_zero_emoji_policy_across_entire_repo():
    """DevOps Gate: 全项目 Python 源码必须 100% 零 Emoji 违规"""
    violations = []

    for p in _PROJECT_ROOT.rglob("*.py"):
        if any(part in str(p) for part in [".venv", "venv", ".git", "__pycache__", "build", "dist"]):
            continue
        content = p.read_text(encoding="utf-8")
        matches = EMOJI_REGEX.findall(content)
        if matches:
            violations.append(f"{p.relative_to(_PROJECT_ROOT)}: 发现非法 Emoji 字符 -> {set(matches)}")

    assert not violations, "\n[DevOps Gate] 发现 Emoji 违规项:\n" + "\n".join(violations)
