# Copyright (c) 2026 Yy1 (yuan278501381) | MIT License
"""
scripts/setup_git_hooks.py - 本地 Git Pre-Commit / Pre-Push 移情左移门禁安装器 (Shift-Left Guard)

在本地 `.git/hooks/pre-commit` 中注入自动化门禁脚本：
在开发者执行 `git commit` 的瞬间，毫秒级执行 AST 静态导入校验与 API 契约扫描。
在本地提交前运行可复现检查；钩子可被绕过，因此 CI 仍是必要门禁。
"""

import os
import stat
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_GIT_HOOKS_DIR = _PROJECT_ROOT / ".git" / "hooks"


def install_hooks():
    if not _GIT_HOOKS_DIR.exists():
        print(f"未找到 .git 目录: {_GIT_HOOKS_DIR}")
        return

    pre_commit_path = _GIT_HOOKS_DIR / "pre-commit"
    hook_script = """#!/bin/sh
# 2026 Enterprise DevOps Shift-Left Pre-Commit Gate
echo "=================================================="
echo "[GUARD // 门禁]   Running Local Pre-Commit DevOps Quality Gate..."
echo "=================================================="

uv run python scripts/devops.py gate
if [ $? -ne 0 ]; then
    echo "[FAIL]  DevOps Quality Gate failed! Commit aborted."
    exit 1
fi

echo "[PASS]  Pre-Commit Quality Gate Passed! Proceeding with commit."
exit 0
"""
    pre_commit_path.write_text(hook_script, encoding="utf-8")

    # 授予可执行权限
    st_mode = os.stat(pre_commit_path).st_mode
    os.chmod(pre_commit_path, st_mode | stat.S_IEXEC)
    print(f"Git Pre-Commit Hook installed successfully at {pre_commit_path}!")


if __name__ == "__main__":
    install_hooks()
