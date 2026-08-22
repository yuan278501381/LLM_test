# Copyright (c) 2026 Yy1 (yuan278501381) | MIT License
"""
tests/test_devops_import_integrity.py - AST 静态符号级跨文件导入完整性测试 (DevOps Gate 1)

通过 Python 抽象语法树 (AST) 静态遍历全站所有 Python 源码文件，
自动提取每一个 `from module import symbol` 语句，直接验证被导入模块中是否存在对应符号。
毫秒级杜绝任何因重构、函数重命名或误删导致的悬空导入 (Dangling / Broken Import)！
"""

import ast
import importlib
import os
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def get_all_python_files() -> list[Path]:
    """获取项目内所有 Python 源码文件 (排除 .venv, .git, .pytest_cache)"""
    py_files = []
    for root, dirs, files in os.walk(_PROJECT_ROOT):
        dirs[:] = [
            d
            for d in dirs
            if not d.startswith(".") and d not in {".venv", "venv", "__pycache__", "build", "dist"}
        ]
        for file in files:
            if file.endswith(".py"):
                py_files.append(Path(root) / file)
    return sorted(py_files)


def test_cross_module_import_integrity():
    """DevOps Gate 1: 对当前收集到的 Python 文件执行静态 import 符号解析。"""
    py_files = get_all_python_files()
    assert len(py_files) >= 20, f"发现的 Python 源码文件数量异常: {len(py_files)}"

    broken_imports = []

    for file_path in py_files:
        try:
            content = file_path.read_text(encoding="utf-8")
            tree = ast.parse(content, filename=str(file_path))
        except Exception as e:
            broken_imports.append((str(file_path), 0, "SYNTAX_ERROR", str(e)))
            continue

        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                module_name = node.module
                if not module_name:
                    continue

                # 仅针对项目自身模块 (dashboard, nn_core) 进行深度符号反射校验
                if module_name.startswith("dashboard") or module_name.startswith("nn_core"):
                    try:
                        mod = importlib.import_module(module_name)
                    except Exception as exc:
                        broken_imports.append(
                            (
                                str(file_path.relative_to(_PROJECT_ROOT)),
                                node.lineno,
                                f"from {module_name} import ...",
                                f"模块导入失败: {exc}",
                            )
                        )
                        continue

                    for alias in node.names:
                        sym_name = alias.name
                        if sym_name == "*":
                            continue
                        if not hasattr(mod, sym_name):
                            broken_imports.append(
                                (
                                    str(file_path.relative_to(_PROJECT_ROOT)),
                                    node.lineno,
                                    f"from {module_name} import {sym_name}",
                                    f"符号 '{sym_name}' 在模块 '{module_name}' 中不存在！",
                                )
                            )

    error_msg = ""
    if broken_imports:
        error_msg = f"\n[DevOps Gate 1] 发现 {len(broken_imports)} 处断裂的悬空 import 引用:\n"
        for file, lineno, stmt, err in broken_imports:
            error_msg += f"  - {file}:{lineno} -> {stmt} ({err})\n"

    assert not broken_imports, error_msg
