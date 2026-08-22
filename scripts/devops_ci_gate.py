# Copyright (c) 2026 Yy1 (yuan278501381) | MIT License
"""
scripts/devops_ci_gate.py - 可复现的静态检查、分层测试与差异卫生门禁

统一执行格式、静态分析、差异卫生、契约、页面启动、控件采样、性能与覆盖率门禁。
"""

import os
import subprocess
import sys
import time

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["PYTHONUTF8"] = "1"


def run_step(step_name: str, cmd: list[str]) -> None:
    print(f"\n{'=' * 70}")
    print(f"[CI/CD STAGE] {step_name}")
    print(f"   Command: {' '.join(cmd)}")
    print(f"{'=' * 70}")

    start_time = time.time()
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    result = subprocess.run(cmd, capture_output=False, env=env)
    elapsed = time.time() - start_time

    if result.returncode != 0:
        print(f"\n[FAILED] {step_name} 失败！耗时: {elapsed:.2f}s")
        sys.exit(result.returncode)
    else:
        print(f"\n[PASSED] {step_name} 成功！耗时: {elapsed:.2f}s")


def main():
    print("=" * 70)
    print("[DEVOPS PIPELINE] STARTING REPRODUCIBLE QUALITY GATE")
    print("=" * 70)

    # 1. 静态规范与 Lint
    run_step("Stage 1: Ruff 静态代码审查与 Linting", ["uv", "run", "ruff", "check", "."])
    run_step("Stage 2: Ruff 格式一致性", ["uv", "run", "ruff", "format", "--check", "."])
    run_step("Stage 3: Pyright 类型检查", ["uv", "run", "pyright"])
    run_step("Stage 4: Git 差异空白卫生", ["git", "diff", "--check"])

    # 2. 静态导入与符号契约
    run_step(
        "Stage 5: AST 静态跨模块 Import 完整性校验 (Gate 5)",
        ["uv", "run", "pytest", "tests/test_devops_import_integrity.py", "-v"],
    )

    # 3. 公共 API 契约锁
    run_step(
        "Stage 6: 公共组件库 API 符号契约守卫 (Gate 6)",
        ["uv", "run", "pytest", "tests/test_devops_api_contracts.py", "-v"],
    )

    # 4. 全局 0-Emoji 物理硬门禁
    run_step(
        "Stage 7: 全局 Emoji 规范门禁 (Gate 7)",
        ["uv", "run", "pytest", "tests/test_devops_zero_emoji_policy.py", "-v"],
    )

    # 5. 页面 HUD 罗盘与蓝图契约
    run_step(
        "Stage 8: 页面空间 HUD 与导航契约 (Gate 8)",
        ["uv", "run", "pytest", "tests/test_devops_hud_navigator.py", "-v"],
    )

    # 6. 全页面动态沙盒冒烟
    run_step(
        "Stage 9: 全站 17 页面启动无异常测试 (Gate 9)",
        ["uv", "run", "pytest", "tests/test_devops_smoke_pages.py", "-v"],
    )

    # 7. UI 控件代表值与边界值采样
    run_step(
        "Stage 10: UI 控件代表值与边界值采样 (Gate 10)",
        ["uv", "run", "pytest", "tests/test_devops_widget_fuzzing.py", "-v"],
    )

    # 8. 算法性能预算守卫
    run_step(
        "Stage 11: 核心算子性能预算与延迟守卫 (Gate 11)",
        ["uv", "run", "pytest", "tests/test_devops_performance_budget.py", "-v"],
    )

    # 9. 全量核心回归测试套件 (包含 80% 覆盖率门禁)
    run_step(
        "Stage 12: 全量回归与分支覆盖率门禁 (Gate 12)",
        [
            "uv",
            "run",
            "pytest",
            "--cov=nn_core",
            "--cov=datasets",
            "--cov-branch",
            "--cov-report=term-missing",
            "-q",
        ],
    )

    print("\n" + "=" * 70)
    print("[SUCCESS] ALL CONFIGURED QUALITY GATES PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()
