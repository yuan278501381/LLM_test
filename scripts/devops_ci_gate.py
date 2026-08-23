# Copyright (c) 2026 Yy1 (yuan278501381) | MIT License
"""
scripts/devops_ci_gate.py - 可复现的静态检查、分层测试与差异卫生门禁

统一执行格式、静态分析、差异卫生，以及一次不重复的全量测试与覆盖率门禁。
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

    # 契约、页面、控件、性能与数值测试均由这一次全量运行收集，避免重复执行。
    run_step(
        "Stage 5: 全量回归与分支覆盖率门禁",
        [
            "uv",
            "run",
            "pytest",
            "--cov=nn_core",
            "--cov=datasets",
            "--cov-branch",
            "--cov-report=term-missing",
            "--basetemp=.pytest_tmp",
            "-q",
        ],
    )

    # 真实 Chromium 浏览器端到端交互与延迟导航重试门禁
    run_step(
        "Stage 6: 真实浏览器端到端交互与延迟挂载门禁",
        ["uv", "run", "python", "tests/test_browser_pending_navigation.py"],
    )

    print("\n" + "=" * 70)
    print("[SUCCESS] ALL CONFIGURED QUALITY GATES PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()
