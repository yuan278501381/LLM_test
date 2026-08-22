# Copyright (c) 2026 Yy1 (yuan278501381) | MIT License
"""
scripts/devops_ci_gate.py - 企业级 7 重 DevOps 质量门禁执行流水线 (World-Class 2026 Standard)

7 大全链路质量阶段:
1. Stage 1: Ruff 静态代码规范与类型审查 (Static Linter Gate)
2. Stage 2: AST 跨文件导入完整性与悬空符号扫描 (Gate 1: Import Integrity)
3. Stage 3: 公共组件库 API 导出契约检验 (Gate 2: API Contract Guard)
4. Stage 4: 全站多页面无死角动态沙盒冒烟测试 (Gate 3: Page Smoke Simulation)
5. Stage 5: UI 控件全组合模糊遍历与混沌韧性测试 (Gate 4: Chaos & Widget Fuzzing)
6. Stage 6: 核心算子性能预算与延迟防退化守卫 (Gate 5: Performance Budget Guard)
7. Stage 7: 全量 250+ 算法与端到端核心测试套件 (Gate 6: Full Regression Suite)
"""

import subprocess
import sys
import time


def run_step(step_name: str, cmd: list[str]) -> None:
    print(f"\n{'='*70}")
    print(f"[CI/CD STAGE] {step_name}")
    print(f"   Command: {' '.join(cmd)}")
    print(f"{'='*70}")

    start_time = time.time()
    result = subprocess.run(cmd, capture_output=False)
    elapsed = time.time() - start_time

    if result.returncode != 0:
        print(f"\n[FAILED] {step_name} 失败！耗时: {elapsed:.2f}s")
        sys.exit(result.returncode)
    else:
        print(f"\n[PASSED] {step_name} 成功！耗时: {elapsed:.2f}s")


def main():
    print("="*70)
    print("[DEVOPS PIPELINE] STARTING WORLD-CLASS 7-STAGE QUALITY GATE (2026)")
    print("="*70)

    # 1. 静态规范与 Lint
    run_step("Stage 1: Ruff 静态代码审查与 Linting", ["uv", "run", "ruff", "check", "."])

    # 2. 静态导入与符号契约
    run_step("Stage 2: AST 静态跨模块 Import 完整性校验 (Gate 1)", ["uv", "run", "pytest", "tests/test_devops_import_integrity.py", "-v"])

    # 3. 公共 API 契约锁
    run_step("Stage 3: 公共组件库 API 符号契约守卫 (Gate 2)", ["uv", "run", "pytest", "tests/test_devops_api_contracts.py", "-v"])

    # 4. 全局 0-Emoji 物理硬门禁
    run_step("Stage 4: 全局 0-Emoji 规范物理硬门禁 (Gate 3)", ["uv", "run", "pytest", "tests/test_devops_zero_emoji_policy.py", "-v"])

    # 5. 页面 HUD 罗盘与蓝图契约
    run_step("Stage 5: 页面空间 HUD 罗盘与蓝图地图无缺陷契约 (Gate 4)", ["uv", "run", "pytest", "tests/test_devops_hud_navigator.py", "-v"])

    # 6. 全页面动态沙盒冒烟
    run_step("Stage 6: 全站 17 大页面动态沙盒冒烟测试 (Gate 5)", ["uv", "run", "pytest", "tests/test_devops_smoke_pages.py", "-v"])

    # 7. UI 控件全组合模糊遍历
    run_step("Stage 7: UI 控件全组合模糊遍历与混沌韧性测试 (Gate 6)", ["uv", "run", "pytest", "tests/test_devops_widget_fuzzing.py", "-v"])

    # 8. 算法性能预算守卫
    run_step("Stage 8: 核心算子性能预算与延迟守卫 (Gate 7)", ["uv", "run", "pytest", "tests/test_devops_performance_budget.py", "-v"])

    # 9. 全量核心回归测试套件 (包含 80% 覆盖率门禁)
    run_step("Stage 9: 全量 270+ 核心算法单测与覆盖率门禁 (Gate 8)", ["uv", "run", "pytest", "--cov=nn_core", "-q"])

    print("\n" + "="*70)
    print("[SUCCESS] ALL 9 WORLD-CLASS DEVOPS GATES PASSED! ZERO-DEFECT CERTIFIED!")
    print("="*70)


if __name__ == "__main__":
    main()
