# Copyright (c) 2026 Yy1 (yuan278501381) | MIT License
"""
scripts/devops_ci_gate.py - 持续集成质量门禁入口 (已收敛至 scripts/devops.py)

兼容性委托入口：底层由统一的 scripts/devops.py gate 驱动。
"""

import sys

from scripts.devops import IdempotentDeployEngine


def main():
    engine = IdempotentDeployEngine()
    report = engine.execute(command="gate")
    sys.exit(0 if report.overall_status == "SUCCESS" else 1)


if __name__ == "__main__":
    main()
