# Copyright (c) 2026 Yy1 (yuan278501381) | MIT License
"""
scripts/devops_deploy.py - 持续部署生命周期入口 (已收敛至 scripts/devops.py)

兼容性委托入口：底层由统一的 scripts/devops.py 驱动。
"""

from scripts.devops import main as devops_main


def main():
    devops_main()


if __name__ == "__main__":
    main()
