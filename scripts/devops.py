# Copyright (c) 2026 Yy1 (yuan278501381) | MIT License
"""
scripts/devops.py - 统一工业级 DevOps 质量门禁 (CI) 与幂等部署 (CD) 引擎

将持续集成质量门禁 (Quality Gate) 与持续部署生命周期管理 (Idempotent Deployment) 收敛为统一的 DevOps 控制台。

支持的核心子命令 (Subcommands):
1. gate / ci / test:
   - 运行 7 大阶段质量门禁 (Ruff Lint, Format, Pyright, Git Diff, Pytest 覆盖率, Playwright E2E, 幂等性测试)。
2. deploy / start:
   - 运行强幂等性部署 (环境依赖同步, Git Hook 注入, 端口探活与自愈收敛, 守护进程健康检查)。
3. stop / restart:
   - 服务的平滑优雅停止与安全重启。
4. status / healthcheck:
   - 探活探测指定端口服务健康状态 (HTTP 200 /_stcore/health)。
5. pipeline / all:
   - 一键全链路流转：先执行 Gate 质量门禁，全部通过后自动执行 Deploy 幂等部署。
"""

import argparse
import hashlib
import json
import logging
import os
import platform
import socket
import subprocess
import sys
import time
import urllib.request
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["PYTHONUTF8"] = "1"

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_PORT = 8501
_HEALTH_PATH = "/_stcore/health"


# ===========================================================================
# 1. 数据契约与报告对象 (Data Contracts & Telemetry)
# ===========================================================================
@dataclass
class StepResult:
    """单个步骤/子系统的执行结果"""

    step_name: str
    status: str  # "CONVERGED_SUCCESS" | "SKIPPED_IDEMPOTENT" | "PASSED" | "FAILED"
    details: str
    elapsed_seconds: float


@dataclass
class DevOpsReport:
    """DevOps 统一执行报告"""

    trace_id: str
    command: str
    target_port: int
    system_arch: str
    os_name: str
    overall_status: str  # "SUCCESS" | "FAILED"
    is_idempotent_noop: bool
    steps: list[dict[str, Any]]
    total_elapsed_seconds: float
    timestamp: str


# ===========================================================================
# 2. 持续集成质量门禁执行器 (CI Quality Gate Runner)
# ===========================================================================
class QualityGateRunner:
    """统一质量门禁调度器 (7 大阶段)"""

    def __init__(self, trace_id: str, logger: logging.Logger):
        self.trace_id = trace_id
        self.logger = logger

    def run_step(self, step_name: str, cmd: list[str]) -> StepResult:
        self.logger.info(f"[CI GATE] 启动阶段: {step_name} | Command: {' '.join(cmd)}")
        t0 = time.time()
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUTF8"] = "1"

        res = subprocess.run(
            cmd,
            cwd=str(_PROJECT_ROOT),
            capture_output=False,
            env=env,
        )
        elapsed = time.time() - t0

        if res.returncode != 0:
            self.logger.error(
                f"[CI GATE] 阶段失败: {step_name} (退出码: {res.returncode}, 耗时: {elapsed:.2f}s)"
            )
            return StepResult(step_name, "FAILED", f"退出码 {res.returncode}", elapsed)
        else:
            self.logger.info(f"[CI GATE] 阶段成功: {step_name} (耗时: {elapsed:.2f}s)")
            return StepResult(step_name, "PASSED", "阶段检查全部通过", elapsed)

    def run_all_gates(self) -> list[StepResult]:
        steps: list[StepResult] = []
        gates = [
            ("Stage 1: Ruff 静态代码审查与 Linting", ["uv", "run", "ruff", "check", "."]),
            ("Stage 2: Ruff 格式一致性检查", ["uv", "run", "ruff", "format", "--check", "."]),
            ("Stage 3: Pyright 类型系统检查", ["uv", "run", "pyright"]),
            ("Stage 4: Git 差异空白卫生检查", ["git", "diff", "--check"]),
            (
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
            ),
            (
                "Stage 6: 真实浏览器端到端交互与延迟挂载门禁",
                ["uv", "run", "python", "tests/test_browser_pending_navigation.py"],
            ),
            (
                "Stage 7: DevOps 部署与运维幂等性门禁",
                ["uv", "run", "pytest", "tests/test_devops_idempotent_deploy.py", "-q"],
            ),
        ]

        for name, cmd in gates:
            step_res = self.run_step(name, cmd)
            steps.append(step_res)
            if step_res.status == "FAILED":
                break

        return steps


# ===========================================================================
# 3. 持续部署与生命周期引擎 (CD Idempotent Deployment Engine)
# ===========================================================================
class IdempotentDeployEngine:
    """工业级幂等部署与状态收敛引擎"""

    def __init__(
        self,
        port: int = _DEFAULT_PORT,
        trace_id: str | None = None,
        verbose: bool = False,
        headless: bool = True,
    ):
        self.port = port
        self.trace_id = (
            trace_id or f"devops-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
        )
        self.verbose = verbose
        self.headless = headless
        self._setup_logger()

    def _setup_logger(self) -> None:
        self.logger = logging.getLogger(f"DevOps[{self.trace_id}]")
        self.logger.setLevel(logging.DEBUG if self.verbose else logging.INFO)
        if not self.logger.handlers:
            handler = logging.StreamHandler(sys.stderr)
            formatter = logging.Formatter(
                f"[%(asctime)s] [%(levelname)s] [TraceID: {self.trace_id}] %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def log(self, level: str, message: str) -> None:
        lvl = getattr(logging, level.upper(), logging.INFO)
        self.logger.log(lvl, message)

    def is_port_listening(self, host: str = "127.0.0.1", port: int | None = None) -> bool:
        check_port = port or self.port
        try:
            with socket.create_connection((host, check_port), timeout=0.8):
                return True
        except (OSError, ConnectionRefusedError):
            return False

    def check_health(self, timeout: float = 3.0) -> bool:
        url = f"http://127.0.0.1:{self.port}{_HEALTH_PATH}"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": f"DevOps-{self.trace_id}"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.status == 200
        except Exception:
            return False

    def find_pids_on_port(self, port: int | None = None) -> list[int]:
        check_port = port or self.port
        pids: set[int] = set()

        if platform.system() == "Windows":
            try:
                cmd = ["netstat", "-ano"]
                out = subprocess.check_output(cmd, text=True, errors="ignore")
                for line in out.splitlines():
                    tokens = line.strip().split()
                    if len(tokens) >= 5 and tokens[0].upper() in ("TCP", "UDP"):
                        local_addr = tokens[1]
                        if local_addr.endswith(f":{check_port}"):
                            pid_str = tokens[-1]
                            if pid_str.isdigit():
                                pid = int(pid_str)
                                if pid > 0:
                                    pids.add(pid)
            except Exception as e:
                self.log("WARN", f"Windows netstat 扫描异常: {e}")
        else:
            try:
                cmd = ["lsof", "-t", f"-i:{check_port}"]
                out = subprocess.check_output(cmd, text=True, errors="ignore")
                for pid_str in out.split():
                    if pid_str.isdigit():
                        pids.add(int(pid_str))
            except Exception:
                try:
                    cmd = ["fuser", f"{check_port}/tcp"]
                    out = subprocess.check_output(cmd, text=True, errors="ignore")
                    for pid_str in out.split():
                        if pid_str.isdigit():
                            pids.add(int(pid_str))
                except Exception:
                    pass

        return sorted(pids)

    def terminate_process_safe(self, pid: int, timeout: float = 5.0) -> bool:
        self.log("INFO", f"正在平滑终止端口占用进程 PID: {pid}...")
        if platform.system() == "Windows":
            try:
                subprocess.run(
                    ["taskkill", "/PID", str(pid), "/T", "/F"],
                    capture_output=True,
                    timeout=timeout,
                )
                return True
            except Exception as e:
                self.log("WARN", f"taskkill PID {pid} 异常: {e}")
                return False
        else:
            try:
                import signal

                os.kill(pid, signal.SIGTERM)
                start = time.time()
                while time.time() - start < timeout:
                    try:
                        os.kill(pid, 0)
                        time.sleep(0.2)
                    except OSError:
                        return True
                os.kill(pid, signal.SIGKILL)
                return True
            except Exception as e:
                self.log("WARN", f"终止 PID {pid} 异常: {e}")
                return False

    def sync_environment(self) -> StepResult:
        t0 = time.time()
        self.log("INFO", "[STAGE 1/3] 检查 Python 运行环境与 uv 包依赖...")

        try:
            uv_ver = subprocess.check_output(["uv", "--version"], text=True).strip()
            self.log("DEBUG", f"发现包管理器: {uv_ver}")
        except Exception:
            self.log("FATAL", "系统未检测到 uv 包管理器，请先安装 uv。")
            return StepResult("EnvironmentSync", "FAILED", "uv 包管理器缺失", time.time() - t0)

        pyproject_file = _PROJECT_ROOT / "pyproject.toml"
        if not pyproject_file.exists():
            return StepResult(
                "EnvironmentSync", "FAILED", "未找到 pyproject.toml", time.time() - t0
            )

        try:
            env = os.environ.copy()
            env["PYTHONIOENCODING"] = "utf-8"
            env["PYTHONUTF8"] = "1"
            self.log("DEBUG", "执行 uv sync --all-extras 验证依赖一致性...")
            res = subprocess.run(
                ["uv", "sync", "--all-extras"],
                cwd=str(_PROJECT_ROOT),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=env,
            )
            if res.returncode != 0:
                err_msg = (res.stderr or "").strip()
                return StepResult(
                    "EnvironmentSync",
                    "FAILED",
                    f"uv sync 失败: {err_msg}",
                    time.time() - t0,
                )

            details = "Python 依赖环境已收敛至 pyproject.toml / uv.lock 最新状态"
            self.log("INFO", f"环境与依赖已处于目标状态 ({details})")
            return StepResult("EnvironmentSync", "CONVERGED_SUCCESS", details, time.time() - t0)
        except Exception as e:
            return StepResult("EnvironmentSync", "FAILED", str(e), time.time() - t0)

    def sync_git_hooks(self) -> StepResult:
        t0 = time.time()
        self.log("INFO", "[STAGE 2/3] 检查 Git Pre-Commit 左移质量门禁钩子...")

        git_dir = _PROJECT_ROOT / ".git"
        if not git_dir.exists():
            self.log("WARN", "当前目录非 Git 仓库根目录，跳过 Git Hook 注入。")
            return StepResult(
                "GitHooksSync", "SKIPPED_IDEMPOTENT", "非 Git 仓库，跳过", time.time() - t0
            )

        hooks_dir = git_dir / "hooks"
        hooks_dir.mkdir(parents=True, exist_ok=True)
        pre_commit_file = hooks_dir / "pre-commit"

        desired_content = """#!/bin/sh
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
        desired_hash = hashlib.sha256(desired_content.encode("utf-8")).hexdigest()

        if pre_commit_file.exists():
            current_content = pre_commit_file.read_text(encoding="utf-8", errors="ignore")
            current_hash = hashlib.sha256(current_content.encode("utf-8")).hexdigest()
            if current_hash == desired_hash:
                self.log(
                    "INFO",
                    "Git Pre-Commit 钩子内容与 SHA-256 校验一致，无需修改 (SKIPPED_IDEMPOTENT)。",
                )
                return StepResult(
                    "GitHooksSync",
                    "SKIPPED_IDEMPOTENT",
                    "Hook 内容一致，无需覆写",
                    time.time() - t0,
                )

        pre_commit_file.write_text(desired_content, encoding="utf-8")
        try:
            import stat

            st = os.stat(pre_commit_file)
            os.chmod(pre_commit_file, st.st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
        except Exception:
            pass

        self.log("INFO", f"已成功原子更新 Git Pre-Commit 钩子: {pre_commit_file}")
        return StepResult(
            "GitHooksSync", "CONVERGED_SUCCESS", "Hook 已更新并赋权", time.time() - t0
        )

    def deploy_service(self, force_restart: bool = False) -> StepResult:
        t0 = time.time()
        self.log("INFO", f"[STAGE 3/3] 检查 Streamlit 仪表板服务状态 (端口: {self.port})...")

        is_listening = self.is_port_listening()
        is_healthy = self.check_health(timeout=1.5) if is_listening else False

        if is_listening and is_healthy and not force_restart:
            self.log(
                "INFO",
                f"服务已在端口 {self.port} 处于健康运行状态 (HTTP 200)。无需重复启动，实现零副作用幂等复用 (NOOP_REUSED)。",
            )
            return StepResult(
                "ServiceDeploy",
                "SKIPPED_IDEMPOTENT",
                f"服务已在端口 {self.port} 健康运行，状态无需改变",
                time.time() - t0,
            )

        if is_listening:
            self.log(
                "WARN",
                f"端口 {self.port} 被占用 (健康状态: {is_healthy}, 强制重启: {force_restart})，正在收敛端口占用...",
            )
            pids = self.find_pids_on_port()
            for pid in pids:
                self.terminate_process_safe(pid)
            time.sleep(1.0)
            if self.is_port_listening():
                self.log("WARN", f"端口 {self.port} 仍被占用，等待系统释放 socket...")
                time.sleep(1.5)

        self.log(
            "INFO",
            f"正在后台启动 Streamlit 可视化仪表板 (Port: {self.port}, Headless: {self.headless})...",
        )
        cmd = [
            "uv",
            "run",
            "streamlit",
            "run",
            "dashboard/app.py",
            f"--server.port={self.port}",
            f"--server.headless={str(self.headless).lower()}",
            "--browser.gatherUsageStats=false",
        ]

        proc = subprocess.Popen(
            cmd,
            cwd=str(_PROJECT_ROOT),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
            if platform.system() == "Windows"
            else 0,
        )

        max_wait = 25.0
        start_wait = time.time()
        ready = False
        while time.time() - start_wait < max_wait:
            if self.check_health(timeout=1.0):
                ready = True
                break
            if proc.poll() is not None:
                return StepResult(
                    "ServiceDeploy",
                    "FAILED",
                    f"Streamlit 进程异常退出，退出码: {proc.returncode}",
                    time.time() - t0,
                )
            time.sleep(0.4)

        if not ready:
            return StepResult(
                "ServiceDeploy",
                "FAILED",
                f"服务启动后未能在 {max_wait}s 内通过健康检查",
                time.time() - t0,
            )

        self.log("INFO", f"服务已成功启动并在端口 {self.port} 通过健康检查 (PID: {proc.pid})！")
        return StepResult(
            "ServiceDeploy",
            "CONVERGED_SUCCESS",
            f"服务已启动并在端口 {self.port} 处于健康状态 (PID: {proc.pid})",
            time.time() - t0,
        )

    def stop_service(self) -> StepResult:
        t0 = time.time()
        self.log("INFO", f"执行服务停止操作 (端口: {self.port})...")
        if not self.is_port_listening():
            self.log(
                "INFO", f"端口 {self.port} 未在监听，服务本就不处于运行状态 (SKIPPED_IDEMPOTENT)。"
            )
            return StepResult("ServiceStop", "SKIPPED_IDEMPOTENT", "服务未在运行", time.time() - t0)

        pids = self.find_pids_on_port()
        for pid in pids:
            self.terminate_process_safe(pid)

        time.sleep(1.0)
        if self.is_port_listening():
            return StepResult(
                "ServiceStop", "FAILED", f"端口 {self.port} 仍未完全释放", time.time() - t0
            )

        self.log("INFO", f"服务已成功停止，端口 {self.port} 已完全释放。")
        return StepResult("ServiceStop", "CONVERGED_SUCCESS", "服务已停止", time.time() - t0)

    def execute(self, command: str = "deploy", force: bool = False) -> DevOpsReport:
        t_start = time.time()
        self.log(
            "INFO",
            f"================ 启动 DEVOPS 统一控制台 (Command: {command.upper()}) ================",
        )

        steps: list[StepResult] = []

        if command in ("gate", "ci", "test"):
            gate_runner = QualityGateRunner(self.trace_id, self.logger)
            steps = gate_runner.run_all_gates()
            overall = "SUCCESS" if all(s.status != "FAILED" for s in steps) else "FAILED"
            return self._build_report(command, steps, overall, t_start)

        elif command in ("deploy", "start", "restart"):
            step1 = self.sync_environment()
            steps.append(step1)
            if step1.status == "FAILED":
                return self._build_report(command, steps, "FAILED", t_start)

            step2 = self.sync_git_hooks()
            steps.append(step2)
            if step2.status == "FAILED":
                return self._build_report(command, steps, "FAILED", t_start)

            step3 = self.deploy_service(force_restart=(command == "restart" or force))
            steps.append(step3)
            overall = "SUCCESS" if step3.status != "FAILED" else "FAILED"
            return self._build_report(command, steps, overall, t_start)

        elif command == "stop":
            step = self.stop_service()
            steps.append(step)
            overall = "SUCCESS" if step.status != "FAILED" else "FAILED"
            return self._build_report(command, steps, overall, t_start)

        elif command in ("status", "healthcheck"):
            t0 = time.time()
            listening = self.is_port_listening()
            healthy = self.check_health(timeout=2.0) if listening else False
            status_str = (
                "HEALTHY" if healthy else ("LISTENING_UNHEALTHY" if listening else "STOPPED")
            )
            self.log("INFO", f"当前服务运行状态: {status_str} (Port: {self.port})")
            steps.append(
                StepResult(
                    "StatusCheck", "CONVERGED_SUCCESS", f"状态: {status_str}", time.time() - t0
                )
            )
            return self._build_report(command, steps, "SUCCESS", t_start)

        elif command in ("pipeline", "all"):
            # 1. 质量门禁阶段
            gate_runner = QualityGateRunner(self.trace_id, self.logger)
            gate_steps = gate_runner.run_all_gates()
            steps.extend(gate_steps)
            if any(s.status == "FAILED" for s in gate_steps):
                self.log("FATAL", "质量门禁未通过，熔断终止部署流程。")
                return self._build_report(command, steps, "FAILED", t_start)

            # 2. 部署阶段
            step1 = self.sync_environment()
            steps.append(step1)
            if step1.status == "FAILED":
                return self._build_report(command, steps, "FAILED", t_start)

            step2 = self.sync_git_hooks()
            steps.append(step2)
            if step2.status == "FAILED":
                return self._build_report(command, steps, "FAILED", t_start)

            step3 = self.deploy_service(force_restart=force)
            steps.append(step3)
            overall = "SUCCESS" if step3.status != "FAILED" else "FAILED"
            return self._build_report(command, steps, overall, t_start)

        else:
            self.log("FATAL", f"未知 command: {command}")
            return self._build_report(command, steps, "FAILED", t_start)

    def _build_report(
        self,
        command: str,
        steps: list[StepResult],
        overall_status: str,
        t_start: float,
    ) -> DevOpsReport:
        is_all_skipped = all(s.status == "SKIPPED_IDEMPOTENT" for s in steps) if steps else False
        report = DevOpsReport(
            trace_id=self.trace_id,
            command=command,
            target_port=self.port,
            system_arch=platform.machine(),
            os_name=f"{platform.system()} {platform.release()}",
            overall_status=overall_status,
            is_idempotent_noop=is_all_skipped,
            steps=[asdict(s) for s in steps],
            total_elapsed_seconds=round(time.time() - t_start, 3),
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        )
        self.log(
            "INFO",
            f"================ DEVOPS 执行完成: {overall_status} (耗时: {report.total_elapsed_seconds}s, 幂等无损跳过: {is_all_skipped}) ================",
        )
        return report


def main():
    parser = argparse.ArgumentParser(
        description="统一工业级 DevOps 质量门禁 (CI) 与幂等部署 (CD) 引擎",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
常用命令示例:
  uv run python scripts/devops.py gate                # 执行 7 大阶段 CI 质量门禁
  uv run python scripts/devops.py deploy              # 执行 幂等部署 与服务生命周期收敛
  uv run python scripts/devops.py pipeline            # 执行 完整 CI 门禁 + CD 部署流水线
  uv run python scripts/devops.py status              # 检查当前服务健康与端口监听状态
  uv run python scripts/devops.py stop                # 安全优雅停止服务并释放端口
        """,
    )
    parser.add_argument(
        "command",
        nargs="?",
        default="gate",
        choices=[
            "gate",
            "ci",
            "test",
            "deploy",
            "start",
            "restart",
            "stop",
            "status",
            "healthcheck",
            "pipeline",
            "all",
        ],
        help="执行子命令 (默认: gate)",
    )
    parser.add_argument("--port", type=int, default=_DEFAULT_PORT, help="目标服务端口 (默认: 8501)")
    parser.add_argument("--force", action="store_true", help="强制重启或重新拉起服务")
    parser.add_argument("--trace-id", type=str, default=None, help="自定义链路追踪 ID")
    parser.add_argument("--verbose", action="store_true", help="输出调试详细日志")
    parser.add_argument("--json", action="store_true", help="输出标准化 JSON 格式报告")
    parser.add_argument(
        "--no-headless", action="store_true", help="非无头模式（打开前端浏览器视窗）"
    )

    args = parser.parse_args()

    engine = IdempotentDeployEngine(
        port=args.port,
        trace_id=args.trace_id,
        verbose=args.verbose,
        headless=not args.no_headless,
    )

    report = engine.execute(command=args.command, force=args.force)

    if args.json:
        print(json.dumps(asdict(report), ensure_ascii=False, indent=2))

    sys.exit(0 if report.overall_status == "SUCCESS" else 1)


if __name__ == "__main__":
    main()
