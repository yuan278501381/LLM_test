# Copyright (c) 2026 Yy1 (yuan278501381) | MIT License
"""
tests/test_devops_idempotent_deploy.py - DevOps 部署与运维生命周期幂等性 (Idempotency) 全链路自动化验证
"""

import json
import subprocess
from pathlib import Path

from scripts.devops import IdempotentDeployEngine

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_devops_deploy_engine_environment_idempotency():
    """验证环境同步机制具备强幂等性 (连续多次运行均收敛为成功且不破坏环境)"""
    engine = IdempotentDeployEngine(port=8599, trace_id="test-env-idempotent-001")
    res1 = engine.sync_environment()
    assert res1.status in ("CONVERGED_SUCCESS", "SKIPPED_IDEMPOTENT")

    res2 = engine.sync_environment()
    assert res2.status in ("CONVERGED_SUCCESS", "SKIPPED_IDEMPOTENT")


def test_devops_deploy_engine_git_hooks_idempotency(tmp_path: Path):
    """验证 Git 钩子在 SHA-256 一致时无损跳过 (SKIPPED_IDEMPOTENT)，漂移时自动收敛"""
    engine = IdempotentDeployEngine(port=8599, trace_id="test-hook-idempotent-002")

    # 首次同步现有 hooks
    res1 = engine.sync_git_hooks()
    assert res1.status in ("CONVERGED_SUCCESS", "SKIPPED_IDEMPOTENT")

    # 再次同步，由于内容完全匹配，必定跳过 (SKIPPED_IDEMPOTENT)
    res2 = engine.sync_git_hooks()
    assert res2.status == "SKIPPED_IDEMPOTENT"
    assert "无需修改" in res2.details or "无需覆写" in res2.details


def test_devops_deploy_engine_healthcheck_and_status():
    """验证健康检查与状态探测返回正确格式"""
    engine = IdempotentDeployEngine(port=8598, trace_id="test-status-003")
    report = engine.execute(command="status")
    assert report.overall_status == "SUCCESS"
    assert report.command == "status"
    assert report.trace_id == "test-status-003"
    assert len(report.steps) == 1
    assert "状态:" in report.steps[0]["details"]


def test_devops_deploy_engine_service_idempotent_noop():
    """验证当端口上已有健康服务时，执行 deploy 产生零副作用幂等复用 (SKIPPED_IDEMPOTENT)"""
    engine = IdempotentDeployEngine(port=8501, trace_id="test-service-idempotent-004")

    # 如果当前 8501 正在运行且健康，调用 deploy_service 必须返回 SKIPPED_IDEMPOTENT
    if engine.is_port_listening() and engine.check_health(timeout=2.0):
        step_res = engine.deploy_service(force_restart=False)
        assert step_res.status == "SKIPPED_IDEMPOTENT"
        assert "无需改变" in step_res.details or "健康运行" in step_res.details


def test_devops_deploy_cli_json_report_contract():
    """验证 CLI 接口在 `--json` 模式下输出符合 OpenAPI/DevOps 契约的标准化 JSON 报告"""
    cmd = [
        "uv",
        "run",
        "python",
        "scripts/devops.py",
        "status",
        "--port",
        "8501",
        "--json",
        "--trace-id",
        "test-trace-contract-json",
    ]
    out = subprocess.check_output(cmd, cwd=str(_PROJECT_ROOT), text=True)
    report_dict = json.loads(out)

    assert report_dict["trace_id"] == "test-trace-contract-json"
    assert report_dict["command"] == "status"
    assert report_dict["target_port"] == 8501
    assert "system_arch" in report_dict
    assert "os_name" in report_dict
    assert "total_elapsed_seconds" in report_dict
    assert "steps" in report_dict
    assert isinstance(report_dict["steps"], list)
