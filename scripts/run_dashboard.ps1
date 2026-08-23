#!/usr/bin/env pwsh
# Copyright (c) 2026 Yy1 (yuan278501381) | MIT License
# 一键幂等启动可视化仪表板

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path $PSScriptRoot -Parent
Set-Location $ProjectRoot

Write-Host "[DEPLOY] 启动 NN Playground 可视化仪表板 (幂等环境与服务检查)..." -ForegroundColor Cyan
Write-Host "         目标地址: http://localhost:8501" -ForegroundColor Green
Write-Host ""

uv run python scripts/devops.py deploy --port 8501
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
