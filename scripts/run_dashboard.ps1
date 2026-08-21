#!/usr/bin/env pwsh
# Copyright (c) 2026 Yy1 (yuan278501381) | MIT License
# 一键启动可视化仪表板

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path $PSScriptRoot -Parent
Set-Location $ProjectRoot

Write-Host "🧠 启动 NN Playground 可视化仪表板..." -ForegroundColor Cyan
Write-Host "   地址: http://localhost:8501" -ForegroundColor Green
Write-Host ""

uv run streamlit run dashboard/app.py --server.port 8501 --server.headless true
