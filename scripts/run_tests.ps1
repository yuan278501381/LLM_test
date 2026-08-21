#!/usr/bin/env pwsh
# Copyright (c) 2026 Yy1 (yuan278501381) | MIT License
# 运行全部单元测试

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path $PSScriptRoot -Parent
Set-Location $ProjectRoot

Write-Host "🧪 运行全部单元测试..." -ForegroundColor Cyan
Write-Host ""

uv run pytest tests/ -v --tb=short
