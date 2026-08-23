#!/usr/bin/env pwsh
# Copyright (c) 2026 Yy1 (yuan278501381) | MIT License
# 一键幂等性 DevOps 部署、门禁与运维控制脚本

param (
    [Parameter(Position = 0)]
    [ValidateSet("deploy", "gate", "pipeline", "restart", "stop", "status", "healthcheck")]
    [string]$Command = "deploy",

    [Parameter()]
    [int]$Port = 8501,

    [Parameter()]
    [switch]$Force,

    [Parameter()]
    [switch]$Verbose,

    [Parameter()]
    [switch]$Json
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path $PSScriptRoot -Parent
Set-Location $ProjectRoot

$ArgsList = @("run", "python", "scripts/devops.py", $Command, "--port", $Port)

if ($Force) {
    $ArgsList += "--force"
}
if ($Verbose) {
    $ArgsList += "--verbose"
}
if ($Json) {
    $ArgsList += "--json"
}

uv @ArgsList
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
