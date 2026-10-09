param([int]$Port = 8081)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path $PSScriptRoot -Parent
Set-Location -LiteralPath $projectRoot
$projectPython = Join-Path $projectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $projectPython)) { throw '请先运行 scripts\setup.ps1' }
Write-Output "个人回复助手：http://127.0.0.1:$Port"
& $projectPython -m uvicorn app.main:app --host 127.0.0.1 --port $Port --no-access-log
exit $LASTEXITCODE
