param([string]$Python = 'E:\miniconda\envs\xhs\python.exe')
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path $PSScriptRoot -Parent
Set-Location -LiteralPath $projectRoot
if (-not (Test-Path -LiteralPath '.venv\Scripts\python.exe')) {
    & $Python -m venv '.venv'
    if ($LASTEXITCODE -ne 0) { throw '虚拟环境创建失败，需要 Python 3.11' }
}
$projectPython = Join-Path $projectRoot '.venv\Scripts\python.exe'
& $projectPython -m pip install --index-url https://download.pytorch.org/whl/cu128 'torch==2.8.0'
if ($LASTEXITCODE -ne 0) { throw 'CUDA版PyTorch安装失败，勿继续使用CPU包' }
& $projectPython -m pip install -r 'requirements.lock'
if ($LASTEXITCODE -ne 0) { throw '项目依赖安装失败' }
& $projectPython -m pip install --no-deps -e '.'
if ($LASTEXITCODE -ne 0) { throw '项目安装失败' }
& $projectPython (Join-Path $PSScriptRoot 'download_model.py')
if ($LASTEXITCODE -ne 0) { throw '模型下载失败；资料不会丢失，可稍后重试' }
& $projectPython -m scripts.verify_gpu --output '.verification\gpu-setup.json' --local-model-path 'data\models\bge-small-zh-v1.5'
if ($LASTEXITCODE -ne 0) { throw 'GPU实际模型推理未通过，请查看 .verification\gpu-setup.json' }
Write-Output '安装和GPU验证完成。运行 scripts\start.ps1 启动。'
