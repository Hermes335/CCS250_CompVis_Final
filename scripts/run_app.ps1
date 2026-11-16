param(
    [string]$CheckpointDir = "",
    [int]$Port = 8000
)
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
if (-not (Test-Path ".\.venv\Scripts\Activate.ps1")) { throw "Missing .venv. Create it and install requirements first." }
. .\.venv\Scripts\Activate.ps1
$env:IMAGE_MODE = "real"
if (-not $env:IMAGE_TILE) { $env:IMAGE_TILE = "256" }
if ($CheckpointDir) { $env:IMAGE_CHECKPOINT_DIR = $CheckpointDir }
if ($env:IMAGE_CHECKPOINT_DIR) {
    foreach ($model in @("dncnn", "nafnet", "restormer")) {
        $file = Join-Path $env:IMAGE_CHECKPOINT_DIR "$model.pth"
        if (-not (Test-Path $file)) { throw "Missing checkpoint: $file" }
    }
}
python -m uvicorn app.main:app --host 127.0.0.1 --port $Port
