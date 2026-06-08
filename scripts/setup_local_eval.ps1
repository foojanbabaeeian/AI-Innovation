# One-time setup: point eval scripts at your synced Drive + checkpoint.
# Usage:
#   $env:DATA_ROOT = "G:/My Drive/AI-Innovation-Data"   # your synced path
#   .\scripts\setup_local_eval.ps1
#   .\scripts\run_paper_eval.ps1

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path $PSScriptRoot -Parent

$DataRoot = $env:DATA_ROOT
if (-not $DataRoot) {
    Write-Host "Set DATA_ROOT first, e.g.:"
    Write-Host '  $env:DATA_ROOT = "G:/My Drive/AI-Innovation-Data"'
    exit 1
}

if (-not (Test-Path "$DataRoot\Processed")) {
    Write-Host "ERROR: Processed/ not found under $DataRoot"
    Write-Host "Add AI-Innovation-Data from Drive and wait for sync."
    exit 1
}

# Patch gpu_local.yaml data_root (Windows-safe forward slashes)
$Config = Join-Path $RepoRoot "configs\gpu_local.yaml"
$DataRootYaml = ($DataRoot -replace '\\', '/')
(Get-Content $Config -Raw) -replace '(?m)^  data_root: .*', "  data_root: `"$DataRootYaml`"" | Set-Content $Config -NoNewline
Write-Host "Updated configs/gpu_local.yaml data_root -> $DataRootYaml"

$CkptSrc = "c:\Users\anhle\Downloads\checkpoint_best.pt"
$CkptDst = Join-Path $RepoRoot "outputs\ai_audio_detection\checkpoint_best.pt"
if (Test-Path $CkptSrc) {
    New-Item -ItemType Directory -Force -Path (Split-Path $CkptDst) | Out-Null
    Copy-Item $CkptSrc $CkptDst -Force
    Write-Host "Checkpoint -> $CkptDst"
} elseif (-not (Test-Path $CkptDst)) {
    Write-Host "ERROR: No checkpoint at $CkptDst or Downloads"
    exit 1
}

Write-Host "`nReady. Run: .\scripts\run_paper_eval.ps1"
