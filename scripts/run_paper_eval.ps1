# Quick paper experiments that only need checkpoint + synced Drive data.
# Usage (from repo root, preprocessing branch):
#   $env:DATA_ROOT = "G:/My Drive/AI-Innovation-Data"
#   $env:CHECKPOINT = "outputs/ai_audio_detection/checkpoint_best.pt"
#   .\scripts\run_paper_eval.ps1

$ErrorActionPreference = "Stop"
$Config = "configs/gpu_local.yaml"
$Checkpoint = if ($env:CHECKPOINT) { $env:CHECKPOINT } else { "outputs/ai_audio_detection/checkpoint_best.pt" }

if (-not (Test-Path $Checkpoint)) {
    Write-Host "ERROR: Checkpoint not found: $Checkpoint"
    Write-Host "Download from Drive checkpoints/ai_audio_detection/ or set `$env:CHECKPOINT"
    exit 1
}

if ($env:DATA_ROOT) {
    Write-Host "Using DATA_ROOT=$env:DATA_ROOT (patch gpu_local.yaml manually if paths differ)"
}

Write-Host "`n=== §5.2 Full eval + metrics.json ==="
python scripts/evaluate.py `
    --config $Config `
    --checkpoint $Checkpoint `
    --output-dir outputs/eval/full_model `
    --label "Ours (full)"

Write-Host "`n=== §5.4 Codec robustness sweep ==="
python scripts/eval_codec.py `
    --config $Config `
    --checkpoint $Checkpoint `
    --output-dir outputs/eval/codec_sweep

Write-Host "`n=== §7.1 Attention visualization ==="
python scripts/attention_viz.py `
    --config $Config `
    --checkpoint $Checkpoint `
    --output-dir outputs/figures/attention

Write-Host "`n=== §7.2 Grad-CAM ==="
python scripts/gradcam.py `
    --config $Config `
    --checkpoint $Checkpoint `
    --output-dir outputs/figures/gradcam `
    --auto-pick

Write-Host "`nDone. Check:"
Write-Host "  outputs/eval/full_model/metrics.json"
Write-Host "  outputs/eval/codec_sweep/codec_metrics.json"
Write-Host "  outputs/figures/attention/"
Write-Host "  outputs/figures/gradcam/"
