param(
    [switch]$PrepareOnly,
    [int]$MaxCells = 0
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"
$OutputDir = "D:\lewm_runs\bridgeengine_head_to_head\run_100"
$Manifest = Join-Path $OutputDir "pi07_command_manifest.json"
$Snapshot = "snap_2026_05_11_1dde3edf5d_human_gold_labels"
$PlanDir = Join-Path $RepoRoot "head_to_head_results\preregistered_100"

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Python interpreter not found: $Python"
}

& $Python -m bridgeengine.benchmark.pi07_fixed prepare `
    --snapshot $Snapshot `
    --output-dir $OutputDir `
    --plan-dir $PlanDir `
    --sizes 25 50 100 `
    --seeds 42 137 256 `
    --max-epochs 20 `
    --batch-size 16 `
    --lr 5e-5
if ($LASTEXITCODE -ne 0) {
    throw "pi0.7 manifest preparation failed with exit code $LASTEXITCODE"
}

if ($PrepareOnly) {
    Write-Host "PrepareOnly set. Manifest written: $Manifest"
    exit 0
}

$RunArgs = @(
    "-m", "bridgeengine.benchmark.pi07_fixed", "run-manifest",
    "--manifest", $Manifest
)
if ($MaxCells -gt 0) {
    $RunArgs += @("--max-cells", "$MaxCells")
}

& $Python @RunArgs
if ($LASTEXITCODE -ne 0) {
    throw "pi0.7 head-to-head run failed with exit code $LASTEXITCODE"
}

& $Python -m bridgeengine.benchmark.pi07_fixed summarize --output-dir $OutputDir
if ($LASTEXITCODE -ne 0) {
    throw "pi0.7 summary failed with exit code $LASTEXITCODE"
}
