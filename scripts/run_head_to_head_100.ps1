param(
    [string]$Snapshot = "snap_2026_05_11_1dde3edf5d_human_gold_labels",
    [string]$OutputDir = "head_to_head_results\run_100",
    [switch]$PrepareOnly,
    [switch]$SkipCvAux,
    [switch]$SkipPi07
)

$ErrorActionPreference = "Stop"
$Python = ".\.venv\Scripts\python.exe"

if (-not (Test-Path $Python)) {
    throw "Missing venv Python at $Python. Run from the annotation_pipeline repo root."
}

Write-Host "Verifying LeWM depth/track signal files..."
& $Python -m bridgeengine.benchmark.head_to_head_runner verify-signals `
    --manifest D:\bridgedata_v2_subset\manifest_100.json `
    --local-root D:\bridgedata_v2_subset `
    --plan-dir head_to_head_results\preregistered_100

Write-Host "Preparing fixed-split HDF5 datasets and LeWM configs..."
& $Python -m bridgeengine.benchmark.head_to_head_runner prepare `
    --plan-dir head_to_head_results\preregistered_100 `
    --source-h5 D:\bridgedata_v2_subset\datasets\bridgedata_v2_100ep.h5 `
    --manifest D:\bridgedata_v2_subset\manifest_100.json `
    --output-dir $OutputDir `
    --lewm-root C:\Users\Kevin\projects\LeWM_testbed `
    --sizes 25 50 100 `
    --seeds 42 137 256 `
    --max-epochs 20 `
    --batch-size 16 `
    --lr 5e-5

if ($PrepareOnly) {
    Write-Host "PrepareOnly set. Stop before long training."
    exit 0
}

if (-not $SkipCvAux) {
    Write-Host "Running native LeWM CV aux-head cells. This is the long part."
    & $Python -m bridgeengine.benchmark.head_to_head_runner run-manifest `
        --manifest "$OutputDir\command_manifest.json"
}

if (-not $SkipPi07) {
    Write-Host "Running BridgeEngine pi0.7 conditioning scale curve with matched hyperparameters."
    $env:BRIDGEENGINE_LEWM_EPOCHS = "20"
    $env:BRIDGEENGINE_LEWM_BATCH_SIZE = "16"
    $env:BRIDGEENGINE_LEWM_LR = "5e-5"
    & $Python -m bridgeengine.benchmark.scale_curve `
        --snapshot $Snapshot `
        --sizes 25 50 100 `
        --heldout-count 10 `
        --quality-stratified `
        --seed 0 `
        --benchmark-seeds 42 137 256 `
        --output-dir "$OutputDir\bridgeengine_pi07" `
        --run
}

Write-Host "Head-to-head run complete. CV aux eval JSON files are under $OutputDir\runs; pi0.7 results are under $OutputDir\bridgeengine_pi07."
