param(
    [string]$Snapshot = "snap_2026_05_11_1dde3edf5d_human_gold_labels",
    [string]$OutputDir = "head_to_head_results\run_100",
    [switch]$PrepareOnly,
    [switch]$SkipCvAux,
    [switch]$SkipPi07
)

# Fail fast: stop on the first cmdlet error, and explicitly check the native
# exit code after every Python stage so a failed stage aborts instead of
# cascading into red-herring downstream errors.
$ErrorActionPreference = "Stop"

# Pin the interpreter for EVERY stage. Do not rely on PATH. This single venv has
# torch+CUDA, h5py, stable_pretraining, stable_worldmodel, lightning, and cv2;
# the LeWM aux trainer and the fixed-split evaluator both add LeWM_testbed/src to
# sys.path themselves, so lewm_testbed does not need to be pip-installed here.
$RepoRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"
$LewmRoot = "C:\Users\Kevin\projects\LeWM_testbed"
$EpisodesRoot = "D:\bridgedata_v2_subset\episodes"
$Manifest = "D:\bridgedata_v2_subset\manifest_100.json"
$SourceH5 = "D:\bridgedata_v2_subset\datasets\bridgedata_v2_100ep.h5"
$PlanDir = "head_to_head_results\preregistered_100"

if (-not (Test-Path $Python)) {
    throw "Missing venv Python at $Python. Run from the annotation_pipeline repo root."
}

function Invoke-Stage {
    param([string]$Name, [scriptblock]$Action)
    Write-Host ""
    Write-Host "==> $Name" -ForegroundColor Cyan
    & $Action
    if ($LASTEXITCODE -ne 0) {
        throw "Stage '$Name' failed with exit code $LASTEXITCODE. Aborting head-to-head run."
    }
}

Write-Host "Interpreter: $Python"
& $Python -c "import sys, torch; print('python', sys.version.split()[0]); print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
if ($LASTEXITCODE -ne 0) {
    throw "Interpreter sanity check failed. Verify the venv has torch+CUDA."
}

Invoke-Stage "Verify LeWM depth/track signal files" {
    & $Python -m bridgeengine.benchmark.head_to_head_runner verify-signals `
        --manifest $Manifest `
        --local-root D:\bridgedata_v2_subset `
        --plan-dir $PlanDir
}

Invoke-Stage "Prepare fixed-split HDF5 datasets and LeWM configs" {
    & $Python -m bridgeengine.benchmark.head_to_head_runner prepare `
        --plan-dir $PlanDir `
        --episodes-root $EpisodesRoot `
        --source-h5 $SourceH5 `
        --manifest $Manifest `
        --output-dir $OutputDir `
        --lewm-root $LewmRoot `
        --sizes 25 50 100 `
        --seeds 42 137 256 `
        --max-epochs 20 `
        --batch-size 16 `
        --lr 5e-5
}

if ($PrepareOnly) {
    Write-Host ""
    Write-Host "PrepareOnly set. Split HDF5s and configs are written. Stopping before long training." -ForegroundColor Green
    exit 0
}

if (-not $SkipCvAux) {
    Invoke-Stage "Native LeWM CV aux-head cells (long part)" {
        & $Python -m bridgeengine.benchmark.head_to_head_runner run-manifest `
            --manifest "$OutputDir\command_manifest.json"
    }
}

if (-not $SkipPi07) {
    $env:BRIDGEENGINE_LEWM_EPOCHS = "20"
    $env:BRIDGEENGINE_LEWM_BATCH_SIZE = "16"
    $env:BRIDGEENGINE_LEWM_LR = "5e-5"
    Invoke-Stage "BridgeEngine pi0.7 conditioning scale curve (matched hyperparameters)" {
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
}

Write-Host ""
Write-Host "Head-to-head run complete. CV aux eval JSON files are under $OutputDir\runs; pi0.7 results are under $OutputDir\bridgeengine_pi07." -ForegroundColor Green
