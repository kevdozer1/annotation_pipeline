param(
  [switch]$AllowFallback,
  [switch]$RunBenchmark,
  [string]$VlmBackend = "openai",
  [string]$VlmModel = ""
)

$ErrorActionPreference = "Stop"
$RootDir = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RootDir

function Assert-LastExit {
  param([string]$Step)
  if ($LASTEXITCODE -ne 0) {
    throw "$Step failed with exit code $LASTEXITCODE"
  }
}

if (-not (Test-Path ".venv")) {
  py -3.10 -m venv .venv
}

$Py = ".\.venv\Scripts\python.exe"
& $Py -m pip install "-e" .
Assert-LastExit "pip install"

$SnapshotJson = & $Py -m bridgeengine.ingest --source bridge_v2 --episodes 13
Assert-LastExit "ingest"
$SnapshotId = $SnapshotJson | & $Py -c "import json, sys; print(json.load(sys.stdin)['snapshot_id'])"
Assert-LastExit "snapshot id parse"
Write-Host "Snapshot: $SnapshotId"

if (-not $AllowFallback) {
  $HasOpenAI = $env:OPENAI_API_KEY -or (Test-Path ".secrets\openai_api_key.txt")
  $HasMoondream = $env:MOONDREAM_API_KEY -or (Test-Path ".secrets\moondream_api_key.txt")
  if ($VlmBackend -eq "openai" -and -not $HasOpenAI) {
    throw "OPENAI_API_KEY is not set. Set it, save .secrets\openai_api_key.txt, choose -VlmBackend moondream, or pass -AllowFallback."
  }
  if ($VlmBackend -eq "moondream" -and -not $HasMoondream) {
    throw "Moondream key is not set. Run .\scripts\set_moondream_key.ps1, set MOONDREAM_API_KEY, or pass -AllowFallback."
  }
}

$LabelArgs = @("-m", "bridgeengine.label", "--snapshot", $SnapshotId, "--vlm-backend", $VlmBackend)
if ($VlmModel) {
  $LabelArgs += @("--vlm-model", $VlmModel)
}
if ($AllowFallback) {
  $LabelArgs += "--allow-fallback"
}
& $Py @LabelArgs
Assert-LastExit "label"
& $Py -m bridgeengine.inspect_labels --snapshot $SnapshotId
Assert-LastExit "inspect labels"
& $Py -m bridgeengine.query --snapshot $SnapshotId
Assert-LastExit "query"
& $Py -m bridgeengine.export --snapshot $SnapshotId --output-path training_cuts --cut-name cut_mode_a_all_labels
Assert-LastExit "cut export"

if ($RunBenchmark) {
  $BenchArgs = @("-m", "bridgeengine.benchmark.run_grid", "--snapshot", $SnapshotId, "--output-dir", "bench_results")
  if ($AllowFallback) {
    $BenchArgs += "--allow-scaffolding-labels"
  }
  & $Py @BenchArgs
  Assert-LastExit "benchmark"
} else {
  Write-Host "Benchmark skipped. Inspect labels first, then rerun with -RunBenchmark after green-lighting live Moondream outputs."
}

Write-Host "Done. Artifacts:"
Write-Host "  bridgeengine_data/snapshots/$SnapshotId"
Write-Host "  training_cuts/cut_mode_a_all_labels"
if ($RunBenchmark) {
  Write-Host "  bench_results/bench_results.csv"
  Write-Host "  bench_results/bench_bar.png"
}
