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

& $Py -m bridgeengine.label --snapshot $SnapshotId
Assert-LastExit "label"
& $Py -m bridgeengine.query --snapshot $SnapshotId
Assert-LastExit "query"
& $Py -m bridgeengine.export --snapshot $SnapshotId --output-path training_cuts --cut-name cut_mode_a_all_labels
Assert-LastExit "cut export"
& $Py -m bridgeengine.benchmark.run_grid --snapshot $SnapshotId --output-dir bench_results
Assert-LastExit "benchmark"

Write-Host "Done. Artifacts:"
Write-Host "  bridgeengine_data/snapshots/$SnapshotId"
Write-Host "  training_cuts/cut_mode_a_all_labels"
Write-Host "  bench_results/bench_results.csv"
Write-Host "  bench_results/bench_bar.png"
