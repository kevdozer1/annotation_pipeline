param(
    [int]$TargetEpisodes = 1000,
    [int]$ScanSample = 5000,
    [string]$OutputRoot = "D:\bridgedata_v2_subset"
)

$ErrorActionPreference = "Stop"
$Python = ".\.venv\Scripts\python.exe"

if (-not (Test-Path $Python)) {
    throw "Missing venv Python at $Python. Run from the annotation_pipeline repo root."
}

Write-Host "This command plans and downloads more BridgeData episodes. It can spend network/disk."
Write-Host "TargetEpisodes=$TargetEpisodes ScanSample=$ScanSample OutputRoot=$OutputRoot"

& $Python -m bridgeengine.scaleout_1000 download `
    --target-episodes $TargetEpisodes `
    --scan-sample $ScanSample `
    --output-root $OutputRoot
