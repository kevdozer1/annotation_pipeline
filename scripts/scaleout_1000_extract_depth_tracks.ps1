param(
    [string]$Manifest = "D:\bridgedata_v2_subset\manifest_1000.json",
    [string]$Device = "cuda",
    [int]$GridSize = 20
)

$ErrorActionPreference = "Stop"
$Python = ".\.venv\Scripts\python.exe"
$LeWMExtract = "C:\Users\Kevin\projects\LeWM_testbed\scripts\extract_3d_signals.py"

if (-not (Test-Path $Python)) {
    throw "Missing venv Python at $Python. Run from the annotation_pipeline repo root."
}
if (-not (Test-Path $LeWMExtract)) {
    throw "Missing LeWM extractor at $LeWMExtract."
}
if (-not (Test-Path $Manifest)) {
    throw "Missing manifest at $Manifest. Run scripts\scaleout_1000_download.ps1 first."
}

Write-Host "Extracting Video-Depth-Anything depth and CoTracker3 tracks for manifest $Manifest"
& $Python $LeWMExtract --manifest $Manifest --device $Device --grid-size $GridSize
