param(
    [string]$SourceRoot = "D:\bridgedata_v2_subset",
    [int]$Episodes = 1000,
    [string]$Backend = "gemini",
    [string]$Model = "gemini-2.5-flash"
)

$ErrorActionPreference = "Stop"
$Python = ".\.venv\Scripts\python.exe"

if (-not (Test-Path $Python)) {
    throw "Missing venv Python at $Python. Run from the annotation_pipeline repo root."
}

Write-Host "Ingesting $Episodes episodes from $SourceRoot..."
$ingest = & $Python -m bridgeengine.ingest --source $SourceRoot --episodes $Episodes | Out-String
Write-Host $ingest

$snapshot = ($ingest | Select-String -Pattern '"snapshot_id":\s*"([^"]+)"').Matches.Groups[1].Value
if (-not $snapshot) {
    throw "Could not parse snapshot_id from ingest output."
}

Write-Host "Labeling snapshot $snapshot with $Backend / $Model..."
& $Python -m bridgeengine.label --snapshot $snapshot --vlm-backend $Backend --vlm-model $Model

Write-Host "Running quality and cost reports..."
& $Python -m bridgeengine.quality_report --snapshot $snapshot
& $Python -m bridgeengine.cost_probe --snapshot $snapshot --projection 1000
