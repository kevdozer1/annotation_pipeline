param(
    [string]$DestinationRoot = "D:\bridgeengine_runtime\annotation_pipeline",
    [switch]$Apply,
    [switch]$IncludeVenv,
    [switch]$IncludeLewmTestbed
)

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
$DestinationRoot = [System.IO.Path]::GetFullPath($DestinationRoot)

if (-not $DestinationRoot.StartsWith("D:\", [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "DestinationRoot must be on D: for this migration. Got: $DestinationRoot"
}

function New-MigrationItem {
    param(
        [string]$Source,
        [string]$Destination
    )
    [PSCustomObject]@{
        Source = [System.IO.Path]::GetFullPath($Source)
        Destination = [System.IO.Path]::GetFullPath($Destination)
    }
}

$items = @(
    New-MigrationItem -Source (Join-Path $RepoRoot "bridgeengine_data") -Destination (Join-Path $DestinationRoot "bridgeengine_data")
    New-MigrationItem -Source (Join-Path $RepoRoot "training_cuts") -Destination (Join-Path $DestinationRoot "training_cuts")
)

if ($IncludeVenv) {
    $items += New-MigrationItem -Source (Join-Path $RepoRoot ".venv") -Destination (Join-Path $DestinationRoot ".venv")
}

if ($IncludeLewmTestbed) {
    $LewmRoot = "C:\Users\Kevin\projects\LeWM_testbed"
    $LewmDest = "D:\bridgeengine_runtime\LeWM_testbed"
    $items += New-MigrationItem -Source (Join-Path $LewmRoot ".venv") -Destination (Join-Path $LewmDest ".venv")
    $items += New-MigrationItem -Source (Join-Path $LewmRoot "outputs") -Destination (Join-Path $LewmDest "outputs")
}

function Get-DirectorySizeGB {
    param([string]$Path)
    if (-not (Test-Path -LiteralPath $Path)) {
        return 0.0
    }
    $sum = (Get-ChildItem -LiteralPath $Path -Recurse -Force -File -ErrorAction SilentlyContinue | Measure-Object Length -Sum).Sum
    return [math]::Round(($sum / 1GB), 3)
}

function Assert-SafeSource {
    param([string]$Path)
    $resolved = [System.IO.Path]::GetFullPath($Path)
    $allowedRoots = @(
        [System.IO.Path]::GetFullPath($RepoRoot),
        [System.IO.Path]::GetFullPath("C:\Users\Kevin\projects\LeWM_testbed")
    )
    foreach ($root in $allowedRoots) {
        if ($resolved.StartsWith($root, [System.StringComparison]::OrdinalIgnoreCase)) {
            return
        }
    }
    throw "Refusing to migrate unexpected source path: $resolved"
}

function Move-DirectoryToJunction {
    param(
        [string]$Source,
        [string]$Destination
    )

    New-Item -ItemType Directory -Path $Destination -Force | Out-Null
    $robocopyArgs = @(
        $Source,
        $Destination,
        "/E",
        "/MOVE",
        "/NFL",
        "/NDL",
        "/NJH",
        "/NJS",
        "/NP"
    )
    & robocopy @robocopyArgs | Out-Null
    $exitCode = $LASTEXITCODE
    if ($exitCode -ge 8) {
        throw "robocopy failed while moving $Source to $Destination with exit code $exitCode"
    }

    if (Test-Path -LiteralPath $Source) {
        $remainingFiles = (Get-ChildItem -LiteralPath $Source -Recurse -Force -File -ErrorAction SilentlyContinue | Measure-Object).Count
        if ($remainingFiles -gt 0) {
            throw "Source still contains $remainingFiles files after move: $Source"
        }
        Remove-Item -LiteralPath $Source -Recurse -Force
    }
    New-Item -ItemType Junction -Path $Source -Target $Destination | Out-Null
}

Write-Host "BridgeEngine runtime migration"
Write-Host "RepoRoot: $RepoRoot"
Write-Host "DestinationRoot: $DestinationRoot"
Write-Host "Mode: $(if ($Apply) { 'APPLY' } else { 'DRY RUN' })"
Write-Host ""

foreach ($item in $items) {
    Assert-SafeSource $item.Source
    $source = $item.Source
    $dest = $item.Destination
    $sourceExists = Test-Path -LiteralPath $source
    $sourceInfo = if ($sourceExists) { Get-Item -LiteralPath $source -Force } else { $null }
    $sizeGB = if ($sourceExists) { Get-DirectorySizeGB $source } else { 0.0 }

    if ($sourceExists -and $sourceInfo.LinkType) {
        Write-Host "[already linked] $source -> $($sourceInfo.Target) ($sizeGB GB visible)"
        continue
    }
    if (-not $sourceExists) {
        if (Test-Path -LiteralPath $dest) {
            Write-Host "[restore junction] $source -> $dest"
            if ($Apply) {
                New-Item -ItemType Junction -Path $source -Target $dest | Out-Null
            }
            continue
        }
        Write-Host "[missing] $source"
        continue
    }
    if (Test-Path -LiteralPath $dest) {
        $destSize = Get-DirectorySizeGB $dest
        Write-Host "[resume move] $source ($sizeGB GB remaining) -> $dest ($destSize GB already on D:)"
        Write-Host "[junction] $source -> $dest"
        if ($Apply) {
            Move-DirectoryToJunction -Source $source -Destination $dest
        }
        continue
    }

    Write-Host "[move] $source ($sizeGB GB) -> $dest"
    Write-Host "[junction] $source -> $dest"
    if (-not $Apply) {
        continue
    }

    Move-DirectoryToJunction -Source $source -Destination $dest
}

Write-Host ""
Write-Host "Done. Validate with:"
Write-Host "  Get-Item bridgeengine_data,training_cuts,.venv -Force | Select Name,LinkType,Target"
Write-Host "  .\.venv\Scripts\python.exe -m pytest tests\test_head_to_head.py -q"
