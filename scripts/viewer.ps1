$ErrorActionPreference = "Stop"
$RootDir = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RootDir

$Py = ".\.venv\Scripts\python.exe"
if (-not (Test-Path $Py)) {
  py -3.10 -m venv .venv
}

& $Py -m pip install -e .
if ($LASTEXITCODE -ne 0) { throw "pip install failed with exit code $LASTEXITCODE" }

& $Py -m streamlit run bridgeengine/viewer/app.py

