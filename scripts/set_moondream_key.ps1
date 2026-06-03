param()

$ErrorActionPreference = "Stop"
$RootDir = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RootDir

$SecretDir = ".secrets"
$SecretPath = Join-Path $SecretDir "moondream_api_key.txt"
New-Item -ItemType Directory -Force -Path $SecretDir | Out-Null

$Secure = Read-Host "Moondream API key" -AsSecureString
$Ptr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($Secure)
try {
  $Plain = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($Ptr)
  if (-not $Plain) {
    throw "No key entered."
  }
  $Utf8NoBom = New-Object System.Text.UTF8Encoding($false)
  [System.IO.File]::WriteAllText((Resolve-Path $SecretDir).Path + "\moondream_api_key.txt", $Plain, $Utf8NoBom)
} finally {
  [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($Ptr)
}

Write-Host "Saved Moondream key to ignored local file: $SecretPath"
