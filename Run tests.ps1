$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$viewerPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $viewerPython)) {
    $viewerPython = Join-Path $PSScriptRoot '..\..\work\venv\Scripts\python.exe'
}
& $viewerPython -m pytest -q
exit $LASTEXITCODE
