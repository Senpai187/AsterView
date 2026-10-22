$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$taskParentWork = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..\work'))
$taskStore = if (Test-Path -LiteralPath (Join-Path $taskParentWork 'venv\Scripts\python.exe')) { $taskParentWork } else { Join-Path $PSScriptRoot 'work' }
$taskMainPython = if (Test-Path -LiteralPath (Join-Path $taskStore 'venv\Scripts\python.exe')) { Join-Path $taskStore 'venv\Scripts\python.exe' } else { Join-Path $PSScriptRoot '.venv\Scripts\python.exe' }
if (-not (Test-Path -LiteralPath $taskMainPython)) {
    & (Join-Path $PSScriptRoot 'Setup.ps1')
    if ($LASTEXITCODE -ne 0) { throw 'The main AsterView Python runtime could not be installed.' }
}
$taskRuntime = Join-Path $taskStore 'registration-venv'
$taskPython = Join-Path $taskRuntime 'Scripts\python.exe'
if (-not (Test-Path -LiteralPath $taskPython)) {
    & $taskMainPython -m venv $taskRuntime
    if ($LASTEXITCODE -ne 0) { throw 'Unable to create the isolated registration runtime.' }
}
& $taskPython -m pip install --disable-pip-version-check -r (Join-Path $PSScriptRoot 'requirements-registration-lock.txt')
if ($LASTEXITCODE -ne 0) { throw 'CPU registration dependency installation failed.' }
& $taskPython -m pip check
if ($LASTEXITCODE -ne 0) { throw 'CPU registration dependency verification failed.' }
& $taskPython -c 'import SimpleITK,numpy; assert SimpleITK.Version_VersionString()=="2.5.2"; assert numpy.__version__=="2.2.6"; print("CPU registration runtime is ready.")'
if ($LASTEXITCODE -ne 0) { throw 'CPU registration runtime verification failed.' }
Write-Host 'Registration is available locally. No AI models or GPU packages are required.'
