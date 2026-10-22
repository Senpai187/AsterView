$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
Set-Location -LiteralPath $PSScriptRoot

# Keep this runtime separate from the viewer's Qt/VTK environment. Reuse the
# task's work directory when present; portable copies use project-local work.
$taskParentWork = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..\work'))
$taskStore = if (Test-Path -LiteralPath (Join-Path $taskParentWork 'venv\Scripts\python.exe')) { $taskParentWork } else { Join-Path $PSScriptRoot 'work' }
$taskAssets = Join-Path $taskStore 'segmentation'
$taskRuntime = Join-Path $taskStore 'ai-venv'
$taskPython = Join-Path $taskRuntime 'Scripts\python.exe'
New-Item -ItemType Directory -Force -Path $taskAssets | Out-Null

if (-not (Test-Path -LiteralPath $taskPython)) {
    $taskBasePython = Join-Path $taskParentWork 'venv\Scripts\python.exe'
    if (Test-Path -LiteralPath $taskBasePython) { & $taskBasePython -m venv $taskRuntime }
    else { & py -3.11 -m venv $taskRuntime }
    if ($LASTEXITCODE -ne 0) { throw 'Python 3.11 x64 is required for the AI runtime.' }
}

function Get-VerifiedAsset([string]$Url, [string]$Path, [string]$ExpectedHash) {
    if ((Test-Path -LiteralPath $Path) -and ((Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant() -eq $ExpectedHash)) { return }
    Write-Host ('Downloading official asset: ' + [IO.Path]::GetFileName($Path))
    & curl.exe --fail --location --retry 3 --output $Path $Url
    if ($LASTEXITCODE -ne 0) { throw 'An official asset download failed. Run this setup again to retry.' }
    if ((Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant() -ne $ExpectedHash) { throw 'Downloaded asset SHA256 mismatch.' }
}

Write-Host 'Installing local CUDA inference (about 2.8 GB of downloads plus dependencies).'
$taskWheel = Join-Path $taskAssets 'torch-2.6.0+cu124-cp311-cp311-win_amd64.whl'
Get-VerifiedAsset 'https://download.pytorch.org/whl/cu124/torch-2.6.0%2Bcu124-cp311-cp311-win_amd64.whl' $taskWheel '6a1fb2714e9323f11edb6e8abf7aad5f79e45ad25c081cde87681a18d99c29eb'
& $taskPython -m pip install --disable-pip-version-check $taskWheel
if ($LASTEXITCODE -ne 0) { throw 'CUDA PyTorch installation failed.' }
& $taskPython -m pip install --disable-pip-version-check -r (Join-Path $PSScriptRoot 'requirements-ai-lock.txt')
if ($LASTEXITCODE -ne 0) { throw 'AI dependency installation failed.' }

$taskArchive = Join-Path $taskAssets 'Dataset112_DentalSegmentator_v100.zip'
Get-VerifiedAsset 'https://zenodo.org/api/records/10829675/files/Dataset112_DentalSegmentator_v100.zip/content' $taskArchive 'bc5510cc93bc2100ab1faccb63512e09c1ca326c738b0a9939c074d82b38a4ac'
$taskModelRoot = Join-Path $taskAssets 'models'
New-Item -ItemType Directory -Force -Path $taskModelRoot | Out-Null
Expand-Archive -LiteralPath $taskArchive -DestinationPath $taskModelRoot -Force
& $taskPython -m pip check
if ($LASTEXITCODE -ne 0) { throw 'AI dependency verification failed.' }

$taskProbe = @'
import importlib.metadata as metadata, json, os, pathlib, sys
import torch
for name in ('nnUNet_raw', 'nnUNet_preprocessed', 'nnUNet_results'):
    os.environ.setdefault(name, str(pathlib.Path(sys.argv[1]).parent / 'runtime-probe' / name))
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
if not torch.cuda.is_available():
    raise RuntimeError('CUDA unavailable. Update the NVIDIA driver and run Setup AI.ps1 again.')
record = {'torch': torch.__version__, 'cuda': torch.version.cuda, 'device': torch.cuda.get_device_name(0), 'nnunetv2': metadata.version('nnunetv2')}
pathlib.Path(sys.argv[1]).write_text(json.dumps(record, indent=2), encoding='utf-8')
print(record)
'@
& $taskPython -c $taskProbe (Join-Path $taskAssets 'runtime-ready.json')
if ($LASTEXITCODE -ne 0) { throw 'The AI runtime failed its CUDA verification.' }
Write-Host 'Ready. Restart AsterView and open AI review. All inference runs locally; new masks require review.'
