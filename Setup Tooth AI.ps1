$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
Set-Location -LiteralPath $PSScriptRoot

$taskParentWork = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..\work'))
$taskStore = if (Test-Path -LiteralPath (Join-Path $taskParentWork 'ai-venv\Scripts\python.exe')) { $taskParentWork } else { Join-Path $PSScriptRoot 'work' }
$taskMainPython = Join-Path $taskStore 'ai-venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $taskMainPython) -or -not (Test-Path -LiteralPath (Join-Path $taskStore 'segmentation\runtime-ready.json'))) {
    & (Join-Path $PSScriptRoot 'Setup AI.ps1')
    if ($LASTEXITCODE -ne 0) { throw 'The initial dental AI runtime could not be installed.' }
    $taskStore = if (Test-Path -LiteralPath (Join-Path $taskParentWork 'ai-venv\Scripts\python.exe')) { $taskParentWork } else { Join-Path $PSScriptRoot 'work' }
    $taskMainPython = Join-Path $taskStore 'ai-venv\Scripts\python.exe'
}

$taskAssets = Join-Path $taskStore 'toothseg'
$taskRuntime = Join-Path $taskStore 'tooth-ai-venv'
$taskPython = Join-Path $taskRuntime 'Scripts\python.exe'
New-Item -ItemType Directory -Force -Path $taskAssets | Out-Null
if (-not (Test-Path -LiteralPath $taskPython)) {
    & $taskMainPython -m venv $taskRuntime
    if ($LASTEXITCODE -ne 0) { throw 'Unable to create the isolated tooth AI runtime.' }
}

# Reuse the verified CUDA libraries while overriding only the five listed
# packages in a separate environment. No user or global Python paths change.
$taskSharedSite = Join-Path $taskStore 'ai-venv\Lib\site-packages'
$taskPth = Join-Path $taskRuntime 'Lib\site-packages\asterview_ai_shared.pth'
& $taskPython -c 'import locale,pathlib,sys; pathlib.Path(sys.argv[1]).write_text(sys.argv[2] + "\n",encoding=locale.getpreferredencoding(False))' $taskPth $taskSharedSite
if ($LASTEXITCODE -ne 0) { throw 'Unable to configure the local shared CUDA runtime.' }
& $taskPython -m pip install --disable-pip-version-check --ignore-installed --no-deps -r (Join-Path $PSScriptRoot 'requirements-tooth-ai-lock.txt')
if ($LASTEXITCODE -ne 0) { throw 'Tooth AI dependency installation failed.' }

function Get-VerifiedToothAsset([string]$Url, [string]$Path, [string]$ExpectedHash) {
    if ((Test-Path -LiteralPath $Path) -and ((Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant() -eq $ExpectedHash)) { return }
    Write-Host ('Downloading official asset: ' + [IO.Path]::GetFileName($Path))
    & curl.exe --fail --location --retry 3 --output $Path $Url
    if ($LASTEXITCODE -ne 0) { throw 'An official ToothSeg download failed. Run this setup again to retry.' }
    if ((Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant() -ne $ExpectedHash) { throw 'ToothSeg asset SHA256 mismatch.' }
}

Write-Host 'Installing official ToothSeg weights (920 MB) and pinned Apache-2.0 inference code.'
$taskArchive = Join-Path $taskAssets 'ToothSeg.zip'
Get-VerifiedToothAsset 'https://zenodo.org/api/records/14893540/files/ToothSeg.zip/content' $taskArchive '803072574d12eb34c9989622a8a1508dc1998be95e9732f9718a4d0d032befd3'
$taskModels = Join-Path $taskAssets 'models'
New-Item -ItemType Directory -Force -Path $taskModels | Out-Null
Expand-Archive -LiteralPath $taskArchive -DestinationPath $taskModels -Force

$taskCodeZip = Join-Path $taskAssets 'ToothSeg-code-b29d1017.zip'
Get-VerifiedToothAsset 'https://github.com/MIC-DKFZ/ToothSeg/archive/b29d1017fa124f89645fa56f98649e3f3f43bdb0.zip' $taskCodeZip '29b64633f408e3f111904d04e9e2e411539c4446fdfd5ab57d05af0b83b62d0f'
$taskCode = Join-Path $taskAssets 'ToothSeg-code'
$taskExtract = @'
from pathlib import Path
import sys,zipfile
target=Path(sys.argv[2]).resolve()
with zipfile.ZipFile(sys.argv[1]) as archive:
    for item in archive.infolist():
        relative=Path(*Path(item.filename).parts[1:])
        if str(relative)=='.': continue
        destination=(target/relative).resolve()
        destination.relative_to(target)
        if item.is_dir(): destination.mkdir(parents=True,exist_ok=True)
        else:
            destination.parent.mkdir(parents=True,exist_ok=True)
            with archive.open(item) as source, destination.open('wb') as out:
                import shutil
                shutil.copyfileobj(source,out)
'@
& $taskPython -c $taskExtract $taskCodeZip $taskCode
if ($LASTEXITCODE -ne 0) { throw 'Pinned ToothSeg source extraction failed.' }
& $taskPython -m pip check
if ($LASTEXITCODE -ne 0) { throw 'Tooth AI dependency verification failed.' }

$taskProbe = @'
import importlib.metadata as metadata,json,os,pathlib,sys
import torch
sys.path.insert(0,sys.argv[1])
from asterview.tooth_inference import ARTIFACT_HASHES,CODE_HASHES,CODE_COMMIT
from asterview.inference import _sha256
assets=pathlib.Path(sys.argv[2])
for name in ('nnUNet_raw','nnUNet_preprocessed','nnUNet_results'):
    os.environ[name]=str(assets/'runtime-probe'/name)
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
for folder,artifacts in ((assets/'models/ToothSeg',ARTIFACT_HASHES),(assets/'ToothSeg-code',CODE_HASHES)):
    for name,expected in artifacts.items():
        if _sha256(folder/name)!=expected: raise RuntimeError('Model/source hash verification failed: '+name)
if not torch.cuda.is_available(): raise RuntimeError('CUDA is unavailable in the Tooth AI runtime.')
record={'torch':torch.__version__,'nnunetv2':metadata.version('nnunetv2'),'code_commit':CODE_COMMIT,'device':torch.cuda.get_device_name(0)}
(assets/'runtime-ready.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
print(record)
'@
& $taskPython -c $taskProbe $PSScriptRoot $taskAssets
if ($LASTEXITCODE -ne 0) { throw 'The tooth AI runtime or model failed verification.' }
Write-Host 'Ready. Restart AsterView and choose Individual teeth + FDI. All masks and suggested numbers need review.'
