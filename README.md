# AsterView

AsterView is an early-stage local Windows workstation for CT and dental CBCT
software development. It includes DICOM volume loading, linked multiplanar views,
panoramic reformats, GPU volume rendering, comparison tools, and editable draft
report workflows. Source volume processing runs locally.

This development prototype has not been validated for diagnosis, measurements
used in patient care, or surgical planning. Automated diagnostic interpretation
is not enabled. Review generated masks, measurements, and draft text independently.

## Install and run

Use Windows x64, Python 3.11, and a graphics driver compatible with VTK rendering.
Open PowerShell in the repository directory:

```powershell
.\Setup.ps1
& '.\Start AsterView.cmd'
```

Alternatively, after setup:

```powershell
.\.venv\Scripts\python.exe -m asterview
```

The application starts with a synthetic demonstration. Open a local DICOM folder
from the application, or add `--empty` to start without the demonstration.

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Some checks require a functioning Qt display or GPU environment. Test success
checks software behavior and does not establish clinical validity.

## Optional local integrations

`Setup AI.ps1`, `Setup Tooth AI.ps1`, and `Setup Registration.ps1` prepare separate
local runtimes for optional workflows. They download dependencies or model assets;
inspect the scripts before running them. DentalSegmentator and ToothSeg are
independent upstream projects. Their models and third-party code are not included
here and retain their respective licenses and attribution requirements.

- DentalSegmentator model: https://doi.org/10.5281/zenodo.10829675
- ToothSeg code: https://github.com/MIC-DKFZ/ToothSeg
- ToothSeg models: https://doi.org/10.5281/zenodo.14893540

## Public source release

This initial public source release contains application code, synthetic tests,
setup scripts, and a new generic report template. It excludes patient data,
scan screenshots, local validation artifacts, proprietary vendor software, model
weights, and institutional report branding. No public adoption or external
validation is claimed.

AsterView code is distributed under the MIT License in `LICENSE`. Dependency and
model licenses remain separate from the AsterView code license.
