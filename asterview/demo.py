"""Deterministic synthetic dental phantom; contains no patient information."""
from __future__ import annotations

import numpy as np

from .model import Volume


def make_demo() -> Volume:
    """A 192 x 192 x 160 synthetic phantom, never a clinical example."""
    nz, ny, nx = 160, 192, 192
    spacing = (0.6, 0.6, 0.6)
    z, y, x = np.ogrid[:nz, :ny, :nx]
    x = (x - (nx - 1) / 2) * spacing[0]
    y = (y - (ny - 1) / 2) * spacing[1]
    z = (z - (nz - 1) / 2) * spacing[2]
    data = np.full((nz, ny, nx), -1000, dtype=np.float32)
    head = (x / 48) ** 2 + (y / 46) ** 2 + (z / 43) ** 2
    data[head < 1] = 35
    shell = (head > 0.88) & (head < 1)
    data[shell] = 800
    # Open U-shaped bony arch with paired condyles, suggestive only.
    arch = np.sqrt((x / 1.03) ** 2 + (y - 4) ** 2)
    jaw = (arch > 30) & (arch < 38) & (y < 22) & (z > -28) & (z < -12)
    data[jaw] = 650
    cortex = jaw & ((arch < 31.7) | (arch > 36.3) | (z < -26.5))
    data[cortex] = 1450
    upper = (arch > 29) & (arch < 37) & (y < 19) & (z > 8) & (z < 18)
    data[upper] = 850
    for angle in np.linspace(np.pi * 0.10, np.pi * 0.90, 12):
        cx, cy = 33 * np.cos(angle), 4 - 33 * np.sin(angle)
        crown = ((x - cx) / 3.1) ** 2 + ((y - cy) / 3.5) ** 2 + ((z + 8) / 5.2) ** 2
        root = ((x - cx) / 1.8) ** 2 + ((y - cy) / 2.0) ** 2 + ((z + 17) / 9) ** 2
        data[crown < 1] = 2400
        data[root < 1] = 1300
        upper_crown = ((x - cx) / 3.1) ** 2 + ((y - cy) / 3.5) ** 2 + ((z - 3) / 4.6) ** 2
        data[upper_crown < 1] = 2250
    for cx in (-35, 35):
        condyle = ((x - cx) / 8) ** 2 + ((y - 25) / 6) ** 2 + ((z + 7) / 9) ** 2
        data[condyle < 1] = 1050
    return Volume(
        data=data,
        spacing=spacing,
        origin=-np.array([(nx - 1), (ny - 1), (nz - 1)]) * 0.3,
        direction=np.eye(3),
        metadata={"modality": "SYNTHETIC", "series_description": "Synthetic dental phantom", "synthetic": True,
                  "units": "Synthetic intensity", "photometric_interpretation": "MONOCHROME2"},
        window=(600, 2800),
        is_hu=False,
        warnings=["SYNTHETIC PHANTOM — not patient data and not a diagnostic reference."],
    )
