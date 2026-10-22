"""Isolated ML worker; invoked only by inference.run_segmentation.

Do not import this module into the Qt viewer. Only the pinned, hash-verified
official DentalSegmentator checkpoint is eligible for pickle-based loading.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import sys
import time

import numpy as np


def progress(message):
    print("ASTER_PROGRESS:" + message, flush=True)


def source_image(array, geometry):
    import SimpleITK as sitk
    image = sitk.GetImageFromArray(array)
    image.SetSpacing(geometry["spacing"])
    image.SetOrigin(geometry["origin"])
    image.SetDirection(np.asarray(geometry["direction"], dtype=float).reshape(-1).tolist())
    return image


def canonical_image(image):
    """Resample into increasing patient L/P/S axes for anatomical consistency.

    Axis-aligned input takes the lossless permutation/flip path. Oblique input
    uses linear scalar interpolation; its labels are later restored by nearest
    neighbor using the exact original image as reference.
    """
    import SimpleITK as sitk
    oriented = sitk.DICOMOrient(image, "LPS")
    direction = np.asarray(oriented.GetDirection()).reshape(3, 3)
    if np.allclose(direction, np.eye(3), atol=1e-6, rtol=0):
        return oriented, "axis-permutation-and-flip"
    spacing = np.asarray(oriented.GetSpacing())
    points = [oriented.TransformIndexToPhysicalPoint(list(p)) for p in itertools.product(*[(0, n - 1) for n in oriented.GetSize()])]
    low, high = np.min(points, axis=0), np.max(points, axis=0)
    # Effective sampling interval along each physical axis, preserving CT
    # anisotropy without imposing an arbitrary isotropic upsampling grid.
    target_spacing = 1 / np.sqrt(np.sum(direction ** 2 / spacing[None, :] ** 2, axis=1))
    size = np.ceil((high - low) / target_spacing).astype(int) + 1
    if int(np.prod(size, dtype=np.int64)) > 400_000_000:
        raise ValueError("The canonical AI grid exceeds 400 million voxels.")
    result = sitk.Resample(oriented, size.tolist(), sitk.Transform(), sitk.sitkLinear,
                           low.tolist(), target_spacing.tolist(), np.eye(3).reshape(-1).tolist(),
                           float(sitk.GetArrayViewFromImage(oriented).min()), sitk.sitkFloat32)
    return result, "linear-canonical-LPS"


def restore_labels(labels, canonical, source):
    import SimpleITK as sitk
    image = sitk.GetImageFromArray(np.asarray(labels, dtype=np.uint16))
    image.CopyInformation(canonical)
    restored = sitk.Resample(image, source, sitk.Transform(), sitk.sitkNearestNeighbor, 0, sitk.sitkUInt16)
    return np.asarray(sitk.GetArrayFromImage(restored), dtype=np.uint16)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", required=True)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    job, model = Path(args.job), Path(args.model)
    # Import only the small provider module to use one authoritative hash list.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from asterview.inference import ARTIFACT_HASHES
    for relative, expected in ARTIFACT_HASHES.items():
        digest = hashlib.sha256()
        with (model / relative).open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != expected:
            raise ValueError("Official model integrity verification failed: " + relative)
    # nnU-Net v2.2.1 checkpoints include architecture metadata in Python pickle.
    # Opt out of PyTorch >=2.6's changed default only after pin verification,
    # in this disposable worker; never applies to arbitrary imported weights.
    os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    import torch
    torch.set_num_threads(8)
    torch.set_num_interop_threads(4)
    torch.backends.cudnn.benchmark = True
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in the local AI runtime. Install the supported CUDA PyTorch wheel.")
    import SimpleITK as sitk
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(8)
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    started = time.monotonic()
    progress("Preparing patient-space CT/CBCT volume…")
    geometry = json.loads((job / "geometry.json").read_text(encoding="utf-8"))
    array = np.load(job / "input.npy", allow_pickle=False)
    if list(array.shape) != geometry["shape"] or array.ndim != 3 or array.dtype != np.float32 or not np.isfinite(array).all():
        raise ValueError("Invalid source array.")
    source = source_image(array, geometry)
    del array
    canonical, method = canonical_image(source)
    image = sitk.GetArrayFromImage(canonical).astype(np.float32, copy=False)[None]
    progress("Loading DentalSegmentator on " + torch.cuda.get_device_name(0) + "…")
    predictor = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=False,
                               perform_everything_on_gpu=True, device=torch.device("cuda", 0),
                               verbose=False, verbose_preprocessing=False, allow_tqdm=False)
    predictor.initialize_from_trained_model_folder(str(model), use_folds=(0,), checkpoint_name="checkpoint_final.pth")
    progress("Segmenting jaws, teeth and mandibular canals locally…")
    labels = predictor.predict_single_npy_array(image, {"spacing": canonical.GetSpacing()[::-1]}, None, None, False)
    progress("Restoring editable masks to the exact source grid…")
    output = restore_labels(labels, canonical, source)
    if list(output.shape) != geometry["shape"] or np.any(output > 5):
        raise ValueError("Invalid model label output.")
    np.save(job / "output.npy", output, allow_pickle=False)
    manifest = {"model": "DentalSegmentator v1", "grid_preparation": method, "source": geometry,
                "canonical_shape_zyx": list(image.shape[1:]), "canonical_spacing_xyz": list(canonical.GetSpacing()),
                "canonical_origin_lps_mm": list(canonical.GetOrigin()),
                "label_restoration": "SimpleITK nearest-neighbor resampling to exact source geometry",
                "device": torch.cuda.get_device_name(0), "torch": torch.__version__,
                "elapsed_seconds": time.monotonic() - started,
                "label_voxel_counts": {str(i): int(np.count_nonzero(output == i)) for i in range(6)}}
    (job / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    progress("Segmentation complete; human review required.")


if __name__ == "__main__":
    main()
