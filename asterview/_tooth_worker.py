"""Isolated ToothSeg worker using two verified independent learned branches.

The original Apache-2.0 ToothSeg postprocessor is imported from a pinned local
dependency checkout; it is not Romexis code. Instance IDs are retained before
the upstream final class-map export would combine repeated FDI suggestions.
"""
from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

import numpy as np


def progress(text):
    print("ASTER_PROGRESS:" + text, flush=True)


def toothseg_model_image(source):
    """Map true patient geometry to this released model's R/P/S array frame.

    ToothFairy2's published direction metadata is not physically accurate.
    Its learned left/right class layout and shipped pair distributions have
    the opposite x handedness from the in-house ToothSeg models. This is an
    explicit image-space transform, never a post-hoc FDI-number permutation.
    """
    import SimpleITK as sitk
    from asterview._inference_worker import canonical_image
    physical_lps, method = canonical_image(source)
    return sitk.DICOMOrient(physical_lps, "RPS"), method


def prior_coordinates(ras_centroids):
    """Convert true RAS offsets to the shipped ToothFairy2 LAS prior frame."""
    result = np.asarray(ras_centroids, dtype=np.float64).copy()
    result[:, 0] *= -1
    return result


def model_class_to_fdi(value):
    """The unchanged 32-class contract from the verified model dataset.json."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not 1 <= int(value) <= 32:
        raise ValueError("ToothSeg requires a model class from 1 through 32.")
    value = int(value)
    return 10 * ((value - 1) // 8 + 1) + (value - 1) % 8 + 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", required=True)
    parser.add_argument("--models", required=True)
    parser.add_argument("--code", required=True)
    args = parser.parse_args()
    job, models, code = Path(args.job), Path(args.models), Path(args.code)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from asterview.tooth_inference import ARTIFACT_HASHES, CODE_HASHES, SEMANTIC_SUBDIR, INSTANCE_SUBDIR, CODE_COMMIT
    from asterview.inference import _sha256
    from asterview._inference_worker import source_image, restore_labels
    for root, artifacts in ((models, ARTIFACT_HASHES), (code, CODE_HASHES)):
        for name, digest in artifacts.items():
            if _sha256(root / name) != digest:
                raise ValueError("ToothSeg integrity verification failed: " + name)
    os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    import torch
    import SimpleITK as sitk
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    from acvl_utils.instance_segmentation.instance_as_semantic_seg import convert_semantic_to_instanceseg, postprocess_instance_segmentation
    from scipy.stats import multivariate_normal
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in the local ToothSeg runtime.")
    torch.set_num_threads(8)
    torch.set_num_interop_threads(4)
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(8)
    started = time.monotonic()
    geometry = json.loads((job / "geometry.json").read_text(encoding="utf-8"))
    data = np.load(job / "input.npy", allow_pickle=False)
    if data.dtype != np.float32 or list(data.shape) != geometry["shape"] or not np.isfinite(data).all():
        raise ValueError("Invalid tooth input volume.")
    source = source_image(data, geometry)
    del data
    canonical, canonical_method = toothseg_model_image(source)
    model_voxels = np.prod(np.ceil(np.array(canonical.GetSize()) * canonical.GetSpacing() / .3))
    if model_voxels > 50_000_000:
        raise ValueError("Canonical tooth ROI exceeds bounded model memory.")

    def predictor_for(subdir):
        predictor = nnUNetPredictor(tile_step_size=.5, use_gaussian=True, use_mirroring=False,
                                   perform_everything_on_device=False, device=torch.device("cuda", 0),
                                   verbose=False, verbose_preprocessing=False, allow_tqdm=False)
        predictor.initialize_from_trained_model_folder(str(models / subdir), use_folds=(5,), checkpoint_name="checkpoint_final.pth")
        return predictor

    progress("Predicting permanent tooth classes with the ToothSeg semantic network…")
    predictor = predictor_for(SEMANTIC_SUBDIR)
    _, probabilities = predictor.predict_single_npy_array(sitk.GetArrayFromImage(canonical).astype(np.float32)[None],
                                                          {"spacing": canonical.GetSpacing()[::-1]}, None, None, True)
    semantic = job / "semantic"
    semantic.mkdir(exist_ok=True)
    # Truncation is explicitly recommended in upstream ToothSeg documentation
    # for compact probability interchange. It is not a confidence calibration.
    probabilities[probabilities < 1e-6] = 0
    np.savez_compressed(semantic / "case.npz", probabilities=probabilities)
    del probabilities, predictor
    gc.collect()
    torch.cuda.empty_cache()

    progress("Predicting tooth cores and boundaries with the instance network…")
    target_spacing = np.array([.2, .2, .2])
    target_size = np.ceil((np.array(canonical.GetSize()) - 1) * canonical.GetSpacing() / target_spacing).astype(int) + 1
    if int(np.prod(target_size)) > 180_000_000:
        raise ValueError("The high-resolution tooth instance region exceeds bounded model memory.")
    highres = sitk.Resample(canonical, target_size.tolist(), sitk.Transform(), sitk.sitkLinear, canonical.GetOrigin(),
                           target_spacing.tolist(), canonical.GetDirection(), float(sitk.GetArrayViewFromImage(canonical).min()), sitk.sitkFloat32)
    predictor = predictor_for(INSTANCE_SUBDIR)
    border_core = predictor.predict_single_npy_array(sitk.GetArrayFromImage(highres).astype(np.float32, copy=False)[None],
                                                    {"spacing": highres.GetSpacing()[::-1]}, None, None, False).astype(np.uint8)
    del predictor
    gc.collect()
    torch.cuda.empty_cache()
    progress("Separating learned tooth instances and checking their identities…")
    instances = convert_semantic_to_instanceseg(border_core, spacing=highres.GetSpacing()[::-1],
                                                small_center_threshold=16., isolated_border_as_separate_instance_threshold=0.)
    del border_core
    ids, counts = np.unique(instances, return_counts=True)
    for identifier, count in zip(ids, counts):
        if identifier and count * .2 ** 3 < 16.:
            instances[instances == identifier] = 0
    instances = postprocess_instance_segmentation(instances)
    image = sitk.GetImageFromArray(instances.astype(np.uint16))
    image.CopyInformation(highres)
    del instances, highres
    image = sitk.Resample(image, canonical, sitk.Transform(), sitk.sitkNearestNeighbor, 0, sitk.sitkUInt16)
    instance_dir, assigned_dir = job / "instances", job / "assigned"
    instance_dir.mkdir(exist_ok=True)
    assigned_dir.mkdir(exist_ok=True)
    instance_file = instance_dir / "case.nii.gz"
    sitk.WriteImage(image, str(instance_file))
    del image
    postprocessor = code / "toothseg/toothseg/postprocess_predictions/assign_mincost_tooth_labels.py"
    spec = importlib.util.spec_from_file_location("aster_toothseg_postprocessor", postprocessor)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    centroids, probabilities, instance_xyz = module.prepare_instances(semantic, instance_file, assigned_dir, True)
    if not len(centroids):
        raise RuntimeError("The model found no tooth instances in this region.")
    keep = probabilities[:, 0] < .95
    active_ids = np.flatnonzero(keep) + 1
    centroids, probabilities = centroids[keep], probabilities[keep]
    if not len(active_ids):
        raise RuntimeError("The semantic network rejected every tooth instance as background.")
    pair_path = code / "toothseg/datasets/toothfairy2/fdi_pair_distrs.json"
    pair = json.loads(pair_path.read_text(encoding="utf-8"))
    normals = [[None if i // 16 != j // 16 else multivariate_normal(mean=pair["means"][i][j][:2], cov=np.array(pair["covs"][i][j])[:2, :2])
                for j in range(32)] for i in range(32)]
    lower = probabilities[:, 17:].sum(-1) > probabilities[:, 1:17].sum(-1)
    # The official helper returns RAS-oriented physical centroids. The released
    # ToothFairy2 distributions encode x-positive-left offsets (LAS), unlike
    # the in-house distributions. Convert coordinates before applying them.
    centroids_for_prior = prior_coordinates(centroids)
    learned_classes = np.zeros(len(active_ids), dtype=int)
    for is_lower in (False, True):
        arch_ids = np.flatnonzero(lower == is_lower)
        if not len(arch_ids):
            continue
        arch_prob = probabilities[arch_ids, 17:] if is_lower else probabilities[arch_ids, 1:17]
        arch_prob = np.maximum(arch_prob, 1e-8)
        arch_prob /= arch_prob.sum(axis=1, keepdims=True)
        sequence, _ = module.determine_sequence(centroids_for_prior[arch_ids])
        transitions = module.determine_transition_probabilities(normals, centroids_for_prior[arch_ids], is_lower, sequence)
        path, _ = module.dynamic_programming(arch_prob, sequence, transitions)
        learned_classes[arch_ids[sequence]] = path + 16 * is_lower + 1
    output_xyz = np.zeros_like(instance_xyz, dtype=np.uint16)
    records = []
    for index, (instance_id, predicted_class) in enumerate(zip(active_ids, learned_classes)):
        identifier = 1001 + index
        if not 1 <= predicted_class <= 32 or index >= 512:
            raise RuntimeError("Invalid ToothSeg class or instance count.")
        fdi = model_class_to_fdi(predicted_class)
        output_xyz[instance_xyz == instance_id] = identifier
        records.append({"id": identifier, "predicted_fdi": fdi})
    del probabilities, instance_xyz
    progress("Restoring individual teeth to the original patient-space region…")
    restored = restore_labels(output_xyz.transpose(2, 1, 0), canonical, source)
    records = [item for item in records if np.any(restored == item["id"])]
    if not records:
        raise RuntimeError("No instances survive source-grid restoration.")
    np.save(job / "output.npy", restored, allow_pickle=False)
    (job / "instances.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    manifest = {"model": "ToothSeg", "code_commit": CODE_COMMIT, "source_region": geometry,
                "canonical_method": canonical_method, "canonical_shape_zyx": list(canonical.GetSize()[::-1]),
                "model_array_orientation": "RPS", "model_direction_in_LPS": list(canonical.GetDirection()),
                "spatial_prior_orientation": "LAS; explicit x reflection from official helper RAS centroids",
                "instance_spacing_xyz": [.2, .2, .2], "semantic_fold": 5, "instance_fold": 5, "test_time_mirroring": False,
                "postprocessing": "Official learned border/core separation and semantic/spatial minimum-cost labeling; physical instance IDs retained",
                "source_grid_restoration": "Nearest-neighbor to exact input region; provider inserts into original full source grid",
                "instances": records, "device": torch.cuda.get_device_name(0), "elapsed_seconds": time.monotonic() - started,
                "counts": {str(item["id"]): int(np.count_nonzero(restored == item["id"])) for item in records}}
    (job / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    progress("Individual tooth masks and FDI suggestions are ready for review.")


if __name__ == "__main__":
    main()
