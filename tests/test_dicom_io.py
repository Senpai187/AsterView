from pathlib import Path
from hashlib import sha256

import numpy as np
import pytest
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.sequence import Sequence
from pydicom.uid import CTImageStorage, EnhancedCTImageStorage, ExplicitVRLittleEndian, generate_uid
from pydicom.uid import RLELossless
import pydicom

from asterview.dicom_io import DicomLoadError, LoadingCancelled, SeriesInfo, _Frame, _geometry, discover_series, load_series


@pytest.fixture
def dicom_factory(tmp_path):
    series, study, frame_of_reference = generate_uid(), generate_uid(), generate_uid()

    def write(name, *, z=0, pixels=None, slope=1, intercept=0, orientation=None,
              pixel_spacing=(0.7, 0.4), **attributes):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = FileMetaDataset()
        meta.TransferSyntaxUID = ExplicitVRLittleEndian
        meta.MediaStorageSOPClassUID = CTImageStorage
        meta.MediaStorageSOPInstanceUID = generate_uid()
        dataset = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
        dataset.SOPClassUID = meta.MediaStorageSOPClassUID
        dataset.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
        dataset.SeriesInstanceUID, dataset.StudyInstanceUID = series, study
        dataset.FrameOfReferenceUID = frame_of_reference
        dataset.Modality = "CT"
        dataset.SeriesDescription = "Test phantom"
        dataset.ImageType = ["ORIGINAL", "PRIMARY", "AXIAL"]
        dataset.ImageOrientationPatient = orientation or [1, 0, 0, 0, 1, 0]
        dataset.ImagePositionPatient = [11, 21, z]
        dataset.PixelSpacing = list(pixel_spacing)
        dataset.SliceThickness = 2
        dataset.SamplesPerPixel = 1
        dataset.PhotometricInterpretation = "MONOCHROME2"
        pixels = np.asarray(pixels if pixels is not None else [[1, 2], [3, 4]], dtype=np.int16)
        dataset.Rows, dataset.Columns = pixels.shape[-2:]
        dataset.BitsAllocated = dataset.BitsStored = 16
        dataset.HighBit, dataset.PixelRepresentation = 15, 1
        dataset.RescaleSlope, dataset.RescaleIntercept = slope, intercept
        for key, value in attributes.items():
            if value is None:
                if key in dataset:
                    del dataset[key]
            else:
                setattr(dataset, key, value)
        dataset.PixelData = pixels.tobytes()
        dataset.save_as(path, enforce_file_format=True)
        return path

    return write


def _info(paths):
    return SeriesInfo("test", "test", list(paths), "CT", len(paths))


def test_sort_by_patient_geometry_not_filename_or_instance_number(dicom_factory):
    last = dicom_factory("a.dcm", z=7, pixels=[[30, 31], [32, 33]], slope=3, intercept=-100,
                         InstanceNumber=1, RescaleType="HU")
    first = dicom_factory("z.dcm", z=3, pixels=[[10, 11], [12, 13]], slope=2, intercept=-1000,
                          InstanceNumber=999, RescaleType="HU")
    middle = dicom_factory("m.dcm", z=5, pixels=[[20, 21], [22, 23]], slope=1.5, intercept=-500,
                           InstanceNumber=500, RescaleType="HU")
    volume = load_series(_info([last, first, middle]))
    assert volume.spacing == (0.4, 0.7, 2)
    assert np.allclose(volume.origin, [11, 21, 3])
    assert np.array_equal(volume.data[:, 0, 0], [-980, -470, -10])
    assert volume.is_hu
    assert np.allclose(volume.index_to_world([1, 1, 2]), [11.4, 21.7, 7])


def test_cbct_exported_as_ct_without_explicit_units_is_not_hu(dicom_factory):
    path = dicom_factory("image")
    volume = load_series(_info([path]))
    assert not volume.is_hu
    assert volume.metadata["units"] == "Unspecified intensity"
    assert any("must not be assumed" in warning for warning in volume.warnings)


def test_discovery_recursive_extensionless_and_splits_temporal_runs(dicom_factory, tmp_path):
    dicom_factory("nested/00001", z=0, TemporalPositionIdentifier=1)
    dicom_factory("nested/00002", z=2, TemporalPositionIdentifier=1)
    dicom_factory("nested/00003", z=0, TemporalPositionIdentifier=2)
    (tmp_path / "ignore.txt").write_text("This is not DICOM")
    series = discover_series(tmp_path)
    assert sorted(s.count for s in series) == [1, 2]
    assert len({s.uid for s in series}) == 2


@pytest.mark.parametrize("positions,match", [
    ([0, 2, 2], "Duplicate"),
    ([0, 2, 6], "nonuniform"),
])
def test_bad_slice_positions_fail_without_silently_joining(dicom_factory, positions, match):
    paths = [dicom_factory(f"{i}.dcm", z=z) for i, z in enumerate(positions)]
    with pytest.raises(DicomLoadError, match=match):
        load_series(_info(paths))


def test_missing_3d_geometry_is_not_inferred_from_slice_thickness(dicom_factory):
    a = dicom_factory("a", ImagePositionPatient=None)
    b = dicom_factory("b", z=2)
    with pytest.raises(DicomLoadError, match="Missing geometry"):
        load_series(_info([a, b]))


def test_single_image_fallback_discloses_uncalibrated_geometry(dicom_factory):
    path = dicom_factory("single", PixelSpacing=None, ImagePositionPatient=None,
                         ImageOrientationPatient=None)
    volume = load_series(_info([path]))
    assert volume.data.shape == (1, 2, 2)
    assert volume.spacing == (1, 1, 2)
    assert not volume.metadata["geometry_calibrated"]
    assert any("distances are uncalibrated" in warning for warning in volume.warnings)


def test_monochrome1_inversion_is_display_metadata_not_voxel_negation(dicom_factory):
    path = dicom_factory("single", PhotometricInterpretation="MONOCHROME1")
    volume = load_series(_info([path]))
    assert volume.metadata["monochrome1"]
    assert np.array_equal(volume.data[0], [[1, 2], [3, 4]])


def test_oblique_stack_preserves_patient_axes(dicom_factory):
    orientation = [0, 1, 0, 0, 0, 1]
    paths = [dicom_factory(f"{i}", orientation=orientation, ImagePositionPatient=[i * 2, 5, 8])
             for i in [2, 0, 1]]
    volume = load_series(_info(paths))
    assert np.allclose(volume.direction, [[0, 0, 1], [1, 0, 0], [0, 1, 0]])
    assert np.allclose(volume.index_to_world([1, 1, 2]), [4, 5.4, 8.7])


def test_incompatible_orientation_and_gantry_tilt_rejected(dicom_factory):
    a = dicom_factory("a", z=0)
    b = dicom_factory("b", z=2, ImagePositionPatient=[11.5, 21, 2])
    with pytest.raises(DicomLoadError, match="gantry tilt"):
        load_series(_info([a, b]))
    c = dicom_factory("c", z=2, orientation=[0, 1, 0, -1, 0, 0])
    with pytest.raises(DicomLoadError, match="orientations vary"):
        load_series(_info([a, c]))


def test_allocation_budget_checked_before_decode(dicom_factory):
    path = dicom_factory("small")
    with pytest.raises(DicomLoadError, match="safety budget"):
        load_series(_info([path]), memory_budget_bytes=1)


def test_cancel_during_discovery_and_loading(dicom_factory, tmp_path):
    path = dicom_factory("one")
    with pytest.raises(LoadingCancelled):
        discover_series(tmp_path, cancel=lambda: True)
    with pytest.raises(LoadingCancelled):
        load_series(_info([path]), cancel=lambda: True)


def _enhanced(dicom_factory, *, missing_positions=False, phases=(1, 1, 1)):
    values = np.array([[[10, 11], [12, 13]], [[20, 21], [22, 23]], [[30, 31], [32, 33]]], dtype=np.int16)
    groups = []
    for i, z in enumerate([4, 0, 2]):
        group = Dataset()
        if not missing_positions:
            position = Dataset()
            position.ImagePositionPatient = [11, 21, z]
            group.PlanePositionSequence = Sequence([position])
        transform = Dataset()
        transform.RescaleSlope, transform.RescaleIntercept, transform.RescaleType = i + 1, -100 * (i + 1), "HU"
        group.PixelValueTransformationSequence = Sequence([transform])
        content = Dataset()
        content.TemporalPositionIndex = phases[i]
        content.StackID = "1"
        group.FrameContentSequence = Sequence([content])
        groups.append(group)
    shared = Dataset()
    orientation = Dataset()
    orientation.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
    shared.PlaneOrientationSequence = Sequence([orientation])
    measures = Dataset()
    measures.PixelSpacing = [0.7, 0.4]
    measures.SliceThickness = 2
    shared.PixelMeasuresSequence = Sequence([measures])
    path = dicom_factory("enhanced", pixels=values, NumberOfFrames=3,
                         SOPClassUID=EnhancedCTImageStorage,
                         ImagePositionPatient=None, ImageOrientationPatient=None, PixelSpacing=None,
                         SharedFunctionalGroupsSequence=Sequence([shared]),
                         PerFrameFunctionalGroupsSequence=Sequence(groups))
    return path


def test_enhanced_geometry_and_per_frame_rescale_follow_reordered_frames(dicom_factory):
    path = _enhanced(dicom_factory)
    volume = load_series(_info([path]))
    assert volume.spacing == (0.4, 0.7, 2)
    assert np.array_equal(volume.data[:, 0, 0], [-160, -210, -90])
    assert np.allclose(volume.origin, [11, 21, 0])
    assert volume.is_hu


def test_enhanced_missing_explicit_position_rejected(dicom_factory):
    path = _enhanced(dicom_factory, missing_positions=True)
    with pytest.raises(DicomLoadError, match="explicit patient position"):
        load_series(_info([path]))


def test_enhanced_multiple_temporal_phases_rejected(dicom_factory):
    path = _enhanced(dicom_factory, phases=(1, 2, 1))
    with pytest.raises(DicomLoadError, match="multiple stacks, phases, or echoes"):
        load_series(_info([path]))


def test_legacy_multiframe_cine_not_fabricated_into_volume(dicom_factory):
    path = dicom_factory("cine", pixels=np.ones((3, 2, 2)), NumberOfFrames=3)
    with pytest.raises(DicomLoadError, match="no explicit per-frame patient positions"):
        load_series(_info([path]))


def test_rle_compressed_pixels_decode_and_input_bytes_are_unchanged(dicom_factory):
    path = dicom_factory("compressed", pixels=[[-1000, 0], [250, 2000]], slope=0.5, intercept=-20)
    dataset = pydicom.dcmread(path)
    dataset.compress(RLELossless)
    dataset.save_as(path, enforce_file_format=True)
    before = sha256(path.read_bytes()).digest()
    volume = load_series(_info([path]))
    assert np.array_equal(volume.data[0], [[-520, -20], [105, 980]])
    assert sha256(path.read_bytes()).digest() == before


def test_manual_selection_cannot_combine_different_temporal_acquisitions(dicom_factory):
    paths = [dicom_factory("a", z=0, TemporalPositionIdentifier=1),
             dicom_factory("b", z=2, TemporalPositionIdentifier=2)]
    with pytest.raises(DicomLoadError, match="different acquisitions, phases"):
        load_series(_info(paths))


def test_incomplete_rescale_is_rejected_instead_of_inventing_intercept(dicom_factory):
    path = dicom_factory("bad-rescale", RescaleIntercept=None)
    with pytest.raises(DicomLoadError, match="rescale transform is incomplete"):
        load_series(_info([path]))


def test_real_volume_fingerprint_has_source_instance_digest(dicom_factory):
    paths = [dicom_factory("a", z=0), dicom_factory("b", z=2)]
    volume = load_series(_info(paths))
    reversed_volume = load_series(_info(paths[::-1]))
    assert volume.metadata["source_group"] == "test"
    assert len(volume.metadata["sop_instance_digest"]) == 64
    assert volume.metadata["sop_instance_digest"] == reversed_volume.metadata["sop_instance_digest"]


def test_small_adjacent_spacing_errors_cannot_accumulate_into_millimeter_grid_error():
    differences = np.r_[np.full(600, 1.004), np.full(600, 0.996)]
    positions = np.r_[0, np.cumsum(differences)]
    frames = [_Frame(Path("unused"), i, Dataset(), np.array([0., 0., z]),
                     np.array([1., 0., 0., 0., 1., 0.]), np.array([0.4, 0.4]), Dataset(), 1., ())
              for i, z in enumerate(positions)]
    with pytest.raises(DicomLoadError, match="Cumulative slice positions"):
        _geometry(frames, [])


@pytest.mark.parametrize("volumetric", ["SAMPLED", "DISTORTED", "MIXED"])
def test_nonvolume_root_properties_reject_3d_reconstruction(dicom_factory, volumetric):
    paths = [dicom_factory(f"{i}", z=2 * i, VolumetricProperties=volumetric) for i in range(2)]
    with pytest.raises(DicomLoadError, match="volumetric properties"):
        load_series(_info(paths))


def test_enhanced_per_frame_sampled_properties_rejected_even_when_positions_regular(dicom_factory):
    path = _enhanced(dicom_factory)
    dataset = pydicom.dcmread(path)
    for i, group in enumerate(dataset.PerFrameFunctionalGroupsSequence):
        frame_type = Dataset()
        frame_type.VolumetricProperties = "SAMPLED" if i == 1 else "VOLUME"
        group.CTImageFrameTypeSequence = Sequence([frame_type])
    dataset.save_as(path, enforce_file_format=True)
    with pytest.raises(DicomLoadError, match="volumetric properties"):
        load_series(_info([path]))


def test_mixed_root_requires_and_accepts_explicit_volume_per_frame(dicom_factory):
    path = _enhanced(dicom_factory)
    dataset = pydicom.dcmread(path)
    dataset.VolumetricProperties = "MIXED"
    dataset.save_as(path, enforce_file_format=True)
    with pytest.raises(DicomLoadError, match="every frame to explicitly declare VOLUME"):
        load_series(_info([path]))
    for group in dataset.PerFrameFunctionalGroupsSequence:
        frame_type = Dataset()
        frame_type.VolumetricProperties = "VOLUME"
        group.CTImageFrameTypeSequence = Sequence([frame_type])
    dataset.save_as(path, enforce_file_format=True)
    assert load_series(_info([path])).dimensions == (2, 2, 3)


@pytest.mark.parametrize("technique", ["MAX_IP", "MIN_IP", "CURVED_MPR"])
def test_projection_calculation_techniques_rejected_without_volumetric_tag(dicom_factory, technique):
    paths = [dicom_factory(f"{i}", z=2 * i, VolumeBasedCalculationTechnique=technique) for i in range(2)]
    with pytest.raises(DicomLoadError, match="calculation techniques"):
        load_series(_info(paths))


def test_projection_image_type_fallback_rejected(dicom_factory):
    paths = [dicom_factory(f"{i}", z=2 * i, ImageType=["DERIVED", "SECONDARY", "MAX_IP"]) for i in range(2)]
    with pytest.raises(DicomLoadError, match="image/frame types"):
        load_series(_info(paths))


def test_stored_padding_range_is_excluded_before_different_per_slice_rescales(dicom_factory):
    first = dicom_factory("a", z=0, pixels=[[-2000, -1999], [100, 200]], PixelPaddingValue=-2000,
                          PixelPaddingRangeLimit=-1999, slope=2, intercept=1000)
    second = dicom_factory("b", z=2, pixels=[[-2000, 300], [400, -2000]], PixelPaddingValue=-2000,
                           slope=-2, intercept=1000)
    before = [sha256(path.read_bytes()).digest() for path in [first, second]]
    volume = load_series(_info([second, first]))
    expected_mask = np.array([[[False, False], [True, True]], [[False, True], [True, False]]])
    assert np.array_equal(volume.valid_mask, expected_mask)
    assert np.array_equal(volume.data[expected_mask], [1200, 1400, 400, 200])
    assert (volume.data_min, volume.data_max) == (200, 1400)
    assert np.all(volume.data[~expected_mask] == volume.padding_value)
    assert volume.padding_value < 200
    assert volume.window[1] < 1300  # padding values cannot inflate the automatic window
    assert volume.metadata["padding_voxel_count"] == 4
    assert before == [sha256(path.read_bytes()).digest() for path in [first, second]]


def test_enhanced_padding_uses_stored_pixels_before_per_frame_transform(dicom_factory):
    path = _enhanced(dicom_factory)
    dataset = pydicom.dcmread(path)
    dataset.add_new(0x00280120, "SS", 10)
    dataset.save_as(path, enforce_file_format=True)
    volume = load_series(_info([path]))
    assert not volume.valid_mask[2, 0, 0]  # stored frame zero is reordered to last
    assert volume.data[2, 0, 0] == volume.padding_value
    assert volume.data[2, 0, 1] == -89


def test_all_padding_image_fails_explicitly(dicom_factory):
    path = dicom_factory("all-padding", pixels=[[-2000, -2000], [-2000, -2000]], PixelPaddingValue=-2000)
    with pytest.raises(DicomLoadError, match="no valid non-padding voxels"):
        load_series(_info([path]))


def test_local_patient_identity_is_in_ui_metadata_but_not_progress(dicom_factory, tmp_path):
    dicom_factory("identity", PatientID="CASE-ALPHA", PatientName="Example^Phantom", StudyDate="20261020")
    messages = []
    info = discover_series(tmp_path, progress=lambda percent, message: messages.append(message))[0]
    volume = load_series(info, progress=lambda percent, message: messages.append(message))
    assert "CASE-ALPHA" in info.label and "20261020" in info.label
    assert volume.metadata["patient_id"] == "CASE-ALPHA"
    assert volume.metadata["patient_name"] == "Example^Phantom"
    assert volume.metadata["study_date"] == "20261020"
    assert all("CASE-ALPHA" not in message and "Example" not in message for message in messages)


def test_modality_lut_units_override_unrelated_rescale_type(dicom_factory):
    lut = Dataset()
    lut.LUTDescriptor = [4, 0, 16]
    lut.ModalityLUTType = "US"
    lut.LUTData = np.array([100, 200, 300, 400], dtype="<u2").tobytes()
    path = dicom_factory("lut", pixels=[[0, 1], [2, 3]], RescaleSlope=None, RescaleIntercept=None,
                         RescaleType="HU", ModalityLUTSequence=Sequence([lut]))
    volume = load_series(_info([path]))
    assert np.array_equal(volume.data[0], [[100, 200], [300, 400]])
    assert not volume.is_hu and volume.metadata["units"] == "US"


def test_modality_lut_and_rescale_are_mutually_exclusive(dicom_factory):
    lut = Dataset()
    lut.LUTDescriptor = [4, 0, 16]
    lut.ModalityLUTType = "HU"
    lut.LUTData = np.array([100, 200, 300, 400], dtype="<u2").tobytes()
    path = dicom_factory("ambiguous-lut", ModalityLUTSequence=Sequence([lut]))
    with pytest.raises(DicomLoadError, match="LUT and a linear rescale"):
        load_series(_info([path]))
