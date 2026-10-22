"""One compact acquisition strip; raw DICOM values remain in its tooltip."""
from __future__ import annotations

import re
from datetime import datetime

import numpy as np


def format_time_12h(value):
    """Display only precision supplied by DICOM, without seconds or fractions."""
    raw = str(value or "").strip()
    match = re.fullmatch(r"(\d{2})(?:(\d{2})(?:\d{2}(?:\.\d+)?)?)?", raw)
    if not match:
        return ""
    hour = int(match[1])
    minute = None if match[2] is None else int(match[2])
    if hour > 23 or (minute is not None and minute > 59):
        return ""
    clock = str(hour % 12 or 12)
    if minute:
        clock += f":{minute:02d}"
    return clock + (" AM" if hour < 12 else " PM")


def _date(value):
    raw = str(value or "")
    try:
        if len(raw) != 8 or not raw.isdigit():
            return ""
        parsed = datetime.strptime(raw, "%Y%m%d")
        return f"{parsed.month}/{parsed.day}/{parsed.year % 100:02d}"
    except ValueError:
        return ""


def _timestamp(metadata):
    dt = str(metadata.get("acquisition_datetime") or "")
    match = re.fullmatch(r"(\d{8})(\d{2}(?:\d{2}(?:\d{2}(?:\.\d+)?)?)?)?([+-]\d{4})?", dt)
    if match:
        parts = [_date(match[1]), format_time_12h(match[2])]
        if match[3]:
            zone = match[3]
            parts.append(f"UTC{zone[:3]}:{zone[3:]}")
        return " ".join(p for p in parts if p)
    acquired = bool(metadata.get("acquisition_date") or metadata.get("acquisition_time"))
    prefix = "acquisition_" if acquired else "study_"
    # Never pair a time-only acquisition with an unrelated study date.
    return " ".join(p for p in (_date(metadata.get(prefix + "date")),
                               format_time_12h(metadata.get(prefix + "time"))) if p)


def acquisition_text(metadata, dimensions=None, spacing=None, direction=None):
    values = []
    stamp = _timestamp(metadata)
    if stamp:
        values.append(stamp)
    if dimensions is not None and spacing is not None:
        dims, steps = np.asarray(dimensions), np.asarray(spacing)
        if metadata.get("geometry_calibrated", True):
            extent = dims * steps / 10
            values.append(" × ".join(f"{n:.1f}" for n in extent) + " cm (" +
                          " × ".join(str(int(n)) for n in dims) + ")")
            values.append((f"{steps[0]:.3f}" if np.allclose(steps, steps[0]) else
                           " × ".join(f"{n:.3f}" for n in steps)) + " mm")
        else:
            values.append(" × ".join(str(int(n)) for n in dims) + " voxels · uncalibrated")
    elif metadata.get("source_slice_thickness_mm"):
        values.append(f"{metadata['source_slice_thickness_mm']} mm source thickness")
    exposure = []
    for key, unit in (("kvp", "kV"), ("tube_current_ma", "mA"), ("exposure_mas", "mAs")):
        if metadata.get(key):
            exposure.append(f"{metadata[key]} {unit}")
    if metadata.get("exposure_time_ms"):
        try:
            exposure.append(f"{float(metadata['exposure_time_ms']) / 1000:g} s")
        except (TypeError, ValueError):
            pass
    if exposure:
        values.append(" ".join(exposure))
    if direction is not None:
        normal = np.abs(np.asarray(direction)[:, 2])
        orientation = ("SAGITTAL", "CORONAL", "AXIAL")[int(normal.argmax())]
        values.append(orientation if normal.max() > .999 else "OBLIQUE")
    return " - ".join(values)


def acquisition_details(metadata):
    """Retain exact source timestamps/precision without cluttering each image."""
    fields = (
        ("Acquisition date", "acquisition_date"), ("Acquisition time (DICOM)", "acquisition_time"),
        ("Acquisition date/time (DICOM)", "acquisition_datetime"),
        ("Study date", "study_date"), ("Study time (DICOM)", "study_time"),
        ("Source slice thickness (mm)", "source_slice_thickness_mm"),
        ("Tube voltage (kV)", "kvp"), ("Tube current (mA)", "tube_current_ma"),
        ("Exposure (mAs)", "exposure_mas"), ("Exposure time (ms)", "exposure_time_ms"),
    )
    return "\n".join(f"{label}: {metadata[key]}" for label, key in fields if metadata.get(key))
