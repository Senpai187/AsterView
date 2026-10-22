"""Patient-coordinate geometry for direct manipulation of linked MPR sections.

All points and distances use DICOM LPS millimeters. A section's display frame
depends on that section's normal and anatomical up reference, rather than the
in-plane spin of the shared correction rotation. Rotating reference lines can
therefore correct the other sections while the section under the cursor stays
visually fixed.
"""

from __future__ import annotations

from typing import Any

import numpy as np


PLANE_COLORS = {
    "Axial": "#71A9F7",
    "Coronal": "#6BC7A3",
    "Sagittal": "#E5858B",
}

_PLANES = {
    # basis column, signed section normal, anatomical up, fallback right
    "Axial": (2, -1.0, (0.0, -1.0, 0.0), (1.0, 0.0, 0.0)),
    "Coronal": (1, -1.0, (0.0, 0.0, 1.0), (1.0, 0.0, 0.0)),
    "Sagittal": (0, 1.0, (0.0, 0.0, 1.0), (0.0, 1.0, 0.0)),
}


def _plane_name(name: str) -> str:
    normalized = str(name).strip().capitalize()
    if normalized not in _PLANES:
        raise ValueError("Plane must be Axial, Coronal, or Sagittal.")
    return normalized


def _rotation(rotation: Any) -> np.ndarray:
    matrix = np.asarray(rotation, dtype=float)
    if (matrix.shape != (3, 3) or not np.all(np.isfinite(matrix))
            or not np.allclose(matrix.T @ matrix, np.eye(3), atol=1e-6)
            or not np.isclose(np.linalg.det(matrix), 1.0, atol=1e-6)):
        raise ValueError("Rotation must be a finite proper orthonormal 3 x 3 matrix.")
    return matrix


def _point(value: Any, label: str) -> np.ndarray:
    point = np.asarray(value, dtype=float)
    if point.shape != (3,) or not np.all(np.isfinite(point)):
        raise ValueError(f"{label} must be a finite three-component LPS vector.")
    return point


def plane_normal(name: str, rotation: Any) -> np.ndarray:
    """Return a unit signed normal consistent with ``cross(right, up)``."""
    axis, sign, _, _ = _PLANES[_plane_name(name)]
    normal = _rotation(rotation)[:, axis] * sign
    return normal / np.linalg.norm(normal)


def plane_frame(name: str, rotation: Any) -> tuple[np.ndarray, np.ndarray]:
    """Return anatomical screen-right and screen-up unit vectors in LPS.

    Defaults are axial right +X/up -Y, coronal right +X/up +Z, and
    sagittal right +Y/up +Z. Only the plane normal changes this display frame.
    When anatomical up is within 0.06 degrees of the normal, use projected
    anatomical right to avoid amplifying numerical noise near that pole.
    """
    name = _plane_name(name)
    normal = plane_normal(name, rotation)
    _, _, up_reference, right_reference = _PLANES[name]
    up = np.asarray(up_reference, dtype=float)
    up = up - normal * np.dot(up, normal)
    if np.linalg.norm(up) < 1e-3:
        right = np.asarray(right_reference, dtype=float)
        right = right - normal * np.dot(right, normal)
        # The predefined up/right references are orthogonal, so right cannot
        # also be degenerate at the anatomical-up pole.
        right /= np.linalg.norm(right)
        up = np.cross(normal, right)
    else:
        up /= np.linalg.norm(up)
        right = np.cross(up, normal)
        right /= np.linalg.norm(right)
    return right, up


def plane_references(view_name: str, rotation: Any) -> list[dict[str, Any]]:
    """Describe the other sections' intersection lines inside a viewed plane.

    The line passes through the shared crosshair. ``direction`` and ``normal``
    are world LPS unit vectors; display code projects them into its frame.
    """
    view_name = _plane_name(view_name)
    rotation = _rotation(rotation)
    view_normal = plane_normal(view_name, rotation)
    references = []
    for name in _PLANES:
        if name == view_name:
            continue
        normal = plane_normal(name, rotation)
        direction = np.cross(view_normal, normal)
        direction /= np.linalg.norm(direction)
        references.append({"name": name, "normal": normal, "direction": direction,
                           "color": PLANE_COLORS[name]})
    return references


def rotate_about_view(rotation: Any, name: str, angle_degrees: float) -> np.ndarray:
    """Actively rotate the two reference planes around the viewed normal.

    A positive angle rotates screen right toward screen up. The viewed normal
    and its anatomical display frame remain fixed. The output uses the same
    active LPS matrix convention as ``reformat.rotation_matrix``.
    """
    rotation = _rotation(rotation)
    if not np.isfinite(angle_degrees):
        raise ValueError("The rotation angle must be finite.")
    normal = plane_normal(name, rotation)
    x, y, z = normal
    cross = np.array(((0.0, -z, y), (z, 0.0, -x), (-y, x, 0.0)))
    angle = np.deg2rad(float(angle_degrees))
    delta = (np.eye(3) + np.sin(angle) * cross
             + (1.0 - np.cos(angle)) * (cross @ cross))
    return delta @ rotation


def to_euler(rotation: Any) -> tuple[float, float, float]:
    """Return degrees ``(rx, ry, rz)`` for active ``Rz @ Ry @ Rx``.

    At pitch gimbal lock, choose rz = 0 and retain the equivalent x rotation.
    ``atan2`` pitch extraction avoids loss of precision close to 90 degrees.
    """
    matrix = _rotation(rotation)
    cosine_y = float(np.hypot(matrix[0, 0], matrix[1, 0]))
    ry = np.arctan2(-matrix[2, 0], cosine_y)
    if cosine_y > 1e-10:
        rx = np.arctan2(matrix[2, 1], matrix[2, 2])
        rz = np.arctan2(matrix[1, 0], matrix[0, 0])
    else:
        rx = np.arctan2(-matrix[1, 2], matrix[1, 1])
        rz = 0.0
    values = np.rad2deg((rx, ry, rz))
    values[np.abs(values) < 1e-12] = 0.0
    return tuple(float(value) for value in values)


def translate_reference(crosshair: Any, delta_world: Any, view_name: str,
                        target_name: str | None, rotation: Any) -> np.ndarray:
    """Move the shared crosshair using a line or center-handle drag in LPS.

    A reference-line drag moves along that target plane's normal within the
    viewed section, preserving the other reference line's position. A center
    drag (``target_name=None``) permits free motion inside the viewed section.
    The caller clips the result to its acquired volume if needed.
    """
    crosshair = _point(crosshair, "Crosshair")
    delta = _point(delta_world, "Translation")
    view_name = _plane_name(view_name)
    view_normal = plane_normal(view_name, rotation)
    if target_name is None:
        movement = delta - view_normal * np.dot(delta, view_normal)
    else:
        target_name = _plane_name(target_name)
        if target_name == view_name:
            raise ValueError("The reference target must differ from the viewed plane.")
        target_normal = plane_normal(target_name, rotation)
        tangent = target_normal - view_normal * np.dot(target_normal, view_normal)
        tangent /= np.linalg.norm(tangent)
        movement = tangent * np.dot(delta, tangent)
    return crosshair + movement
