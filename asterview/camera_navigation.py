"""Upright patient-space orbit with a fixed drag anchor and bounded elevation."""
import numpy as np


def upright_orbit(position, focus, start, current, viewport):
    position, focus = np.asarray(position, float), np.asarray(focus, float)
    delta = np.asarray(current, float) - np.asarray(start, float)
    offset = position - focus
    radius = float(np.linalg.norm(offset))
    if radius < 1e-9 or not np.any(delta):
        return position.copy()
    width, height = (max(1., float(value)) for value in viewport)
    azimuth = np.arctan2(offset[1], offset[0]) - delta[0] * (2 * np.pi / width)
    elevation = np.arcsin(np.clip(offset[2] / radius, -1., 1.)) - delta[1] * (np.pi / height)
    elevation = np.clip(elevation, np.deg2rad(-89.5), np.deg2rad(89.5))
    direction = np.array([np.cos(elevation) * np.cos(azimuth),
                          np.cos(elevation) * np.sin(azimuth), np.sin(elevation)])
    return focus + radius * direction
