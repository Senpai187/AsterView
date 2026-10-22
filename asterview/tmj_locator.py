"""Independent bilateral TMJ axis controls in patient coordinates."""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPen

from .slice_view import SliceView


class TMJLocatorView(SliceView):
    """Move a condylar center or rotate its long axis on an axial section.

    Centers retain their source LPS depth when dragged on a different axial
    level. Angles are degrees in the supplied orthonormal correction frame:
    axis = rotation @ [cos(angle), sin(angle), 0]. No anatomy is inferred.
    """

    center_changed = Signal(str, object)
    angle_changed = Signal(str, float)
    COLORS = {"Right": "#71a9f7", "Left": "#e5858b"}
    AXIS_HALF_LENGTH_MM = 12.0

    def __init__(self, title="Axial · TMJ axes", accent="#b9a5d6", parent=None):
        super().__init__(title, accent, parent)
        self.tmj_centers = {"Right": None, "Left": None}
        self.tmj_angles = {"Right": 0.0, "Left": 0.0}
        self.tmj_rotation = np.eye(3)
        self.tmj_axes = {}
        self._tmj_drag = None
        self._tmj_previous_world = None
        self.show_references = False

    def set_tmj_geometry(self, centers, angles, rotation, axes=None):
        rotation = np.asarray(rotation, dtype=float)
        if (rotation.shape != (3, 3) or not np.isfinite(rotation).all()
                or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6)
                or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6)):
            raise ValueError("TMJ rotation must be a finite right-handed orthonormal frame")
        updated_centers, updated_angles = {}, {}
        for side in self.COLORS:
            center = centers.get(side)
            if center is not None:
                center = np.asarray(center, dtype=float)
                if center.shape != (3,) or not np.isfinite(center).all():
                    raise ValueError("TMJ centers must be finite LPS points")
                center = center.copy()
            angle = float(angles.get(side, 0.0))
            if not np.isfinite(angle):
                raise ValueError("TMJ angles must be finite")
            updated_centers[side] = center
            updated_angles[side] = self._normalize_angle(angle)
        updated_axes = {}
        for side, value in (axes or {}).items():
            axis = np.asarray(value, dtype=float)
            if side not in self.COLORS or axis.shape != (3,) or not np.isfinite(axis).all() or np.linalg.norm(axis) < 1e-8:
                raise ValueError("TMJ axes must be finite nonzero LPS directions for a known side")
            updated_axes[side] = axis / np.linalg.norm(axis)
        self.tmj_centers, self.tmj_angles = updated_centers, updated_angles
        self.tmj_axes = updated_axes
        self.tmj_rotation = rotation.copy()
        if self._tmj_drag is not None and self.tmj_centers[self._tmj_drag[1]] is None:
            self._finish_tmj_drag()
        self.update()

    @staticmethod
    def _normalize_angle(angle):
        return float((angle + 180.0) % 360.0 - 180.0)

    def locator_geometry(self):
        """Screen hit geometry recalculated for the current zoom and pan."""
        if self.frame is None or self.data is None or not self.calibrated:
            return []
        result = []
        rect = self._rect()
        for side in self.COLORS:
            center = self.tmj_centers[side]
            if center is None:
                continue
            angle = np.deg2rad(self.tmj_angles[side])
            axis = self.tmj_axes.get(side)
            if axis is None:
                axis = self.tmj_rotation @ np.array([np.cos(angle), np.sin(angle), 0.0])
            q = self.world_to_screen(center)
            endpoint = self.world_to_screen(center + axis * self.AXIS_HALF_LENGTH_MM)
            vector = endpoint - q
            length = np.hypot(vector.x(), vector.y())
            if length < 1e-6 or not rect.contains(q):
                continue
            result.append({"side": side, "center": q, "handle": endpoint,
                           "opposite": q - vector, "direction": vector / length})
        return result

    def _locator_hit(self, point):
        if (self.mode not in ("Navigate", "Correct planes")
                or not self._rect().contains(point)):
            return None, None
        geometry = self.locator_geometry()
        # Centers win when a short projected axis or another handle overlaps.
        for item in geometry:
            delta = point - item["center"]
            if np.hypot(delta.x(), delta.y()) <= 11:
                return "center", item["side"]
        for item in geometry:
            delta = point - item["handle"]
            if np.hypot(delta.x(), delta.y()) <= 11:
                return "rotate", item["side"]
        for item in geometry:
            delta = point - item["center"]
            direction = item["direction"]
            along = delta.x() * direction.x() + delta.y() * direction.y()
            distance = abs(delta.x() * direction.y() - delta.y() * direction.x())
            span = item["handle"] - item["center"]
            if distance <= 6 and abs(along) <= np.hypot(span.x(), span.y()):
                return "axis", item["side"]
        return None, None

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        viewport = QRectF(1, 32, self.width_pixels() - 2, self.height() - 58)
        painter.setClipRect(self._rect().intersected(viewport))
        painter.setFont(QFont("Segoe UI Variable", 9, QFont.Weight.DemiBold))
        for item in self.locator_geometry():
            color = QColor(self.COLORS[item["side"]])
            painter.setPen(QPen(color, 1.6))
            painter.drawLine(item["opposite"], item["handle"])
            painter.setBrush(QColor("#17191d"))
            painter.drawEllipse(item["center"], 4.0, 4.0)
            painter.drawEllipse(item["handle"], 5.0, 5.0)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawText(item["center"] + QPointF(8, -9),
                             f"{item['side'][0]}  {self.tmj_angles[item['side']]:.1f}°")

    def mousePressEvent(self, event):
        kind, side = self._locator_hit(event.position())
        world = self.screen_to_world(event.position())
        left = event.button() == Qt.MouseButton.LeftButton
        right = event.button() == Qt.MouseButton.RightButton
        canvas = QRectF(1, 32, self.width_pixels() - 2, self.height() - 58)
        if right and canvas.contains(event.position()) and self.mode in ("Navigate", "Correct planes"):
            geometry = self.locator_geometry()
            if geometry:
                if side is None:
                    side = min(geometry, key=lambda item: (item["center"] - event.position()).manhattanLength())["side"]
                kind = "rotate"
                world = self.screen_to_world(event.position(), allow_outside=True)
        if world is not None and kind is not None and (left or right):
            action = "rotate-relative" if right else ("rotate" if kind == "rotate" else "center")
            self._tmj_drag = (action, side)
            self._tmj_previous_world = world
            self.pending_measure = None
            self.interaction_started.emit()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._tmj_drag is None:
            kind, _ = self._locator_hit(event.position())
            if kind is not None and self.last_pos is None:
                cursor = Qt.CursorShape.OpenHandCursor if kind == "rotate" else Qt.CursorShape.SizeAllCursor
                self.setCursor(cursor)
                return
            super().mouseMoveEvent(event)
            return
        canvas = QRectF(1, 32, self.width_pixels() - 2, self.height() - 58)
        world = self.screen_to_world(event.position(), allow_outside=True) if canvas.contains(event.position()) else None
        if world is None:
            # Re-entering the viewport must not accumulate an off-screen jump.
            self._tmj_previous_world = None
            return
        action, side = self._tmj_drag
        center = self.tmj_centers[side]
        if center is None:
            self._finish_tmj_drag()
            return
        if action == "center":
            if self._tmj_previous_world is not None:
                center = center + world - self._tmj_previous_world
                self.tmj_centers[side] = center
                self.center_changed.emit(side, center.copy())
        else:
            delta = world - center
            x = float(np.dot(delta, self.tmj_rotation[:, 0]))
            y = float(np.dot(delta, self.tmj_rotation[:, 1]))
            if np.hypot(x, y) >= 1.0:
                angle = self._normalize_angle(np.rad2deg(np.arctan2(y, x)))
                if action == "rotate-relative" and self._tmj_previous_world is not None:
                    previous = self._tmj_previous_world - center
                    px = float(np.dot(previous, self.tmj_rotation[:, 0]))
                    py = float(np.dot(previous, self.tmj_rotation[:, 1]))
                    if np.hypot(px, py) >= 1.0:
                        angle = self.tmj_angles[side] + self._normalize_angle(
                            angle - np.rad2deg(np.arctan2(py, px)))
                    else:
                        angle = self.tmj_angles[side]
                elif action == "rotate-relative":
                    angle = self.tmj_angles[side]
                elif side in self.tmj_axes:
                    axis = self.tmj_axes[side]
                    projected = np.rad2deg(np.arctan2(axis @ self.tmj_rotation[:, 1],
                                                     axis @ self.tmj_rotation[:, 0]))
                    angle -= self._normalize_angle(projected - self.tmj_angles[side])
                # Either end of an undirected condylar axis can be grabbed.
                # Select the equivalent angle nearest the last value.
                previous = self.tmj_angles[side]
                candidates = [self._normalize_angle(angle + offset) for offset in (0.0, 180.0)]
                angle = min(candidates, key=lambda value: abs(self._normalize_angle(value - previous)))
                if side in self.tmj_axes:
                    radians = np.deg2rad(self._normalize_angle(angle - previous))
                    cosine, sine = np.cos(radians), np.sin(radians)
                    turn = np.array([[cosine, -sine, 0.], [sine, cosine, 0.], [0., 0., 1.]])
                    self.tmj_axes[side] = self.tmj_rotation @ turn @ self.tmj_rotation.T @ self.tmj_axes[side]
                self.tmj_angles[side] = angle
                self.angle_changed.emit(side, angle)
        self._tmj_previous_world = world
        self.update()
        event.accept()

    def _finish_tmj_drag(self):
        if self._tmj_drag is not None:
            self._tmj_drag = None
            self._tmj_previous_world = None
            self.unsetCursor()
            self.interaction_finished.emit()

    def mouseReleaseEvent(self, event):
        if self._tmj_drag is not None:
            self._finish_tmj_drag()
            event.accept()
            return
        super().mouseReleaseEvent(event)
