"""Source-bound continuation state for local dental workflow preparation.

This module does not run a model or modify a workstation. A request captures the
manual setup at the click, so completing an anatomy job cannot overwrite edits
or apply a continuation to another scan.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import numpy as np


@dataclass
class AutomaticSetupRequest:
    kind: str
    volume: object
    segmentation: object
    segmentation_revision: int | None
    epoch: int
    rotation: np.ndarray
    crosshair: np.ndarray
    arch_points: np.ndarray
    tmj_centers: dict
    tmj_angles: dict

    @classmethod
    def capture(cls, workstation, kind):
        if kind not in ("arch", "tmj"):
            raise ValueError("Unsupported automatic dental workflow.")
        segmentation = workstation.segmentation
        return cls(kind, workstation.volume, segmentation,
                   None if segmentation is None else segmentation.revision,
                   getattr(workstation, "_auto_setup_epoch", 0),
                   workstation.rotation().copy(), workstation.crosshair.copy(),
                   np.asarray(workstation.arch_points).copy(),
                   {key: None if value is None else value.copy()
                    for key, value in workstation.tmj_centers.items()},
                   {key: item.value() for key, item in workstation.tmj_angles.items()})

    def with_segmentation(self, segmentation):
        """Bind a successful source-checked anatomy result to this continuation."""
        return replace(self, segmentation=segmentation,
                       segmentation_revision=segmentation.revision)

    def source_matches(self, workstation):
        return (workstation.volume is self.volume
                and getattr(workstation, "_auto_setup_epoch", 0) == self.epoch
                and not workstation._closing)

    def matches(self, workstation):
        if (not self.source_matches(workstation)
                or workstation.segmentation is not self.segmentation
                or self.segmentation is not None
                and self.segmentation.revision != self.segmentation_revision
                or not np.allclose(self.rotation, workstation.rotation())):
            return False
        if self.kind == "arch":
            normal = self.rotation[:, 2]
            return (np.array_equal(self.arch_points, np.asarray(workstation.arch_points))
                    and abs(np.dot(workstation.crosshair - self.crosshair, normal)) <= 1e-5)
        return (all((value is None and workstation.tmj_centers[key] is None)
                    or (value is not None and workstation.tmj_centers[key] is not None
                        and np.array_equal(value, workstation.tmj_centers[key]))
                    for key, value in self.tmj_centers.items())
                and all(value == workstation.tmj_angles[key].value()
                        for key, value in self.tmj_angles.items()))

