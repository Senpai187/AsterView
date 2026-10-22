"""Restore must invalidate automatic view proposals even at identical geometry."""
from types import SimpleNamespace

import numpy as np
import pytest

from asterview.model import Volume
from asterview.workflow_controls import WorkflowControls


class ProposalHarness(WorkflowControls):
    def __init__(self):
        self.volume = Volume(np.zeros((10, 10, 10), np.float32), (1, 1, 1),
                             np.zeros(3), np.eye(3))
        self.segmentation = SimpleNamespace(revision=7)
        self.crosshair = self.volume.center.copy()
        self.arch_points = []
        self.tmj_centers = {"Right": None, "Left": None}
        self.tmj_angles = {name: SimpleNamespace(value=lambda: 0.)
                           for name in self.tmj_centers}
        self.auto_setup_provenance = {}
        self._auto_setup_epoch = 1
        self._closing = False
        self.message = ""

    def _ensure_automatic_anatomy(self, kind):
        return True

    def volume_fingerprint(self):
        return {"source": "unchanged-test-source"}

    def rotation(self):
        return np.eye(3)

    def start_job(self, compute, install):
        self.install = install
        return True

    def statusBar(self):
        return self

    def showMessage(self, value):
        self.message = value


@pytest.mark.parametrize("workflow", ["auto_arch", "auto_tmj"])
def test_restore_epoch_rejects_inflight_proposal_at_identical_geometry(workflow):
    viewer = ProposalHarness()
    source, masks = viewer.volume, viewer.segmentation
    source_pixels = source.data.copy()
    getattr(viewer, workflow)()
    # This is Restore's generation transition. All geometry remains exactly
    # equal to its proposal-start state, so geometry-only guards cannot help.
    viewer._auto_setup_epoch += 1
    viewer.clear_pending_automatic_setup()
    viewer.install(object())  # rejected before any result content is examined
    assert not viewer.arch_points
    assert all(value is None for value in viewer.tmj_centers.values())
    assert not viewer.auto_setup_provenance
    assert viewer._auto_pano_pending is None
    assert viewer.volume is source and viewer.segmentation is masks
    assert masks.revision == 7
    np.testing.assert_array_equal(source.data, source_pixels)
    assert "changed" in viewer.message
