"""Geometry/memory checks; GPU rendering is exercised separately on real hardware."""
from types import SimpleNamespace
import os
import threading
import time
import unittest
from unittest.mock import patch

import numpy as np
from vtkmodules.util.numpy_support import vtk_to_numpy
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QThread
from vtkmodules.vtkCommonCore import vtkCommand
from vtkmodules.vtkCommonCore import vtkOutputWindow, vtkStringOutputWindow

import asterview.render3d as render3d
from asterview.render3d import (
    VolumeView, _make_image, _plane_box_polygon, _proxy_shape, _volume_corners,
)
from asterview.reformat import rotation_matrix


class VolumeImageTests(unittest.TestCase):
    def setUp(self):
        angle = np.deg2rad(28)
        self.volume = SimpleNamespace(
            data=np.arange(9 * 11 * 15, dtype=np.float32).reshape((9, 11, 15)),
            spacing=(0.4, 0.6, 1.1),
            origin=np.array((12.0, -8.0, 36.0)),
            direction=np.array([
                [np.cos(angle), -np.sin(angle), 0.0],
                [np.sin(angle), np.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ]),
        )

    def nerve_path(self, identity="right-canal", **changes):
        local = np.array(((.8, 1.2, 2.2), (2.8, 3.0, 4.4), (4.8, 4.8, 6.6)))
        points = local @ self.volume.direction.T + self.volume.origin
        curve = np.vstack((points[0], (points[0] + points[1]) / 2, points[1],
                           (points[1] + points[2]) / 2, points[2]))
        path = {"id": identity, "points_lps_mm": points.tolist(),
                "sampled_points_lps_mm": curve.tolist(), "diameter_mm": 1.2,
                "color": "#55AABB", "review": "pending", "visible": True}
        path.update(changes)
        return path

    @staticmethod
    def world_display(widget, point):
        widget.renderer.SetWorldPoint(*point, 1.)
        widget.renderer.WorldToDisplay()
        return np.asarray(widget.renderer.GetDisplayPoint())

    def test_nerve_tube_and_handles_preserve_oblique_lps_geometry_without_texture_upload(self):
        app = QApplication.instance() or QApplication([])
        widget = VolumeView()
        try:
            widget.set_volume(self.volume)
            image = widget._image
            scalar_mtime = image.GetPointData().GetScalars().GetMTime()
            mask = np.ones(self.volume.data.shape, dtype=np.uint16)
            widget.set_segmentation(mask, {1: {"color": "#55AABB", "review": "pending"}}, {1})
            segment_image = widget._segment_image
            path = self.nerve_path()
            widget.set_nerve_paths([path], path["id"], 1, edit_enabled=True)
            record = widget._nerve_paths[path["id"]]
            actor, tube = record["actor"], record["tube"]
            handles = list(record["handles"])
            np.testing.assert_allclose(vtk_to_numpy(record["polydata"].GetPoints().GetData()),
                                       path["sampled_points_lps_mm"], atol=1e-12)
            np.testing.assert_allclose([source.GetCenter() for source, actor in handles],
                                       path["points_lps_mm"], atol=1e-12)
            self.assertAlmostEqual(tube.GetRadius(), .6)
            self.assertTrue(all(actor.GetVisibility() for source, actor in handles))
            self.assertGreater(handles[1][0].GetRadius(), handles[0][0].GetRadius())
            self.assertIs(widget._node_renderer.GetActiveCamera(), widget.renderer.GetActiveCamera())
            self.assertEqual(widget._node_renderer.GetLayer(), 1)
            self.assertEqual(widget.marker.GetRenderer().GetLayer(), 2)
            self.assertFalse(widget._node_renderer.GetInteractive())
            self.assertTrue(widget._node_renderer.GetPreserveColorBuffer())
            self.assertFalse(widget._node_renderer.GetPreserveDepthBuffer())
            for source, node_actor in handles:
                self.assertTrue(widget._node_renderer.HasViewProp(node_actor))
                self.assertFalse(widget.renderer.HasViewProp(node_actor))
            path["points_lps_mm"][1][0] += .3
            path["sampled_points_lps_mm"][2][0] += .3
            widget.set_nerve_paths([path], path["id"], 1, edit_enabled=True)
            self.assertIs(record["actor"], actor)
            self.assertIs(record["tube"], tube)
            self.assertEqual(record["handles"], handles)
            np.testing.assert_allclose(handles[1][0].GetCenter(), path["points_lps_mm"][1])
            self.assertIs(widget._image, image)
            self.assertIs(widget.mapper.GetInput(), image)
            self.assertIs(widget._segment_image, segment_image)
            self.assertEqual(image.GetPointData().GetScalars().GetMTime(), scalar_mtime)
        finally:
            widget.cleanup()

    def test_nerve_visibility_review_and_selection_gate_display_and_picking(self):
        app = QApplication.instance() or QApplication([])
        widget = VolumeView()
        try:
            widget.set_volume(self.volume)
            shown = self.nerve_path("shown")
            rejected = self.nerve_path("rejected", review="rejected")
            hidden = self.nerve_path("hidden", visible=False)
            widget.set_nerve_paths([shown, rejected, hidden], "shown", 0, edit_enabled=True)
            self.assertTrue(widget._nerve_paths["shown"]["actor"].GetVisibility())
            for identity in ("rejected", "hidden"):
                self.assertFalse(widget._nerve_paths[identity]["actor"].GetVisibility())
                self.assertTrue(all(not actor.GetVisibility()
                                    for source, actor in widget._nerve_paths[identity]["handles"]))
            self.assertEqual(set(widget._nerve_node_lookup.values()), {("shown", 0), ("shown", 1), ("shown", 2)})
            widget.set_nerve_paths([shown], "shown", 0, edit_enabled=False)
            self.assertEqual(widget._nerve_node_lookup, {})
            self.assertTrue(widget._nerve_paths["shown"]["handles"][0][1].GetVisibility())
        finally:
            widget.cleanup()

    def test_framing_a_nerve_path_is_explicit_and_preserves_camera_orientation(self):
        app = QApplication.instance() or QApplication([])
        widget = VolumeView()
        try:
            widget.set_volume(self.volume)
            widget.set_view_orientation("Isometric")
            path = self.nerve_path()
            path["points_lps_mm"] = (np.asarray(path["points_lps_mm"]) * .25
                                      + np.asarray(path["points_lps_mm"])[1] * .75).tolist()
            path["sampled_points_lps_mm"] = path["points_lps_mm"]
            camera = widget.renderer.GetActiveCamera()
            original = camera.GetPosition(), camera.GetFocalPoint(), camera.GetViewUp()
            direction = camera.GetDirectionOfProjection()
            scale = camera.GetParallelScale()
            widget.set_nerve_paths([path], path["id"], 1, edit_enabled=True)
            self.assertEqual((camera.GetPosition(), camera.GetFocalPoint(), camera.GetViewUp()), original)
            self.assertTrue(widget.focus_nerve_path(path["id"]))
            np.testing.assert_allclose(camera.GetDirectionOfProjection(), direction, atol=1e-12)
            np.testing.assert_allclose(camera.GetViewUp(), original[2], atol=1e-12)
            points = np.asarray(path["points_lps_mm"])
            np.testing.assert_allclose(camera.GetFocalPoint(), (points.min(axis=0) + points.max(axis=0)) / 2)
            self.assertLess(camera.GetParallelScale(), scale)
            self.assertFalse(widget.focus_nerve_path("missing"))
        finally:
            widget.cleanup()

    def test_native_nerve_node_pick_and_drag_are_camera_plane_constrained_in_lps(self):
        app = QApplication.instance() or QApplication([])
        for perspective in (False, True):
            with self.subTest(perspective=perspective):
                widget = VolumeView()
                try:
                    widget.set_volume(self.volume)
                    widget._window.SetSize(800, 600)
                    widget.set_view_orientation("Isometric")
                    path = self.nerve_path()
                    node = np.asarray(path["points_lps_mm"])[1]
                    widget.set_nerve_paths([path], path["id"], 1, edit_enabled=True)
                    camera = widget.renderer.GetActiveCamera()
                    camera.SetParallelProjection(not perspective)
                    camera.OrthogonalizeViewUp()
                    display = np.round(self.world_display(widget, node)[:2]).astype(int)
                    self.assertEqual(widget._pick_nerve_node(*display), (path["id"], 1))
                    events, threads, downstream = [], [], []
                    widget.path_node_dragged.connect(lambda event: (events.append(event), threads.append(QThread.currentThread())))

                    def synchronize_model(event):
                        if event["phase"] in ("move", "end"):
                            path["points_lps_mm"][event["node_index"]] = event["position_lps_mm"].tolist()
                            path["sampled_points_lps_mm"] = path["points_lps_mm"]
                        widget.set_nerve_paths([path], path["id"], event["node_index"], edit_enabled=True)

                    widget.path_node_dragged.connect(synchronize_model)
                    for event_id in (vtkCommand.LeftButtonPressEvent, vtkCommand.MouseMoveEvent,
                                     vtkCommand.LeftButtonReleaseEvent):
                        widget._interactor.AddObserver(event_id, lambda *args: downstream.append(args), .5)
                    original_camera = (camera.GetPosition(), camera.GetFocalPoint(), camera.GetViewUp())
                    widget._interactor.SetEventPosition(*display)
                    widget._interactor.InvokeEvent(vtkCommand.LeftButtonPressEvent)
                    self.assertIsNotNone(widget._nerve_drag)
                    normal = np.asarray(camera.GetDirectionOfProjection())
                    right = np.cross(normal, camera.GetViewUp())
                    right /= np.linalg.norm(right)
                    up = np.cross(right, normal)
                    if perspective:
                        depth = float((node - np.asarray(camera.GetPosition())) @ normal)
                        pitch = 2 * depth * np.tan(np.deg2rad(camera.GetViewAngle() / 2)) / 600
                    else:
                        pitch = 2 * camera.GetParallelScale() / 600
                    for offset in ((9, -5), (27, -16)):
                        widget._interactor.SetEventPosition(*(display + offset))
                        widget._interactor.InvokeEvent(vtkCommand.MouseMoveEvent)
                    widget._interactor.InvokeEvent(vtkCommand.LeftButtonReleaseEvent)
                    self.assertEqual([event["phase"] for event in events], ["start", "move", "move", "end"])
                    self.assertTrue(all(event["path_id"] == path["id"] and event["node_index"] == 1 for event in events))
                    np.testing.assert_allclose(events[0]["position_lps_mm"], node, atol=1e-10)
                    expected = node + pitch * (right * 27 - up * 16)
                    np.testing.assert_allclose(events[-1]["position_lps_mm"], expected, atol=1e-9)
                    self.assertAlmostEqual(float((events[-1]["position_lps_mm"] - node) @ normal), 0, places=10)
                    self.assertEqual(threads, [app.thread()] * 4)
                    self.assertEqual(downstream, [])
                    self.assertIsNone(widget._nerve_drag)
                    np.testing.assert_allclose(widget._nerve_paths[path["id"]]["nodes"][1], expected, atol=1e-9)
                    self.assertEqual((camera.GetPosition(), camera.GetFocalPoint(), camera.GetViewUp()), original_camera)
                    # A miss must propagate to existing camera/event observers.
                    widget._interactor.SetEventPosition(1, 1)
                    widget._interactor.InvokeEvent(vtkCommand.LeftButtonPressEvent)
                    self.assertTrue(downstream)
                finally:
                    widget.cleanup()

    def test_nerve_geometry_requires_gui_thread_and_source_change_clears_paths(self):
        app = QApplication.instance() or QApplication([])
        widget = VolumeView()
        try:
            widget.set_volume(self.volume)
            path = self.nerve_path()
            errors = []

            def wrong_thread():
                try:
                    widget.set_nerve_paths([path])
                except Exception as error:
                    errors.append(error)

            worker = threading.Thread(target=wrong_thread)
            worker.start()
            worker.join()
            self.assertEqual(len(errors), 1)
            self.assertIsInstance(errors[0], RuntimeError)
            self.assertIn("GUI thread", str(errors[0]))
            self.assertEqual(widget._nerve_paths, {})
            widget.set_nerve_paths([path], path["id"], 1, edit_enabled=True)
            actor = widget._nerve_paths[path["id"]]["actor"]
            handle = widget._nerve_paths[path["id"]]["handles"][0][1]
            newer = SimpleNamespace(**vars(self.volume))
            widget.set_volume(newer)
            self.assertEqual(widget._nerve_paths, {})
            self.assertEqual(widget._nerve_node_lookup, {})
            self.assertIsNone(widget._nerve_drag)
            self.assertFalse(widget.renderer.HasViewProp(actor))
            self.assertFalse(widget._node_renderer.HasViewProp(handle))
        finally:
            widget.cleanup()

    def test_invalid_nerve_path_update_is_atomic(self):
        app = QApplication.instance() or QApplication([])
        widget = VolumeView()
        try:
            widget.set_volume(self.volume)
            path = self.nerve_path()
            widget.set_nerve_paths([path], path["id"], 1, edit_enabled=True)
            actor = widget._nerve_paths[path["id"]]["actor"]
            with self.assertRaisesRegex(ValueError, "finite N x 3"):
                widget.set_nerve_paths([self.nerve_path("invalid", points_lps_mm=[[0, np.nan, 0]])])
            self.assertEqual(set(widget._nerve_paths), {path["id"]})
            self.assertIs(widget._nerve_paths[path["id"]]["actor"], actor)
            self.assertTrue(widget._nerve_edit_enabled)
        finally:
            widget.cleanup()

    def test_full_resolution_scalars_share_original_memory(self):
        image, shared, proxy = _make_image(self.volume, "Balanced")
        self.assertFalse(proxy)
        self.assertIs(shared, self.volume.data)
        self.assertTrue(np.shares_memory(vtk_to_numpy(image.GetPointData().GetScalars()), shared))
        self.assertEqual(image.GetDimensions(), (15, 11, 9))
        world = [0.0, 0.0, 0.0]
        image.TransformContinuousIndexToPhysicalPoint((4.3, 5.1, 6.8), world)
        expected = self.volume.origin + self.volume.direction @ (np.array((4.3, 5.1, 6.8)) * self.volume.spacing)
        np.testing.assert_allclose(world, expected)

    def test_bounded_category_range_preserves_high_uint16_ids_and_bounds_lookup_size(self):
        app = QApplication.instance() or QApplication([])
        widget = VolumeView()
        try:
            widget.set_volume(self.volume)
            mask = np.zeros(self.volume.data.shape, dtype=np.uint16)
            mask[2:5, 2:5, 2:5] = 1001
            mask[5:7, 6:9, 7:11] = 1032
            labels = {1001: {"color": "#FF0000", "review": "pending"},
                      1002: {"color": "#00FF00", "review": "rejected"},
                      1032: {"color": "#0000FF", "review": "pending"}}
            widget.set_segmentation(mask, labels, set(labels))
            self.assertFalse(widget._segment_display_remapped)
            self.assertIs(widget._segment_display_data, mask)
            self.assertTrue(np.shares_memory(vtk_to_numpy(widget._segment_image.GetPointData().GetScalars()), mask))
            self.assertEqual(widget.segment_mapper.GetColorRangeType(), widget.segment_mapper.SCALAR)
            self.assertEqual(widget.segment_mapper.GetScalarOpacityRangeType(), widget.segment_mapper.SCALAR)
            color = widget.segment_property.GetRGBTransferFunction()
            opacity = widget.segment_property.GetScalarOpacity()
            self.assertEqual(color.GetRange(), (1000.5, 1032.5))
            np.testing.assert_allclose(color.GetColor(1001), (1, 0, 0))
            np.testing.assert_allclose(color.GetColor(1032), (0, 0, 1))
            for value in (0, 1, 1000, 1002, 1003, 1020, 1033, 65535):
                self.assertEqual(opacity.GetValue(value), 0)
            self.assertEqual(opacity.GetValue(1001), .8)
            self.assertEqual(opacity.GetValue(1032), .8)
            self.assertLessEqual(color.EstimateMinNumberOfSamples(*color.GetRange()), 2048)
            self.assertLessEqual(opacity.EstimateMinNumberOfSamples(*opacity.GetRange()), 2048)
            actual_range = widget._segment_image.GetScalarRange()
            self.assertLessEqual(color.EstimateMinNumberOfSamples(*actual_range), 8192)
            self.assertLessEqual(opacity.EstimateMinNumberOfSamples(*actual_range), 8192)
        finally:
            widget.cleanup()

    def test_sparse_uint16_categories_compact_display_only_and_refresh_after_mask_edit(self):
        app = QApplication.instance() or QApplication([])
        widget = VolumeView()
        try:
            widget.set_volume(self.volume)
            mask = np.zeros(self.volume.data.shape, dtype=np.uint16)
            mask[2:4, 2:4, 2:4] = 1
            mask[5:7, 6:9, 7:11] = 65535
            original = mask.copy()
            labels = {1: {"color": "#FF0000", "review": "pending"},
                      30000: {"color": "#00FF00", "review": "pending"},
                      65535: {"color": "#0000FF", "review": "pending"}}
            widget.set_segmentation(mask, labels, set(labels))
            self.assertTrue(widget._segment_display_remapped)
            self.assertIs(widget._segment_data, mask)
            display = widget._segment_display_data
            image = widget._segment_image
            self.assertEqual(display.dtype, np.uint8)
            self.assertFalse(np.shares_memory(display, mask))
            np.testing.assert_array_equal(display, np.where(mask == 65535, 3, mask))
            np.testing.assert_array_equal(mask, original)
            scalars = vtk_to_numpy(image.GetPointData().GetScalars()).reshape(mask.shape)
            self.assertTrue(np.shares_memory(scalars, display))
            mask[3, 3, 3] = 30000
            widget.segmentation_modified()
            self.assertIs(widget._segment_image, image)
            self.assertIs(widget._segment_display_data, display)
            self.assertEqual(scalars[3, 3, 3], 2)
            self.assertEqual(mask[3, 3, 3], 30000)
            widget.set_segmentation(mask, labels, {65535})
            self.assertIs(widget._segment_image, image)
            opacity = widget.segment_property.GetScalarOpacity()
            self.assertEqual(opacity.GetValue(1), 0)
            self.assertEqual(opacity.GetValue(2), 0)
            self.assertEqual(opacity.GetValue(3), .8)
            widget.set_segmentation(None)
            self.assertIsNone(widget._segment_display_data)
            self.assertIsNone(widget._segment_display_lookup)
            self.assertFalse(widget._segment_display_remapped)
        finally:
            widget.cleanup()

    def test_512_sparse_categories_have_a_bounded_uint16_display_palette(self):
        app = QApplication.instance() or QApplication([])
        widget = VolumeView()
        try:
            widget.set_volume(self.volume)
            keys = [1 + 128 * number for number in range(512)]
            mask = np.resize(np.asarray(keys, dtype=np.uint16), self.volume.data.shape)
            labels = {key: {"color": "#66AADD", "review": "pending"} for key in keys}
            widget.set_segmentation(mask, labels, set(keys))
            self.assertEqual(widget._segment_display_data.dtype, np.uint16)
            self.assertEqual(widget._segment_display_ids[keys[-1]], 512)
            for function in (widget.segment_property.GetRGBTransferFunction(),
                             widget.segment_property.GetScalarOpacity()):
                self.assertLessEqual(function.EstimateMinNumberOfSamples(*function.GetRange()), 2048)
        finally:
            widget.cleanup()

    @unittest.skipUnless(os.environ.get("ASTERVIEW_GPU_TESTS") == "1", "Explicit real-display GPU verification")
    def test_clean_gpu_capture_temporarily_hides_axes_and_restores_marker_on_failure(self):
        app = QApplication.instance() or QApplication([])
        widget = VolumeView()
        try:
            self.volume.is_hu = False
            widget.resize(640, 480)
            widget.set_volume(self.volume)
            widget.show()
            deadline = time.monotonic() + 5
            while widget._gpu_supported is None and time.monotonic() < deadline:
                app.processEvents()
                time.sleep(.002)
            self.assertTrue(widget._gpu_supported, widget.info)
            self.assertTrue(widget.marker.GetEnabled())
            source_image = widget._image
            camera = tuple(widget.renderer.GetActiveCamera().GetPosition())
            capture_factory = render3d.vtkWindowToImageFilter

            def without_marker():
                self.assertFalse(widget.marker.GetEnabled())
                return capture_factory()

            with patch.object(render3d, "vtkWindowToImageFilter", side_effect=without_marker):
                image = widget.capture_image(clean=True)
            self.assertFalse(image.isNull())
            self.assertGreater(image.width(), 0)
            self.assertGreater(image.height(), 0)
            self.assertTrue(widget.marker.GetEnabled())
            self.assertFalse(widget.marker.GetInteractive())
            self.assertIs(widget._image, source_image)
            self.assertEqual(tuple(widget.renderer.GetActiveCamera().GetPosition()), camera)

            with patch.object(render3d, "vtkWindowToImageFilter",
                              side_effect=RuntimeError("Controlled capture error")):
                with self.assertRaisesRegex(RuntimeError, "Controlled capture error"):
                    widget.capture_image(clean=True)
            self.assertTrue(widget.marker.GetEnabled())

            widget.marker.SetEnabled(False)
            image = widget.capture_image(clean=True)
            self.assertFalse(image.isNull())
            self.assertFalse(widget.marker.GetEnabled())
        finally:
            widget.close()
            app.processEvents()

    @unittest.skipUnless(os.environ.get("ASTERVIEW_GPU_TESTS") == "1", "Explicit real-display GPU verification")
    def test_gpu_high_and_sparse_uint16_categories_render_their_own_colors_without_lut_warning(self):
        app = QApplication.instance() or QApplication([])
        previous_output = vtkOutputWindow.GetInstance()
        output = vtkStringOutputWindow()
        vtkOutputWindow.SetInstance(output)
        try:
            for identities in ((1001, 1002, 1032), (1, 30000, 65535)):
                with self.subTest(identities=identities):
                    volume = SimpleNamespace(data=np.zeros((24, 24, 56), dtype=np.float32),
                        spacing=(1., 1., 1.), origin=np.zeros(3), direction=np.eye(3), is_hu=False)
                    widget = VolumeView()
                    try:
                        widget.resize(720, 480)
                        widget.set_volume(volume)
                        widget.set_volume_visibility(False)
                        widget.renderer.GradientBackgroundOff()
                        widget.renderer.SetBackground(0, 0, 0)
                        widget.annotation.SetVisibility(False)
                        mask = np.zeros(volume.data.shape, dtype=np.uint16)
                        centers = []
                        colors = ("#FF0000", "#00FF00", "#0000FF")
                        for number, identity in enumerate(identities):
                            start = 3 + 20 * number
                            mask[6:18, 4:20, start:start + 10] = identity
                            centers.append(np.array((start + 4.5, 11.5, 11.5)))
                        labels = {identity: {"color": color, "review": "pending"}
                                  for identity, color in zip(identities, colors)}
                        widget.set_segmentation(mask, labels, set(identities))
                        widget.set_view_orientation("Anterior")
                        widget.show()
                        deadline = time.monotonic() + 5
                        while widget._gpu_supported is None and time.monotonic() < deadline:
                            app.processEvents()
                            time.sleep(.002)
                        self.assertTrue(widget._gpu_supported, widget.info)
                        widget.marker.SetEnabled(False)
                        widget._render()

                        def capture():
                            image_filter = render3d.vtkWindowToImageFilter()
                            image_filter.SetInput(widget._window)
                            image_filter.SetInputBufferTypeToRGB()
                            image_filter.ReadFrontBufferOff()
                            image_filter.Update()
                            image = image_filter.GetOutput()
                            width, height, _ = image.GetDimensions()
                            return vtk_to_numpy(image.GetPointData().GetScalars()).reshape(height, width, 3).copy()

                        pixels = capture()
                        for channel, center in enumerate(centers):
                            x, y = np.round(self.world_display(widget, center)[:2]).astype(int)
                            rgb = pixels[y - 2:y + 3, x - 2:x + 3].mean(axis=(0, 1))
                            self.assertGreater(rgb[channel], 220, (identities, channel, rgb))
                            self.assertLess(np.delete(rgb, channel).max(), 10, (identities, channel, rgb))
                        widget.set_segmentation(mask, labels, {identities[0], identities[2]})
                        widget._render()
                        pixels = capture()
                        x, y = np.round(self.world_display(widget, centers[1])[:2]).astype(int)
                        self.assertLess(pixels[y - 2:y + 3, x - 2:x + 3].max(), 5)
                        self.assertIs(widget._segment_data, mask)
                    finally:
                        widget.close()
                        app.processEvents()
            self.assertNotIn("incorrect lookup table", output.GetOutput().lower())
            self.assertNotIn("required texture size", output.GetOutput().lower())
        finally:
            vtkOutputWindow.SetInstance(previous_output)

    @unittest.skipUnless(os.environ.get("ASTERVIEW_GPU_TESTS") == "1", "Explicit real-display GPU verification")
    def test_gpu_node_handles_remain_visible_inside_opaque_mask_and_tube_and_keep_lps_pick(self):
        app = QApplication.instance() or QApplication([])
        widget = VolumeView()
        try:
            volume = SimpleNamespace(data=np.zeros((40, 40, 40), dtype=np.float32),
                spacing=(1., 1., 1.), origin=np.zeros(3), direction=np.eye(3), is_hu=False)
            widget.resize(640, 480)
            widget.set_volume(volume)
            widget.set_volume_visibility(False)
            widget.renderer.GradientBackgroundOff()
            widget.renderer.SetBackground(0, 0, 0)
            widget.annotation.SetVisibility(False)
            mask = np.zeros(volume.data.shape, dtype=np.uint16)
            mask[5:35, 5:35, 5:35] = 1001
            widget.set_segmentation(mask, {1001: {"color": "#CC4455", "review": "accepted"}}, {1001})
            nodes = np.array(((10., 20., 20.), (20., 20., 20.), (30., 20., 20.)))
            path = {"id": "occluded-canal", "points_lps_mm": nodes, "diameter_mm": 4.,
                    "color": "#CC4455", "review": "pending", "visible": True}
            widget.set_nerve_paths([path], path["id"], 1, edit_enabled=True)
            widget.set_view_orientation("Anterior")
            widget.show()
            deadline = time.monotonic() + 5
            while widget._gpu_supported is None and time.monotonic() < deadline:
                app.processEvents()
                time.sleep(.002)
            self.assertTrue(widget._gpu_supported, widget.info)
            widget.marker.SetEnabled(False)

            def capture():
                widget._render()
                grab = render3d.vtkWindowToImageFilter()
                grab.SetInput(widget._window)
                grab.SetInputBufferTypeToRGB()
                grab.ReadFrontBufferOff()
                grab.Update()
                image = grab.GetOutput()
                width, height, _ = image.GetDimensions()
                return vtk_to_numpy(image.GetPointData().GetScalars()).reshape(height, width, 3).copy()

            camera = widget.renderer.GetActiveCamera()
            original_camera = camera.GetPosition(), camera.GetFocalPoint(), camera.GetViewUp()
            widget._node_renderer.DrawOff()
            anatomy = capture()
            widget._node_renderer.DrawOn()
            foreground = capture()
            for index, node in enumerate(nodes):
                x, y = np.round(self.world_display(widget, node)[:2]).astype(int)
                before = anatomy[y - 1:y + 2, x - 1:x + 2].mean(axis=(0, 1))
                after = foreground[y - 1:y + 2, x - 1:x + 2].mean(axis=(0, 1))
                self.assertLess(before[1], 120, (index, before))
                if index == 1:
                    self.assertGreater(after[0], 230, after)
                    self.assertGreater(after[1], 200, after)
                    self.assertTrue(100 < after[2] < 200, after)
                else:
                    self.assertGreater(after.min(), 230, (index, after))
                self.assertEqual(widget._pick_nerve_node(x, y), (path["id"], index))
            x, y = np.round(self.world_display(widget, (8, 10, 8))[:2]).astype(int)
            np.testing.assert_array_equal(foreground[y, x], anatomy[y, x])
            events = []
            widget.path_node_dragged.connect(events.append)
            x, y = np.round(self.world_display(widget, nodes[1])[:2]).astype(int)
            widget._interactor.SetEventPosition(x, y)
            widget._interactor.InvokeEvent(vtkCommand.LeftButtonPressEvent)
            widget._interactor.SetEventPosition(x + 14, y - 9)
            widget._interactor.InvokeEvent(vtkCommand.MouseMoveEvent)
            widget._interactor.InvokeEvent(vtkCommand.LeftButtonReleaseEvent)
            self.assertEqual([event["phase"] for event in events], ["start", "move", "end"])
            np.testing.assert_allclose(events[0]["position_lps_mm"], nodes[1])
            normal = np.asarray(camera.GetDirectionOfProjection())
            self.assertAlmostEqual(float((events[1]["position_lps_mm"] - nodes[1]) @ normal), 0)
            self.assertEqual((camera.GetPosition(), camera.GetFocalPoint(), camera.GetViewUp()), original_camera)
        finally:
            widget.close()
            app.processEvents()

    def test_reference_polygon_is_clipped_to_acquired_oblique_box(self):
        corners = _volume_corners(self.volume)
        center = corners.mean(axis=0) + np.array((.2, -.4, .1))
        rotation = rotation_matrix(31, -17, 24)
        extent = (np.asarray(self.volume.data.shape[::-1]) - 1) * self.volume.spacing
        for axis in range(3):
            normal = rotation[:, axis]
            polygon = _plane_box_polygon(corners, center, normal)
            self.assertGreaterEqual(len(polygon), 3)
            self.assertLessEqual(len(polygon), 6)
            np.testing.assert_allclose((polygon - center) @ normal, 0, atol=1e-9)
            local = (polygon - self.volume.origin) @ self.volume.direction
            self.assertTrue(np.all(local >= -1e-8))
            self.assertTrue(np.all(local <= extent + 1e-8))
            # Every vertex must lie on a real acquired-box face, including
            # when that face is oblique to patient-coordinate axes.
            self.assertTrue(np.all(np.any(np.isclose(local, 0, atol=1e-8)
                                         | np.isclose(local, extent, atol=1e-8), axis=1)))
            edge_normals = np.cross(np.roll(polygon, -1, axis=0) - polygon,
                                    np.roll(polygon, -2, axis=0) - np.roll(polygon, -1, axis=0))
            signs = edge_normals @ normal
            self.assertTrue(np.all(signs >= -1e-8) or np.all(signs <= 1e-8))

    def test_reference_polygon_handles_face_diagonal_and_outside_planes(self):
        corners = render3d._BOX_CORNER_BITS.astype(float)
        face = _plane_box_polygon(corners, np.zeros(3), np.array((0, 0, 1)))
        self.assertEqual(len(face), 4)
        np.testing.assert_allclose(face[:, 2], 0)
        diagonal = _plane_box_polygon(corners, np.full(3, .5), np.ones(3))
        self.assertEqual(len(diagonal), 6)
        outside = _plane_box_polygon(corners, np.full(3, 2.), np.array((0, 0, 1)))
        self.assertEqual(outside.shape, (0, 3))
        tangent = _plane_box_polygon(corners, np.zeros(3), np.ones(3))
        self.assertEqual(tangent.shape, (0, 3))

    def test_reference_planes_share_mpr_rotation_and_reuse_vtk_actors(self):
        app = QApplication.instance() or QApplication([])
        widget = VolumeView()
        self.volume.is_hu = False
        try:
            widget.set_volume(self.volume)
            center = _volume_corners(self.volume).mean(axis=0)
            rotation = rotation_matrix(23, -11, 18)
            actors = dict(widget._reference_actors)
            geometries = dict(widget._reference_geometry)
            widget.set_reference_planes(center, rotation)
            for name, (axis, _) in render3d._REFERENCE_PLANES.items():
                surface, outline = widget._reference_geometry[name]
                points = vtk_to_numpy(surface.GetPoints().GetData())
                np.testing.assert_allclose((points - center) @ rotation[:, axis], 0, atol=1e-8)
                self.assertEqual(surface.GetNumberOfPolys(), 1)
                self.assertEqual(outline.GetNumberOfLines(), 1)
                self.assertTrue(all(actor.GetVisibility() for actor in actors[name]))
            self.assertTrue(widget._cursor_actor.GetVisibility())
            for delta in np.linspace(0, .5, 10):
                widget.set_reference_planes(center + (delta, 0, 0), rotation)
            for name in render3d._REFERENCE_PLANES:
                self.assertEqual(widget._reference_actors[name], actors[name])
                self.assertEqual(widget._reference_geometry[name], geometries[name])
            revision = widget._reference_revision
            widget.set_reference_planes(center + (.5, 0, 0), rotation)
            self.assertEqual(widget._reference_revision, revision)
        finally:
            widget.cleanup()

    def test_reference_plane_toggles_and_volume_visibility_are_independent(self):
        app = QApplication.instance() or QApplication([])
        widget = VolumeView()
        self.volume.is_hu = False
        try:
            widget.set_volume(self.volume)
            center = _volume_corners(self.volume).mean(axis=0)
            widget.set_reference_planes(center, np.eye(3),
                                        enabled_planes={"Axial": False}, opacity=.21)
            self.assertFalse(widget._reference_actors["Axial"][0].GetVisibility())
            self.assertTrue(widget._reference_actors["Coronal"][0].GetVisibility())
            self.assertAlmostEqual(widget._reference_actors["Coronal"][0].GetProperty().GetOpacity(), .21)
            # Simulate a completed capability check without requiring a GL
            # context in this geometry-only test.
            widget._gpu_supported = True
            widget.set_volume_visibility(False)
            self.assertFalse(widget.actor.GetVisibility())
            self.assertTrue(widget._reference_actors["Sagittal"][0].GetVisibility())
            widget.set_volume_visibility(True)
            self.assertTrue(widget.actor.GetVisibility())
            widget.set_reference_planes(center, np.eye(3), visible=False)
            self.assertTrue(widget.actor.GetVisibility())
            self.assertFalse(widget._cursor_actor.GetVisibility())
            self.assertTrue(all(not actor.GetVisibility()
                                for pair in widget._reference_actors.values() for actor in pair))
        finally:
            widget.cleanup()

    def test_reference_geometry_rejects_invalid_physical_coordinates(self):
        app = QApplication.instance() or QApplication([])
        widget = VolumeView()
        try:
            with self.assertRaisesRegex(ValueError, "finite LPS"):
                widget.set_reference_planes([np.nan, 0, 0], np.eye(3))
            with self.assertRaisesRegex(ValueError, "orthonormal"):
                widget.set_reference_planes(np.zeros(3), np.diag((-1, 1, 1)))
            with self.assertRaisesRegex(ValueError, "opacity"):
                widget.set_reference_planes(np.zeros(3), np.eye(3), opacity=1.1)
            with self.assertRaisesRegex(ValueError, "Unknown reference plane"):
                widget.set_reference_planes(np.zeros(3), np.eye(3), enabled_planes={"Typo": True})
        finally:
            widget.cleanup()

    def test_reference_updates_coalesce_render_requests(self):
        app = QApplication.instance() or QApplication([])
        widget = VolumeView()
        self.volume.is_hu = False
        try:
            widget.set_volume(self.volume)
            widget._render_pending = False
            center = _volume_corners(self.volume).mean(axis=0)
            with patch("asterview.render3d.QTimer.singleShot") as schedule:
                for delta in np.linspace(0, .5, 20):
                    widget.set_reference_planes(center + (delta, 0, 0), np.eye(3))
            schedule.assert_called_once()
            self.assertEqual(schedule.call_args.args[0], 16)
        finally:
            widget.cleanup()

    def test_minimal_annotations_can_reveal_render_details_on_demand(self):
        app = QApplication.instance() or QApplication([])
        widget = VolumeView()
        self.volume.is_hu = False
        try:
            widget.set_volume(self.volume)
            self.assertEqual(widget.annotation.GetText(2), "3D anatomy · Bone")
            self.assertEqual(widget.annotation.GetText(0), "")
            widget.set_annotation_detail(True)
            self.assertIn("Relative scalar values (not HU)", widget.annotation.GetText(2))
            self.assertIn("LPS patient coordinates", widget.annotation.GetText(0))
            widget.set_annotation_detail(False)
            self.assertEqual(widget.annotation.GetText(0), "")
        finally:
            widget.cleanup()

    def test_named_camera_views_use_patient_axes_and_preserve_volume_center(self):
        app = QApplication.instance() or QApplication([])
        widget = VolumeView()
        self.volume.is_hu = False
        try:
            widget.set_volume(self.volume)
            center = _volume_corners(self.volume).mean(axis=0)
            for name, axis in (("Anterior", (0, -1, 0)), ("Posterior", (0, 1, 0)),
                               ("Left", (1, 0, 0)), ("Right", (-1, 0, 0)),
                               ("Superior", (0, 0, 1)), ("Inferior", (0, 0, -1))):
                widget.set_view_orientation(name)
                camera = widget.renderer.GetActiveCamera()
                np.testing.assert_allclose(camera.GetFocalPoint(), center)
                view_direction = np.asarray(camera.GetPosition()) - center
                view_direction /= np.linalg.norm(view_direction)
                np.testing.assert_allclose(view_direction, axis, atol=1e-10)
                self.assertTrue(camera.GetParallelProjection())
                self.assertLess(abs(np.dot(view_direction, camera.GetViewUp())), 1e-10)
            widget.set_view_orientation("Isometric")
            self.assertGreater(widget.renderer.GetActiveCamera().GetParallelScale(), 0)
            with self.assertRaisesRegex(ValueError, "Unknown 3D view"):
                widget.set_view_orientation("Unknown")
        finally:
            widget.cleanup()

    def test_proxy_preserves_oblique_world_extent_and_sample_endpoints(self):
        with patch("asterview.render3d._proxy_shape", return_value=(5, 6, 8)):
            image, shared, proxy = _make_image(self.volume, "Balanced")
        self.assertTrue(proxy)
        self.assertEqual(shared.shape, (5, 6, 8))
        self.assertEqual(shared[0, 0, 0], self.volume.data[0, 0, 0])
        self.assertEqual(shared[-1, -1, -1], self.volume.data[-1, -1, -1])
        world = [0.0, 0.0, 0.0]
        image.TransformContinuousIndexToPhysicalPoint((7.0, 5.0, 4.0), world)
        expected = self.volume.origin + self.volume.direction @ (np.array((14, 10, 8)) * self.volume.spacing)
        np.testing.assert_allclose(world, expected)

    def test_proxy_budgets_stay_bounded_for_large_volumes(self):
        for quality, limit in (("Interactive", 64_000_000), ("Balanced", 128_000_000), ("High", 256_000_000)):
            shape = _proxy_shape((2048, 2048, 2048), quality)
            self.assertLessEqual(np.prod(shape), limit)
            self.assertGreater(min(shape), 1)

    def test_camera_frames_volume_before_gpu_probe_makes_it_visible(self):
        app = QApplication.instance() or QApplication([])
        widget = VolumeView()
        self.volume.is_hu = False
        try:
            widget.set_volume(self.volume)
            self.assertFalse(widget.actor.GetVisibility())
            camera = widget.renderer.GetActiveCamera()
            center = self.volume.origin + self.volume.direction @ (
                (np.asarray(self.volume.data.shape[::-1]) - 1) * self.volume.spacing * .5
            )
            np.testing.assert_allclose(camera.GetFocalPoint(), center)
            self.assertGreater(camera.GetParallelScale(), 4.0)
        finally:
            widget.cleanup()

    def test_proxy_worker_installs_vtk_on_gui_thread(self):
        app = QApplication.instance() or QApplication([])
        widget = VolumeView()
        self.volume.is_hu = False
        vtk_threads = []
        make_image = render3d._image_from_data

        def record_thread(*args):
            vtk_threads.append(QThread.currentThread())
            return make_image(*args)

        try:
            with patch("asterview.render3d._proxy_shape", return_value=(5, 6, 8)), \
                 patch("asterview.render3d._image_from_data", side_effect=record_thread):
                widget.set_volume(self.volume)
                self.assertTrue(widget._preparing)
                self.assertIn("background", widget.info)
                deadline = time.monotonic() + 3
                while widget._worker is not None and time.monotonic() < deadline:
                    app.processEvents()
                    QThread.msleep(1)
            self.assertIsNone(widget._worker)
            self.assertFalse(widget._preparing)
            self.assertEqual(widget._shared_array.shape, (5, 6, 8))
            self.assertEqual(vtk_threads, [app.thread()])
        finally:
            widget.cleanup()

    def test_stale_worker_result_cannot_replace_a_new_series(self):
        app = QApplication.instance() or QApplication([])
        widget = VolumeView()
        self.volume.is_hu = False
        newer = SimpleNamespace(**vars(self.volume))
        newer.data = np.ones((2, 3, 4), dtype=np.float32)
        try:
            with patch("asterview.render3d._proxy_shape", side_effect=lambda shape, quality: (5, 6, 8) if shape == (9, 11, 15) else shape):
                widget.set_volume(self.volume)
                stale_generation = widget._generation
                widget.set_volume(newer)
                # Exercise an already-queued old result as well as cancellation.
                widget._accept_prepared(np.zeros((5, 6, 8), np.float32), np.ones(3), True, stale_generation)
                deadline = time.monotonic() + 3
                while widget._worker is not None and time.monotonic() < deadline:
                    app.processEvents()
                    QThread.msleep(1)
            self.assertIs(widget._volume, newer)
            self.assertIs(widget._shared_array, newer.data)
            self.assertEqual(widget._image.GetDimensions(), (4, 3, 2))
        finally:
            widget.cleanup()

    def test_rapid_quality_changes_keep_only_the_latest_proxy(self):
        app = QApplication.instance() or QApplication([])
        widget = VolumeView()
        self.volume.is_hu = False
        targets = {"Balanced": (5, 6, 8), "Interactive": (3, 4, 5), "High": (7, 8, 10)}
        try:
            with patch("asterview.render3d._proxy_shape", side_effect=lambda shape, quality: targets[quality]):
                widget.set_volume(self.volume)
                widget.set_quality("Interactive")
                widget.set_quality("High")
                deadline = time.monotonic() + 3
                while (widget._worker is not None or widget._pending_proxy is not None) and time.monotonic() < deadline:
                    app.processEvents()
                    QThread.msleep(1)
            self.assertIsNone(widget._worker)
            self.assertIsNone(widget._pending_proxy)
            self.assertEqual(widget._shared_array.shape, targets["High"])
            self.assertEqual(widget._quality, "High")
        finally:
            widget.cleanup()

    def test_small_volume_quality_change_reuses_the_same_vtk_texture_input(self):
        app = QApplication.instance() or QApplication([])
        widget = VolumeView()
        self.volume.is_hu = False
        try:
            widget.set_volume(self.volume)
            image = widget._image
            generation = widget._generation
            widget.set_quality("High")
            self.assertIs(widget._image, image)
            self.assertEqual(widget._generation, generation)
            self.assertFalse(widget.mapper.GetAutoAdjustSampleDistances())
        finally:
            widget.cleanup()

    def test_cleanup_interrupts_and_joins_active_proxy_worker(self):
        app = QApplication.instance() or QApplication([])
        widget = VolumeView()
        self.volume.is_hu = False
        started = threading.Event()

        def cancellable_prepare(volume, quality, cancelled=None):
            started.set()
            while not cancelled():
                time.sleep(.001)
            raise render3d._ProxyCancelled()

        try:
            with patch("asterview.render3d._proxy_shape", return_value=(5, 6, 8)), \
                 patch("asterview.render3d._prepare_data", side_effect=cancellable_prepare):
                widget.set_volume(self.volume)
                worker = widget._worker
                self.assertTrue(started.wait(2))
                self.assertIs(worker.volume, self.volume)
                widget.cleanup()
                self.assertFalse(worker.isRunning())
                self.assertIsNone(widget._worker)
        finally:
            widget.cleanup()

    def test_padding_is_excluded_from_relative_preset_ranges_and_opacity(self):
        app = QApplication.instance() or QApplication([])
        widget = VolumeView()
        self.volume.is_hu = False
        self.volume.data.fill(-900_000)
        self.volume.valid_mask = np.zeros(self.volume.data.shape, dtype=bool)
        self.volume.valid_mask[2:7, 3:8, 4:11] = True
        count = np.count_nonzero(self.volume.valid_mask)
        self.volume.data[self.volume.valid_mask] = np.linspace(10, 100, count, dtype=np.float32)
        self.volume.padding_value = -900_000.0
        self.volume.data_min = 10.0
        self.volume.data_max = 100.0
        try:
            widget.set_volume(self.volume)
            self.assertEqual(widget._range, (10.0, 100.0))
            self.assertGreaterEqual(widget._robust_range[0], 10.0)
            self.assertLessEqual(widget._robust_range[1], 100.0)
            for preset in render3d.PRESETS:
                widget.set_preset(preset)
                for offset in (0, -1_000_000):
                    widget.set_threshold(offset)
                    opacity = widget.property.GetScalarOpacity()
                    self.assertEqual(opacity.GetValue(-900_000), 0.0)
                    self.assertEqual(opacity.GetValue(-100), 0.0)
        finally:
            widget.cleanup()

    def test_proxy_samples_touching_padding_remain_transparent_sentinel(self):
        data = np.full((7, 7, 7), -1000, dtype=np.float32)
        mask = np.zeros(data.shape, dtype=bool)
        mask[2:5, 2:5, 2:5] = True
        data[mask] = 100
        volume = SimpleNamespace(data=data, spacing=(1, 1, 1), valid_mask=mask,
                                 padding_value=-1000, origin=np.zeros(3), direction=np.eye(3))
        with patch("asterview.render3d._proxy_shape", return_value=(6, 6, 6)):
            _, shared, proxy = _make_image(volume, "Balanced")
        self.assertTrue(proxy)
        self.assertEqual(set(np.unique(shared)), {-1000, 100})

    def test_loading_single_image_clears_prior_3d_content_and_is_section_only(self):
        app = QApplication.instance() or QApplication([])
        widget = VolumeView()
        self.volume.is_hu = False
        single = SimpleNamespace(**vars(self.volume))
        single.data = np.ones((1, 11, 15), dtype=np.float32)
        try:
            widget.set_volume(self.volume)
            widget.set_implants([{"center": [0, 0, 0], "axis": [0, 0, 1], "diameter": 4, "length": 10}])
            widget.set_reference_planes(_volume_corners(self.volume).mean(axis=0), np.eye(3))
            self.assertTrue(widget._cursor_actor.GetVisibility())
            widget.actor.SetVisibility(True)
            self.assertIsNotNone(widget._image)
            self.assertEqual(len(widget._implant_actors), 1)
            widget.set_volume(single)
            self.assertIsNone(widget._image)
            self.assertIsNone(widget._shared_array)
            self.assertIsNone(widget._render_volume)
            self.assertIsNone(widget.mapper.GetInput())
            self.assertFalse(widget.actor.GetVisibility())
            self.assertEqual(widget._implant_actors, [])
            self.assertFalse(widget._cursor_actor.GetVisibility())
            self.assertTrue(all(not actor.GetVisibility()
                                for pair in widget._reference_actors.values() for actor in pair))
            self.assertEqual(widget.info, render3d._SPATIAL_VOLUME_REQUIRED)
            widget.set_quality("High")
            widget.set_preset("MIP")
            widget.set_threshold(-200)
            widget.reset_camera()
            self.assertIsNone(widget._image)
            self.assertIsNone(widget._gpu_supported)
            self.assertIn("Single 2D image", widget.annotation.GetText(2))
            with self.assertRaisesRegex(RuntimeError, "requires a spatial volume"):
                widget.capture_image()
        finally:
            widget.cleanup()

    def test_single_image_cancels_active_proxy_without_accepting_stale_content(self):
        app = QApplication.instance() or QApplication([])
        widget = VolumeView()
        self.volume.is_hu = False
        single = SimpleNamespace(**vars(self.volume))
        single.data = np.ones((1, 11, 15), dtype=np.float32)
        try:
            with patch("asterview.render3d._proxy_shape", return_value=(5, 6, 8)):
                widget.set_volume(self.volume)
                widget.set_volume(single)
                deadline = time.monotonic() + 3
                while widget._worker is not None and time.monotonic() < deadline:
                    app.processEvents()
                    QThread.msleep(1)
            self.assertIsNone(widget._worker)
            self.assertIsNone(widget._pending_proxy)
            self.assertIsNone(widget._image)
            self.assertFalse(widget.actor.GetVisibility())
            self.assertEqual(widget.info, render3d._SPATIAL_VOLUME_REQUIRED)
        finally:
            widget.cleanup()


if __name__ == "__main__":
    unittest.main()
