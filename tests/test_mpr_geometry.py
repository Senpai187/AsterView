"""Physical-coordinate checks for direct linked-section manipulation."""

import unittest

import numpy as np

from asterview.mpr_geometry import (
    PLANE_COLORS, plane_frame, plane_normal, plane_references, rotate_about_view,
    to_euler, translate_reference,
)
from asterview.reformat import rotation_matrix


class MPRGeometryTests(unittest.TestCase):
    def test_identity_frames_match_clinical_radiological_orientation(self):
        frames = {
            "Axial": ((1, 0, 0), (0, -1, 0), (0, 0, -1)),
            "Coronal": ((1, 0, 0), (0, 0, 1), (0, -1, 0)),
            "Sagittal": ((0, 1, 0), (0, 0, 1), (1, 0, 0)),
        }
        for name, (expected_right, expected_up, expected_normal) in frames.items():
            with self.subTest(plane=name):
                right, up = plane_frame(name, np.eye(3))
                np.testing.assert_allclose(right, expected_right)
                np.testing.assert_allclose(up, expected_up)
                np.testing.assert_allclose(plane_normal(name, np.eye(3)), expected_normal)
                np.testing.assert_allclose(np.cross(right, up), expected_normal)

    def test_rotating_references_keeps_viewed_image_frame_fixed(self):
        original = rotation_matrix(23.2, -37.4, 17.8)
        for name in PLANE_COLORS:
            before = plane_frame(name, original)
            normal = plane_normal(name, original)
            for angle in (-178, -45, 17, 90, 181):
                with self.subTest(plane=name, angle=angle):
                    corrected = rotate_about_view(original, name, angle)
                    np.testing.assert_allclose(plane_normal(name, corrected), normal, atol=1e-12)
                    np.testing.assert_allclose(plane_frame(name, corrected), before, atol=1e-12)
                    np.testing.assert_allclose(corrected.T @ corrected, np.eye(3), atol=1e-12)
                    self.assertAlmostEqual(np.linalg.det(corrected), 1.0)
                    for other in PLANE_COLORS:
                        if other == name:
                            continue
                        old_normal = plane_normal(other, original)
                        new_normal = plane_normal(other, corrected)
                        expected_dot = np.cos(np.deg2rad(angle))
                        self.assertAlmostEqual(float(old_normal @ new_normal), expected_dot)
                        self.assertAlmostEqual(float(normal @ new_normal), 0)

    def test_reference_directions_lie_in_both_planes_and_have_correct_names(self):
        rotation = rotation_matrix(-31.3, 67.4, -126.2)
        for name in PLANE_COLORS:
            references = plane_references(name, rotation)
            self.assertEqual({item["name"] for item in references}, set(PLANE_COLORS) - {name})
            for item in references:
                with self.subTest(view=name, target=item["name"]):
                    direction = item["direction"]
                    self.assertAlmostEqual(np.linalg.norm(direction), 1)
                    self.assertAlmostEqual(float(direction @ plane_normal(name, rotation)), 0)
                    self.assertAlmostEqual(float(direction @ item["normal"]), 0)
                    np.testing.assert_allclose(item["normal"], plane_normal(item["name"], rotation))
                    self.assertEqual(item["color"], PLANE_COLORS[item["name"]])

    def test_line_drag_moves_only_target_section_and_stays_inside_view_plane(self):
        rotation = rotation_matrix(-23.7, 31.5, 62.4)
        center = np.array((102.7, -83.5, 312.3))
        delta = np.array((3.4, 5.2, -7.8))
        for view in PLANE_COLORS:
            for target in set(PLANE_COLORS) - {view}:
                with self.subTest(view=view, target=target):
                    moved = translate_reference(center, delta, view, target, rotation)
                    target_normal = plane_normal(target, rotation)
                    expected = target_normal * float(delta @ target_normal)
                    np.testing.assert_allclose(moved - center, expected, atol=1e-12)
                    self.assertAlmostEqual(float((moved - center) @ plane_normal(view, rotation)), 0)
                    other = (set(PLANE_COLORS) - {view, target}).pop()
                    self.assertAlmostEqual(float((moved - center) @ plane_normal(other, rotation)), 0)

    def test_axial_reference_line_drag_has_correct_lps_distance(self):
        center = np.array((20.0, -50.0, 100.0))
        delta = np.array((3.0, 7.0, 13.0))
        np.testing.assert_allclose(
            translate_reference(center, delta, "Axial", "Sagittal", np.eye(3)),
            center + (3, 0, 0),
        )
        np.testing.assert_allclose(
            translate_reference(center, delta, "Axial", "Coronal", np.eye(3)),
            center + (0, 7, 0),
        )
        np.testing.assert_allclose(
            translate_reference(center, delta, "Axial", None, np.eye(3)),
            center + (3, 7, 0),
        )

    def test_center_drag_projects_free_motion_into_oblique_view(self):
        center = np.array((2.0, -3.0, 5.0))
        rotation = rotation_matrix(18.3, -21.2, 30.9)
        delta = np.array((-13.0, 9.0, 12.0))
        for name in PLANE_COLORS:
            normal = plane_normal(name, rotation)
            result = translate_reference(center, delta, name, None, rotation)
            expected = delta - float(delta @ normal) * normal
            np.testing.assert_allclose(result - center, expected)
            self.assertAlmostEqual(float((result - center) @ normal), 0)
            # Pure through-plane motion cannot move this viewport's center.
            np.testing.assert_allclose(
                translate_reference(center, normal * 20, name, None, rotation), center,
                atol=1e-12,
            )

    def test_frame_fallback_is_finite_and_stable_at_anatomical_up_poles(self):
        rotations = {
            "Axial": rotation_matrix(90, 0, 0),
            "Coronal": rotation_matrix(90, 0, 0),
            "Sagittal": rotation_matrix(0, -90, 0),
        }
        for name, rotation in rotations.items():
            right, up = plane_frame(name, rotation)
            self.assertTrue(np.all(np.isfinite((right, up))))
            np.testing.assert_allclose(np.linalg.norm((right, up), axis=1), 1)
            np.testing.assert_allclose(np.cross(right, up), plane_normal(name, rotation), atol=1e-12)
            perturbed = rotation_matrix(.000001, -.000001, .000001) @ rotation
            np.testing.assert_allclose(plane_frame(name, perturbed), (right, up), atol=1e-6)

    def test_euler_round_trip_covers_arbitrary_obliquity_and_gimbal_lock(self):
        rng = np.random.default_rng(8716)
        angles = list(rng.uniform(-360, 360, size=(300, 3)))
        angles.extend((17.3, pitch, -111.8)
                      for pitch in (90, -90, 89.999999, -89.999999, 270, -270))
        for input_angles in angles:
            with self.subTest(angles=input_angles):
                original = rotation_matrix(*input_angles)
                recovered = to_euler(original)
                np.testing.assert_allclose(rotation_matrix(*recovered), original, atol=1e-10)
                self.assertLessEqual(abs(recovered[1]), 90.0 + 1e-10)

    def test_repeated_direct_rotations_remain_proper_and_recoverable_as_euler(self):
        rotation = rotation_matrix(-24, 42, 13)
        for index in range(180):
            name = tuple(PLANE_COLORS)[index % 3]
            rotation = rotate_about_view(rotation, name, 1.3)
        np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-12)
        self.assertAlmostEqual(np.linalg.det(rotation), 1.0)
        np.testing.assert_allclose(rotation_matrix(*to_euler(rotation)), rotation, atol=1e-12)

    def test_invalid_frames_and_drag_inputs_fail_explicitly(self):
        for bad in (np.zeros((3, 3)), np.diag((-1, 1, 1)), np.full((3, 3), np.nan)):
            with self.assertRaisesRegex(ValueError, "proper orthonormal"):
                plane_frame("Axial", bad)
        with self.assertRaisesRegex(ValueError, "Plane must"):
            plane_normal("Unknown", np.eye(3))
        with self.assertRaisesRegex(ValueError, "finite"):
            rotate_about_view(np.eye(3), "Axial", np.nan)
        with self.assertRaisesRegex(ValueError, "differ"):
            translate_reference(np.zeros(3), np.ones(3), "Axial", "Axial", np.eye(3))
        with self.assertRaisesRegex(ValueError, "Translation"):
            translate_reference(np.zeros(3), [0, np.inf, 0], "Axial", None, np.eye(3))


if __name__ == "__main__":
    unittest.main()
