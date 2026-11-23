"""Geometry, provenance, and resume checks for DNA refitting preparation."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

from capture_processing.core.dna_pipeline import prepare_subject, process_dna


class SyntheticReader:
    instances = []
    K = np.array([[90., 0., 59.], [0., 85., 39.], [0., 0., 1.]])
    D = np.array([-.12, .03, .001, -.002, 0.])
    RT = np.eye(4)
    RT[:3, 3] = [1., 2., 3.]

    def __init__(self, path):
        self.released = False
        self.instances.append(self)

    def get_Camera_5mp_info(self):
        return {"num_frame": 10}

    def get_Calibration_all(self):
        return {"12": {"K": self.K, "D": self.D, "RT": self.RT}}

    def get_img(self, *args, **kwargs):
        return np.full((80, 120, 3), kwargs["Frame_id"] * 20, dtype=np.uint8)

    def get_mask(self, *args, **kwargs):
        mask = np.zeros((80, 120), dtype=np.uint8)
        mask[20:60, 30:90] = 255
        return mask

    def release(self):
        self.released = True


@unittest.skipIf(cv2 is None, "DNA export tests require OpenCV")
class DNAProcessingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        for name in ("rgb.smc", "annotations.smc"):
            (self.root / name).touch()
        self.subject = {"images": self.root / "rgb.smc", "smplx": self.root / "annotations.smc"}
        SyntheticReader.instances = []

    def prepare(self, frames=None):
        return prepare_subject(self.subject, self.root / "output", [12], frames or [2, 5], 60, SyntheticReader)

    def test_projection_matches_undistorted_pixels_and_retains_frame_correspondence(self):
        output = self.prepare()
        calibration = json.loads((output / "calibration.json").read_text())["cam12"]
        K = np.asarray(calibration["K"]).reshape(3, 3)
        points = np.array([[1.1, 2.2, 6.], [0.7, 1.9, 5.]])
        R = np.asarray(calibration["R"]).reshape(3, 3)
        T = np.asarray(calibration["T"])
        camera_points = points @ R.T + T
        distorted, _ = cv2.projectPoints(camera_points, np.zeros(3), np.zeros(3), SyntheticReader.K, SyntheticReader.D)
        expected = cv2.undistortPoints(distorted, SyntheticReader.K, SyntheticReader.D, P=K).reshape(-1, 2)
        projected = camera_points @ K.T
        np.testing.assert_allclose(projected[:, :2] / projected[:, 2:], expected, atol=1e-5)
        self.assertEqual(calibration["distCoeff"], [0.] * 5)
        self.assertEqual(json.loads((output / "frame_id_mapping.json").read_text()), {"0": 2, "1": 5})
        for local_id, source_id in enumerate([2, 5]):
            image = cv2.imread(str(output / "images/cam12" / f"{local_id:08d}.jpg"))
            self.assertEqual(image.shape, (40, 60, 3))
            self.assertAlmostEqual(float(image[20, 30].mean()), source_id * 20, delta=2)
            mask = cv2.imread(str(output / "masks/cam12" / f"{local_id:08d}.png"), 0)
            self.assertEqual(set(np.unique(mask)), {0, 255})
            self.assertTrue((output / "mamma_input_masks/images/cam12/masks" / f"mask_{local_id:04d}_01.png").is_file())
        self.assertFalse((output / "smpl_params.npz").exists())
        self.assertTrue(all(r.released for r in SyntheticReader.instances))

    def test_resume_reuses_media_and_rejects_selection_changes(self):
        output = self.prepare()
        image = output / "images/cam12/00000000.jpg"
        before = image.stat().st_mtime_ns
        self.prepare()
        self.assertEqual(image.stat().st_mtime_ns, before)
        self.assertEqual(len(SyntheticReader.instances), 2)
        with self.assertRaisesRegex(SystemExit, "selection or source changed"):
            self.prepare([0, 1])

    def test_interrupted_export_resumes_and_invalid_frames_close_readers(self):
        with self.assertRaisesRegex(SystemExit, "capacity"):
            self.prepare([20])
        self.assertTrue(all(r.released for r in SyntheticReader.instances))
        marker = self.root / "output/.dna_preparation.json"
        self.assertFalse(json.loads(marker.read_text())["complete"])
        # A matching interrupted run can be retried; source selection stays fixed.
        with self.assertRaisesRegex(SystemExit, "capacity"):
            self.prepare([20])

    def test_partial_export_can_resume_and_completed_missing_masks_are_rejected(self):
        output = self.prepare()
        marker = output / ".dna_preparation.json"
        state = json.loads(marker.read_text())
        state["complete"] = False
        marker.write_text(json.dumps(state))
        mask = output / "masks/cam12/00000001.png"
        mask.unlink()
        self.prepare()
        self.assertTrue(mask.is_file())
        self.assertTrue(json.loads(marker.read_text())["complete"])
        mask.unlink()
        with self.assertRaisesRegex(SystemExit, "missing files"):
            self.prepare()

    def test_unmanaged_outputs_are_preserved(self):
        output = self.root / "output"
        output.mkdir()
        (output / "keep.txt").write_text("existing work")
        with self.assertRaisesRegex(SystemExit, "unmanaged"):
            self.prepare()
        self.assertEqual((output / "keep.txt").read_text(), "existing work")

    def test_short_temporal_fit_is_rejected_before_export(self):
        try:
            import h5py
        except ImportError:
            self.skipTest("SMC metadata check requires h5py")
        root = self.root / "Part1"
        for name in ("main", "annotations", "kinect"):
            (root / f"dna_rendering_part1_{name}").mkdir(parents=True)
        with h5py.File(root / "dna_rendering_part1_main/sequence.smc", "w") as f:
            f.create_group("Camera_5mp").attrs["num_frame"] = 2
        (root / "dna_rendering_part1_annotations/sequence_annots.smc").touch()
        args = SimpleNamespace(dataset_root=root, output_root=self.root / "fitted",
                               subjects=["sequence"], frame_start=0, frame_stop=2,
                               frame_step=1, prepare_only=False, mamma_preset=None)
        with self.assertRaisesRegex(SystemExit, "at least three"):
            process_dna(args)
        self.assertFalse(args.output_root.exists())


if __name__ == "__main__":
    unittest.main()
