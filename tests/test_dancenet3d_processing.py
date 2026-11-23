from __future__ import annotations

import argparse
import io
import json
import subprocess
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from capture_processing.core.dancenet3d_pipeline import (
    extract_dancenet3d_session,
    incomplete_extraction_cameras,
    process_dancenet3d,
    read_colmap_images,
)
from capture_processing.utils.mamma import _enabled_mamma_steps, prepare_mamma_runtime_preset


class DanceNet3DProcessingTests(unittest.TestCase):
    def test_runtime_preset_uses_supplied_masks_and_skips_segmentation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            preset = root / "full.yaml"
            masks = root / "provided masks"
            masks.mkdir()
            preset.write_text(
                "ma_cap:\n"
                "  enabled: true\n"
                "ma_masks:\n"
                "  enabled: true\n"
                "ma_2d:\n"
                "  enabled: true\n"
                "  flags: []\n"
                "ma_3d:\n"
                "  enabled: true\n"
                "ma_vis:\n"
                "  enabled: true\n"
            )

            runtime = prepare_mamma_runtime_preset(
                preset, root / "work", masks_root=masks
            )
            runtime_text = runtime.read_text()

            self.assertIn("ma_masks:\n  enabled: false", runtime_text)
            self.assertIn(f"  ma_masks_dir: {masks.resolve()}", runtime_text)
            self.assertEqual(
                _enabled_mamma_steps(runtime),
                ("ma_cap", "ma_2d", "ma_3d", "ma_vis"),
            )

    def test_colmap_images_accept_empty_and_populated_observation_lines(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            images = Path(tmp) / "images.txt"
            images.write_text(
                "1 1 0 0 0 0 0 0 1 0028.png\n"
                "\n"
                "2 1 0 0 0 1 2 3 2 0103.png\n"
                "12.5 20.0 -1 30.0 40.0 7 50.0 60.0 -1 70.0 80.0 9\n"
                "3 1 0 0 0 4 5 6 3 Dance_0244.png\n"
                "\n"
            )

            poses = read_colmap_images(images, ["0028", "0103", "0244"])

            self.assertEqual(list(poses), ["0028", "0103", "0244"])
            self.assertEqual(poses["0103"]["T"], [1.0, 2.0, 3.0])
            self.assertEqual(poses["0244"]["camera_id"], 3)

    def test_resume_detects_only_incomplete_cameras_and_masks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            extracted = Path(tmp)
            sequence_data = {
                "cameras": ["0028", "0103"],
                "frame_ids": ["0000001", "0000002"],
                "image_extension": ".png",
                "content_type": "images_no_lut",
                "has_masks": True,
            }
            sequence_root = extracted / "Dance" / "images_and_masks"
            for frame_id in sequence_data["frame_ids"]:
                for camera in sequence_data["cameras"]:
                    image = (
                        sequence_root
                        / frame_id
                        / "images_no_lut"
                        / f"{camera}.png"
                    )
                    mask = sequence_root / frame_id / "masks" / f"{camera}.png"
                    image.parent.mkdir(parents=True, exist_ok=True)
                    mask.parent.mkdir(parents=True, exist_ok=True)
                    image.write_bytes(b"image")
                    mask.write_bytes(b"mask")
            (sequence_root / "0000002" / "images_no_lut" / "0103.png").unlink()

            cameras, masks_complete = incomplete_extraction_cameras(
                extracted_session_root=extracted,
                sequence="Dance",
                sequence_data=sequence_data,
                selected_cameras=None,
            )
            self.assertEqual(cameras, ["0103"])
            self.assertTrue(masks_complete)

            (sequence_root / "0000002" / "images_no_lut" / "0103.png").write_bytes(
                b"image"
            )
            (sequence_root / "0000001" / "masks" / "0028.png").unlink()
            cameras, masks_complete = incomplete_extraction_cameras(
                extracted_session_root=extracted,
                sequence="Dance",
                sequence_data=sequence_data,
                selected_cameras=None,
            )
            self.assertEqual(cameras, [])
            self.assertFalse(masks_complete)

    def test_processing_reuses_extraction_and_prepares_each_sequence_immediately(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "DanceNet3D"
            extracted = dataset / "extracted" / "s4"
            output = root / "processed"
            (dataset / "s4").mkdir(parents=True)
            sequence_data = {
                "cameras": ["0028"],
                "frame_ids": ["0000001"],
                "image_extension": ".png",
                "has_masks": False,
            }
            (dataset / "s4" / "manifest.json").write_text(
                json.dumps(
                    {
                        "content_type": "images_no_lut",
                        "sequences": {
                            "Complete": sequence_data,
                            "Missing": sequence_data,
                        },
                    }
                )
            )
            complete_image = (
                extracted
                / "Complete"
                / "images_and_masks"
                / "0000001"
                / "images_no_lut"
                / "0028.png"
            )
            complete_image.parent.mkdir(parents=True)
            complete_image.write_bytes(b"image")
            events = []

            def fake_extract(**kwargs) -> None:
                sequence = kwargs["sequences"][0]
                events.append(("extract", sequence, kwargs["cameras"]))
                image = (
                    extracted
                    / sequence
                    / "images_and_masks"
                    / "0000001"
                    / "images_no_lut"
                    / "0028.png"
                )
                image.parent.mkdir(parents=True)
                image.write_bytes(b"image")

            def fake_prepare(**kwargs) -> Path:
                sequence = kwargs["sequence"]
                events.append(("prepare", sequence))
                subject = output / f"s4_{sequence}"
                subject.mkdir(parents=True)
                return subject

            args = argparse.Namespace(
                dataset_root=dataset,
                output_root=output,
                session="s4",
                sequences=None,
                cameras=None,
                extracted_root=None,
                skip_extraction=False,
                skip_smplx=True,
                max_frames=None,
                parallel=1,
            )
            with patch(
                "capture_processing.core.dancenet3d_pipeline.extract_dancenet3d_session",
                side_effect=fake_extract,
            ), patch(
                "capture_processing.core.dancenet3d_pipeline.prepare_dancenet3d_subject",
                side_effect=fake_prepare,
            ):
                subjects = process_dancenet3d(args)

            self.assertEqual(
                events,
                [
                    ("prepare", "Complete"),
                    ("extract", "Missing", ["0028"]),
                    ("prepare", "Missing"),
                ],
            )
            self.assertEqual(
                subjects, [output / "s4_Complete", output / "s4_Missing"]
            )

    def test_extraction_reports_missing_ffmpeg_before_launching(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dataset = Path(tmp)
            (dataset / "video_to_images.py").write_text("")
            (dataset / "s4").mkdir()
            (dataset / "s4" / "manifest.json").write_text("{}")

            with patch(
                "capture_processing.core.dancenet3d_pipeline.shutil.which", return_value=None
            ), self.assertRaisesRegex(SystemExit, "conda install.*ffmpeg"):
                extract_dancenet3d_session(
                    dataset_root=dataset,
                    extracted_session_root=dataset / "extracted",
                    session="s4",
                    sequences=["Dance"],
                    cameras=None,
                    parallel=1,
                )

    def test_extraction_prints_periodic_image_progress(self) -> None:
        class FakeProcess:
            waits = 0

            def wait(self, timeout: int) -> int:
                self.waits += 1
                if self.waits == 1:
                    raise subprocess.TimeoutExpired("extractor", timeout)
                return 0

        with tempfile.TemporaryDirectory() as tmp:
            dataset = Path(tmp)
            extracted = dataset / "extracted" / "s4"
            (dataset / "video_to_images.py").write_text("")
            (dataset / "s4").mkdir()
            (dataset / "s4" / "manifest.json").write_text(
                json.dumps(
                    {
                        "content_type": "images_no_lut",
                        "sequences": {
                            "Dance": {
                                "frame_ids": ["0000001", "0000002"],
                                "cameras": ["0028", "0103"],
                            }
                        },
                    }
                )
            )
            image = (
                extracted
                / "Dance"
                / "images_and_masks"
                / "0000001"
                / "images_no_lut"
                / "0028.png"
            )
            image.parent.mkdir(parents=True)
            image.write_bytes(b"image")
            output = io.StringIO()

            with patch(
                "capture_processing.core.dancenet3d_pipeline.shutil.which", return_value="ffmpeg"
            ), patch(
                "capture_processing.core.dancenet3d_pipeline.importlib.util.find_spec",
                return_value=object(),
            ), patch(
                "capture_processing.core.dancenet3d_pipeline.subprocess.Popen",
                return_value=FakeProcess(),
            ), patch(
                "capture_processing.core.dancenet3d_pipeline.time.monotonic",
                side_effect=[0, 12],
            ), redirect_stdout(output):
                extract_dancenet3d_session(
                    dataset_root=dataset,
                    extracted_session_root=extracted,
                    session="s4",
                    sequences=["Dance"],
                    cameras=None,
                    parallel=1,
                )

            self.assertIn("1/4 images (25.0%)", output.getvalue())

    def test_multiview_colmap_data_is_canonicalized_and_sent_to_mamma(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "DanceNet3D"
            output = root / "processed"
            extracted = root / "extracted"
            session = dataset / "s4"
            sequence = session / "Dance"
            colmap = sequence / "colmap"
            colmap.mkdir(parents=True)
            manifest = {
                "content_type": "images_no_lut",
                "sequences": {
                    "Dance": {
                        "cameras": ["0028", "0103"],
                        "frame_ids": ["0000001", "0000002"],
                        "image_extension": ".png",
                        "has_colmap": True,
                        "has_masks": True,
                    }
                },
            }
            session.mkdir(parents=True, exist_ok=True)
            (session / "manifest.json").write_text(json.dumps(manifest))
            (colmap / "cameras.txt").write_text(
                "1 PINHOLE 800 1280 1000 1001 400 640\n"
                "2 PINHOLE 800 1280 900 901 401 641\n"
            )
            (colmap / "images.txt").write_text(
                "1 1 0 0 0 0 0 0 1 Dance_0028.png\n"
                "0 0 -1\n"
                "2 1 0 0 0 1 2 3 2 Dance_0103.png\n"
                "0 0 -1\n"
            )
            extracted_sequence = (
                extracted / "s4" / "Dance" / "images_and_masks"
            )
            for frame_id in ("0000001", "0000002"):
                for camera in ("0028", "0103"):
                    image = (
                        extracted_sequence
                        / frame_id
                        / "images_no_lut"
                        / f"{camera}.png"
                    )
                    mask = extracted_sequence / frame_id / "masks" / f"{camera}.png"
                    image.parent.mkdir(parents=True, exist_ok=True)
                    mask.parent.mkdir(parents=True, exist_ok=True)
                    image.write_bytes(b"image")
                    mask.write_bytes(b"mask")

            calls = []

            def fake_mamma(subject_root: Path, **kwargs) -> None:
                calls.append((subject_root, kwargs))
                (subject_root / "smpl_params.npz").write_bytes(b"smplx")

            args = argparse.Namespace(
                dataset_root=dataset,
                output_root=output,
                session="s4",
                sequences=["Dance"],
                cameras=None,
                extracted_root=extracted,
                skip_extraction=True,
                skip_smplx=False,
                max_frames=1,
                parallel=1,
                mamma_repo_path=None,
                mamma_preset=None,
                mamma_out_tag="dancenet3d",
                mamma_python="mamma-python",
                mamma_force=False,
                mamma_workers=2,
                device="cuda:1",
            )
            subjects = process_dancenet3d(args, smplx_runner=fake_mamma)

            self.assertEqual(subjects, [output / "s4_Dance"])
            subject = subjects[0]
            self.assertTrue(
                (subject / "images" / "cam00" / "00000000.png").is_symlink()
            )
            self.assertTrue(
                (subject / "masks" / "cam01" / "00000000.png").is_symlink()
            )
            self.assertTrue(
                (
                    subject
                    / "mamma_input_masks"
                    / "images"
                    / "cam00"
                    / "masks"
                    / "mask_0000_01.png"
                ).is_symlink()
            )
            self.assertFalse(
                (subject / "images" / "cam00" / "00000001.png").exists()
            )
            mapping = json.loads((subject / "camera_id_mapping.json").read_text())
            self.assertEqual(mapping, {"cam00": "0028", "cam01": "0103"})
            frame_mapping = json.loads((subject / "frame_id_mapping.json").read_text())
            self.assertEqual(frame_mapping, {"00000000": "0000001"})
            calibration = json.loads((subject / "calibration.json").read_text())
            self.assertEqual(calibration["cam00"]["K"][0], 1000.0)
            self.assertEqual(calibration["cam01"]["T"], [1.0, 2.0, 3.0])
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0][0], subject)
            self.assertEqual(calls[0][1]["smplx_source"], "mamma")
            self.assertEqual(calls[0][1]["device"], "cuda:1")
            self.assertTrue((subject / "smpl_params.npz").is_file())


if __name__ == "__main__":
    unittest.main()
