import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from capture_processing.reprocess import main


class ReprocessingTests(unittest.TestCase):
    def test_external_reader_materializes_calibrated_pair_and_retains_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subject = root / "source" / "actor"
            for kind, mode, value in (("images", "RGB", (255, 0, 0)), ("masks", "L", 1)):
                folder = subject / kind / "cam07"
                folder.mkdir(parents=True)
                Image.new(mode, (8, 6), value).save(folder / "00000018.png")
            (subject / "calibration.json").write_text(json.dumps({"cam07": {
                "K": [10, 0, 4, 0, 10, 3, 0, 0, 1], "R": [1, 0, 0, 0, 1, 0, 0, 0, 1], "T": [0, 0, 0]
            }}))
            before = {p: p.stat().st_mtime_ns for p in subject.rglob("*") if p.is_file()}

            def inspect(argv):
                stage = Path(argv[0]) / "actor"
                frame = stage / "frame_00000000"
                with Image.open(frame / "mask/mask_C0000.png") as mask:
                    self.assertEqual(mask.getextrema(), (255, 255))
                source = json.loads((stage / "calibration_dome.json").read_text())["meta"]["imported"]
                self.assertEqual(source["frame_ids"], [18])
                self.assertEqual(source["camera_ids"], {"C0000": "cam07"})
                self.assertIn("--no-semantic-parsing", argv)

            with patch("capture_processing.reprocess.process", side_effect=inspect) as process:
                self.assertEqual(main(["thuman", str(subject.parent), str(root / "output"),
                                      "--subject", "actor", "--cameras", "cam07", "--frames", "18",
                                      "--", "--no-semantic-parsing"]), 0)
                process.assert_called_once()
            self.assertEqual(before, {p: p.stat().st_mtime_ns for p in before})
