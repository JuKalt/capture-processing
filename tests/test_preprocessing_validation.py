import argparse
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from PIL import Image
from capture_processing.utils.camera import load_source_calibration
from capture_processing.utils.frames import discover_frames
from capture_processing.utils.preprocessing_validation import validate_options, validate_source_media


class InputContractTests(unittest.TestCase):
    def test_rejects_duplicate_and_nonfinite_calibration(self):
        camera = {'camera_id': 'A', 'intrinsics': {'resolution': [8, 8],
                  'camera_matrix': np.eye(3).tolist()},
                  'extrinsics': {'view_matrix': np.eye(4).tolist()}}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'calibration.json'
            for cameras in ([camera, dict(camera, camera_id='a')],
                            [dict(camera, intrinsics={'resolution': [8,8], 'camera_matrix': [float('nan')]*9})]):
                path.write_text(json.dumps({'cameras': cameras}))
                with self.assertRaises(ValueError):
                    load_source_calibration(path)

    def test_rejects_media_geometry_before_conversion(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            Image.new('RGB', (8,8)).save(root/'image.png')
            Image.new('L', (4,4)).save(root/'mask.png')
            job = (root/'image.png', root/'cam00/image.jpg', root/'mask.png', root/'mask.jpg', None,95)
            for size in (16, 8):
                with self.assertRaisesRegex(ValueError, 'resolution mismatch'):
                    validate_source_media([job], {'A': {'W':size,'H':size}}, {'cam00':'A'})

    def test_duplicate_frames_and_invalid_options(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'frame_1').mkdir()
            (root/'frame_000001').mkdir()
            with self.assertRaisesRegex(ValueError, 'duplicate'):
                discover_frames(root)
        for fields in ({'jpeg_quality':101}, {'workers':-1}, {'resize_long_edge':0}, {'mhr_workers':0}):
            with self.assertRaises(ValueError):
                validate_options(argparse.Namespace(**fields))

    def test_pipeline_rejects_geometry_before_fit_and_scales_valid_inputs(self):
        from tests import test_vci_resume as fixtures
        from capture_processing.core.vci_pipeline import run_dataset_creation
        case = fixtures.VCIResumeTests()
        case.setUp()
        try:
            source = case.args.input_path/'calibration.json'
            data = json.loads(source.read_text())
            data['cameras'][0]['intrinsics']['resolution'] = [16,16]
            source.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError,'Calibration/image resolution mismatch'):
                run_dataset_creation(case.args)
            self.assertEqual(case.fit_count,0)
            data['cameras'][0]['intrinsics']['resolution'] = [8,8]
            source.write_text(json.dumps(data))
            case.args.resize_long_edge = 4
            run_dataset_creation(case.args)
            k = json.loads((case.output/'calibration.json').read_text())['cam00']['K']
            self.assertEqual(k[0],.5)
            with Image.open(case.output/'images/cam00/color_00000000.jpg') as image:
                self.assertEqual(image.size,(4,4))
        finally:
            case.doCleanups()

    def test_resize_preserves_valid_dimensions_and_scales_skew(self):
        from capture_processing.utils.camera import resize_to_long_edge, scale_intrinsics
        self.assertEqual(resize_to_long_edge(1,100,1),(1,1))
        k = np.array([[2.,1.,3.],[0.,2.,3.],[0.,0.,1.]])
        np.testing.assert_allclose(scale_intrinsics(k,8,8,4,4),
                                   [[1.,.5,1.5],[0.,1.,1.5],[0.,0.,1.]])


class DatasetKindOptionTests(unittest.TestCase):
    def parse(self, *extra):
        import contextlib, io
        from capture_processing.cli import parse_args
        with contextlib.redirect_stderr(io.StringIO()) as stderr:
            try:
                return parse_args(['in', 'out', *extra]), None
            except SystemExit:
                return None, stderr.getvalue()

    def test_thuman_rejects_vci_only_options(self):
        for option in (['--resize-long-edge', '512'], ['--workers', '4'], ['--skip-images'],
                       ['--drop-cameras', 'C0001'], ['--resume-validation', 'quick'],
                       ['--calibration', 'other.json'], ['--jpeg-quality', '80'], ['--verbose']):
            args, error = self.parse('--dataset-kind', 'thuman', *option)
            self.assertIsNone(args, option)
            self.assertIn(option[0], error)
            self.assertIn('--dataset-kind vci', error)
        args, error = self.parse('--dataset-kind', 'thuman', '--smplx-source', 'existing', '--device', '0')
        self.assertIsNotNone(args, error)

    def test_vci_defaults_and_existing_smplx_rejection(self):
        args, _ = self.parse()
        self.assertEqual(args.smplx_source, 'mamma')
        self.assertFalse(hasattr(args, 'mask_source'))
        args, error = self.parse('--smplx-source', 'existing')
        self.assertIsNone(args)
        self.assertIn('--dataset-kind thuman', error)
        for removed in (['--smplx-source', 'auto'], ['--mhr-visualize'], ['--mask-source', 'existing']):
            self.assertIsNone(self.parse(*removed)[0], removed)
