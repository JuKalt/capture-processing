"""Regression checks for bounded preprocessing and batched landmark exports."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from capture_processing.utils.mamma import prepare_mamma_runtime_preset
from capture_processing.utils.mamma_landmarks import ordered_prefetch, process_masked_camera


class MammaLandmarkTests(unittest.TestCase):
    def test_confirmed_empty_camera_exports_zero_visibility_without_inference(self):
        import numpy as np
        class EmptySource:
            cam_name = 'cam13'
            def __len__(self):
                return 3
            def read_rgb(self, index):
                raise AssertionError('Empty view must not read images')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            masks = root / 'masks/cam13/masks'
            masks.mkdir(parents=True)
            other = root / 'masks/cam12/masks'
            other.mkdir(parents=True)
            (other / 'mask_0000_02.png').touch()
            np.save(masks.parent / 'masks.npy', {0: {'mask': [], 'frame': []}})
            process_masked_camera(EmptySource(), 'cpu', None, SimpleNamespace(num_joints=2),
                                  root / 'out', root / 'masks', None)
            with np.load(root / 'out/cam13.npz') as result:
                self.assertEqual(result['landmarks'].shape, (3, 2, 2, 3))
                for key in ['landmarks', 'visibilities', 'contacts', 'floor_contacts']:
                    self.assertFalse(result[key].any())

    def test_skipped_camera_without_mask_directory_exports_zero_visibility(self):
        import numpy as np
        class EmptySource:
            cam_name = 'cam03'
            def __len__(self):
                return 3
            def read_rgb(self, index):
                raise AssertionError('Empty view must not read images')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sequence = root / 'masks'
            (sequence / 'cam03').mkdir(parents=True)
            other = sequence / 'cam00/masks'
            other.mkdir(parents=True)
            (other / 'mask_0000_01.png').touch()
            (sequence / 'DONE').touch()
            (sequence / 'run.log').write_text(
                "[ma_masks] ERROR | No valid detections found for any ID in 'cam03'. "
                'Skipping propagation. Check matching.start_frame/samples and detection thresholds.\n')
            process_masked_camera(EmptySource(), 'cpu', None, SimpleNamespace(num_joints=2),
                                  root / 'out', sequence, None)
            self.assertFalse((sequence / 'cam03/masks').exists())
            with np.load(root / 'out/cam03.npz') as result:
                self.assertEqual(result['landmarks'].shape, (3, 1, 2, 3))
                for key in ['landmarks', 'visibilities', 'contacts', 'floor_contacts']:
                    self.assertFalse(result[key].any())

    def test_missing_masks_require_explicit_completed_empty_view(self):
        import numpy as np
        for case in ['no_log', 'unfinished', 'other_camera', 'prompt_error',
                     'no_other_bodies', 'nonempty_cache']:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                sequence = root / 'masks'
                (sequence / 'cam03').mkdir(parents=True)
                other = sequence / 'cam00/masks'
                other.mkdir(parents=True)
                if case != 'no_other_bodies':
                    (other / 'mask_0000_01.png').touch()
                if case != 'unfinished':
                    (sequence / 'DONE').touch()
                camera = 'cam04' if case == 'other_camera' else 'cam03'
                diagnostic = (f"No valid detections found for any ID in '{camera}'. "
                              'Skipping propagation. Check matching.start_frame/samples and detection thresholds.')
                if case == 'prompt_error':
                    diagnostic = 'Failed while injecting anchor prompts into SAM2 state: error. Skipping propagation.'
                if case != 'no_log':
                    (sequence / 'run.log').write_text(diagnostic + '\n')
                if case == 'nonempty_cache':
                    np.save(sequence / 'cam03/masks.npy', {0: {'mask': [np.ones((2, 2))]}})
                with self.assertRaisesRegex(ValueError, 'No mask files'):
                    process_masked_camera(SimpleNamespace(cam_name='cam03'), 'cpu', None, None,
                                          root / 'out', sequence, None)
                self.assertFalse((root / 'out/cam03.npz').exists())

    def test_missing_pngs_with_nonempty_segmentation_still_fail(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            masks = root / 'masks/cam13/masks'
            masks.mkdir(parents=True)
            np.save(masks.parent / 'masks.npy', {0: {'mask': [np.ones((2, 2))]}})
            with self.assertRaisesRegex(ValueError, 'No mask files'):
                process_masked_camera(SimpleNamespace(cam_name='cam13'), 'cpu', None, None,
                                      root / 'out', root / 'masks', None)

    def test_prefetch_preserves_order_and_propagates_errors(self):
        self.assertEqual(list(ordered_prefetch(lambda i: i * 2, range(11), 4, 3)),
                         list(range(0, 22, 2)))
        def fail(index):
            if index == 3:
                raise ValueError('bad frame')
            return index
        with self.assertRaisesRegex(ValueError, 'bad frame'):
            list(ordered_prefetch(fail, range(8), 2, 4))

    def test_runtime_preset_routes_standard_script_and_overrides_flags(self):
        import yaml
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            preset = root / 'preset.yaml'
            preset.write_text('ma_2d:\n  enabled: true\n  script: run_ma_2d.py\n'
                              '  flags:\n  - --workers=1\n  - --save_cam_output\n'
                              '  undistort: false\nma_3d:\n  enabled: true\n')
            runtime = prepare_mamma_runtime_preset(preset, root, 6, batch_size=3)
            config = yaml.safe_load(runtime.read_text())
            self.assertTrue(config['ma_2d']['script'].endswith('/mamma_landmarks.py'))
            self.assertEqual(config['ma_2d']['flags'][-3:],
                             ['--no-save_cam_output', '--workers 6', '--batch-size 3'])
            self.assertFalse(config['ma_2d']['undistort'])
            self.assertEqual(config['ma_3d'], {'enabled': True})

    def test_batch_preserves_geometry_missing_bodies_and_partial_batch(self):
        try:
            import cv2
            import numpy as np
            import torch
        except ImportError as exc:
            self.skipTest(f"MAMMA runtime dependency unavailable: {exc}")
        cfg = SimpleNamespace(num_joints=2, data={'train': {'normalize_plus_min_one': True}})
        class Source:
            cam_name = 'cam00'
            def __len__(self):
                return 5
            def read_rgb(self, index):
                return np.full((8, 8, 3), index, dtype=np.uint8)
        class Dataset:
            def __init__(self, cfg, image, mask, boxes):
                self.index = int(image[0, 0, 0])
            def __getitem__(self, index):
                return {'img': np.full((3, 2, 2), self.index, np.float32),
                        'mask': np.ones((1, 2, 2), np.float32),
                        'box_center': np.array([20., 30.]), 'box_size': np.array([10., 12.])}
        class Model:
            def __init__(self):
                self.batches = []
            def __call__(self, images, masks):
                self.batches.append(len(images))
                joints = torch.zeros((len(images), 2, 3))
                joints[:, :, 0] = images[:, 0, 0, 0, None] / 10
                scores = images[:, 0, 0, 0, None, None].expand(-1, 2, 1)
                return {'joints2d': joints, 'visibility': scores, 'contact': scores,
                        'floor_contact': scores}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            masks = root / 'masks' / 'cam00' / 'masks'
            masks.mkdir(parents=True)
            # Body 2 has no masks; keep its zero slots between bodies 1 and 3.
            for index in [0, 2, 3, 4]:
                for body in [1, 3]:
                    cv2.imwrite(str(masks / f'mask_{index:04d}_{body:02d}.png'),
                                np.full((8, 8), 255, np.uint8))
            with patch('cv2.imread', wraps=cv2.imread):
                model = Model()
                process_masked_camera(Source(), 'cpu', model, cfg, root / 'batch',
                                      root / 'masks', Dataset, workers=3, batch_size=3)
            serial = Model()
            process_masked_camera(Source(), 'cpu', serial, cfg, root / 'serial',
                                  root / 'masks', Dataset, workers=1, batch_size=1)
            self.assertEqual(model.batches, [3, 1, 3, 1])
            with np.load(root / 'batch' / 'cam00.npz') as batch, np.load(root / 'serial' / 'cam00.npz') as single:
                for key in ['landmarks', 'visibilities', 'contacts', 'floor_contacts']:
                    np.testing.assert_array_equal(batch[key], single[key])
                    self.assertTrue((batch[key][:, 1] == 0).all())
                    self.assertTrue((batch[key][1] == 0).all())
                self.assertEqual(batch['landmarks'].shape, (5, 3, 2, 3))
                np.testing.assert_allclose(batch['landmarks'][4, 0, :, :2], [[22, 30], [22, 30]])


if __name__ == '__main__':
    unittest.main()
