"""Protect MHR camera conventions, checkpoint references, and visualization reuse."""
import subprocess
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np

from capture_processing.tools.visualize_mhr import camera_mesh_to_world, reference_camera, mesh_overlay
from capture_processing.tools.run_mhr import _save_mhr_params, _load_mhr_checkpoint
from capture_processing.utils.mhr import _merge_mhr_shards, maybe_create_mhr_params, run_mhr_parameterization


def arrays(count):
    return {key: np.zeros((count, *shape), dtype=np.float32) for key, shape in [
        ('num_views', ()), ('global_rot', (3,)), ('body_pose', (133,)), ('translation', (3,)),
        ('shape_params', (45,)), ('scale_params', (28,)), ('hand_pose', (108,)),
        ('expr_params', (72,)), ('triangulated', (70, 4)), ('loss_total', ())]}


class MHRVisualizationTests(unittest.TestCase):
    def setUp(self):
        signature = patch('capture_processing.utils.mhr.estimation_signature', return_value='test-signature')
        signature.start()
        self.addCleanup(signature.stop)
        environment = patch.dict(os.environ, {'VCI_MHR_ESTIMATION_SIGNATURE': 'test-signature'})
        environment.start()
        self.addCleanup(environment.stop)

    def test_reference_camera_is_recorded_or_unambiguous(self):
        data = {'cam_names': np.array(['cam00', 'cam01']), 'num_views': np.array([2, 1]),
                'frame_ids': np.array([5, 17])}
        self.assertEqual(reference_camera(data, 0), 'cam00')
        with self.assertRaisesRegex(ValueError, 'Frame 17'):
            reference_camera(data, 1)
        self.assertEqual(reference_camera(data, 1, 'cam01'), 'cam01')
        data['reference_cam_names'] = np.array(['cam00', 'cam01'])
        self.assertEqual(reference_camera(data, 1), 'cam01')
        with self.assertRaisesRegex(ValueError, 'Unknown'):
            reference_camera(data, 1, 'missing')

    def test_fk_flip_and_translation_are_applied_before_camera_inverse(self):
        camera = {'world_to_cam': np.array([[0, -1, 0, 2], [1, 0, 0, 3], [0, 0, 1, 4], [0, 0, 0, 1]], dtype=np.float32)}
        vertices = np.array([[1, 2, 3]], dtype=np.float32)
        world = camera_mesh_to_world(vertices, np.array([5, 6, 7]), camera)
        np.testing.assert_allclose(world, [[1, -4, 0]])
        np.testing.assert_array_equal(vertices, [[1, 2, 3]])

    def test_checkpoint_resume_and_shards_preserve_frame_references(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first, second, merged = root / 'one.npz', root / 'two.npz', root / 'merged.npz'
            def save(path, ids, references, complete):
                _save_mhr_params(path, frame_ids=ids, cam_names=['cam00', 'cam01'],
                                 faces=np.array([[0, 1, 2]]), **arrays(len(ids)),
                                 reference_cam_names=np.array(references),
                                 checkpointed_frames=len(ids), complete=complete)
            save(first, [2], ['cam01'], False)
            refs = np.array([''], dtype='U5')
            count = _load_mhr_checkpoint(first, frame_ids=[2], **arrays(1), reference_cam_names=refs)
            self.assertEqual(count, 1)
            self.assertEqual(refs.tolist(), ['cam01'])
            save(first, [2], ['cam01'], True)
            save(second, [0], ['cam00'], True)
            _merge_mhr_shards(merged, [first, second], [0, 2])
            with np.load(merged, allow_pickle=False) as data:
                self.assertEqual(data['reference_cam_names'].tolist(), ['cam00', 'cam01'])
            # Legacy exports stay readable without introducing object arrays.
            _save_mhr_params(second, frame_ids=[0], cam_names=['cam00'], faces=np.array([[0, 1, 2]]),
                             **arrays(1), checkpointed_frames=1, complete=True)
            with np.load(second, allow_pickle=False) as data:
                self.assertNotIn('reference_cam_names', data)

    def test_cached_estimate_visualizes_by_default_without_rerunning_estimation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'images').mkdir()
            (root / 'masks').mkdir()
            (root / 'calibration.json').write_text('{}')
            params = root / 'mhr_params.npz'
            _save_mhr_params(params, frame_ids=[0], cam_names=['cam00'], faces=np.array([[0, 1, 2]]),
                             **arrays(1), checkpointed_frames=1, complete=True)
            original = params.read_bytes()
            with patch('capture_processing.utils.mhr.resolve_sam3d_body_rerun_repo', return_value=Path('/repo')), \
                 patch('capture_processing.utils.mhr.run_mhr_visualization') as visualize, \
                 patch('capture_processing.utils.mhr.subprocess.run') as estimate:
                maybe_create_mhr_params(root, source='sam3d-body-rerun', repo_path=None,
                                        python_executable='python', device='2', max_frames=None,
                                        iterations=100, force=False)
            estimate.assert_not_called()
            visualize.assert_called_once_with(root, Path('/repo'), 'python', '2')
            self.assertEqual(params.read_bytes(), original)

    def test_new_estimate_visualizes_after_save_and_failed_visualization_can_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            params = root / 'mhr_params.npz'
            events = []
            def estimate(*args, **kwargs):
                events.append('estimate')
                _save_mhr_params(params, frame_ids=[0], cam_names=['cam00'], faces=np.array([[0, 1, 2]]),
                             **arrays(1), checkpointed_frames=1, complete=True)
            def visualize(*args, **kwargs):
                self.assertTrue(params.is_file())
                events.append('visualize')
                raise subprocess.CalledProcessError(1, 'visualize')
            options = dict(subject_root=root, repo=Path('/repo'), python_executable='python',
                           device='2', max_frames=None, iterations=100, force=False)
            with patch('capture_processing.utils.mhr._mhr_frame_ids', return_value=[0]), \
                 patch('capture_processing.utils.mhr.subprocess.run', side_effect=estimate) as fit, \
                 patch('capture_processing.utils.mhr.run_mhr_visualization', side_effect=visualize):
                with self.assertRaises(subprocess.CalledProcessError):
                    run_mhr_parameterization(**options)
                fit.assert_called_once()
            self.assertEqual(events, ['estimate', 'visualize'])
            original = params.read_bytes()
            with patch('capture_processing.utils.mhr.subprocess.run') as fit, \
                 patch('capture_processing.utils.mhr.run_mhr_visualization') as render:
                self.assertEqual(run_mhr_parameterization(**options), params)
                fit.assert_not_called()
                render.assert_called_once_with(root, Path('/repo'), 'python', '2')
            self.assertEqual(params.read_bytes(), original)

    def test_disabled_mhr_source_skips_estimation_and_visualization(self):
        with patch('capture_processing.utils.mhr.run_mhr_parameterization') as estimate, \
             patch('capture_processing.utils.mhr.run_mhr_visualization') as visualize:
            maybe_create_mhr_params(Path('/unused'), source='none', repo_path=None,
                                    python_executable='python', device=None, max_frames=None,
                                    iterations=100, force=False)
        estimate.assert_not_called()
        visualize.assert_not_called()

    def test_overlay_rejects_triangles_behind_camera(self):
        try:
            import cv2  # noqa: F401  # availability probe
        except ImportError:
            self.skipTest('OpenCV is not installed')
        image = np.zeros((32, 32, 3), dtype=np.uint8)
        camera = {'world_to_cam': np.eye(4), 'K': np.array([[10, 0, 16], [0, 10, 16], [0, 0, 1]])}
        faces = np.array([[0, 1, 2]])
        vertices = np.array([[-1, -1, 2], [1, -1, 2], [0, 1, 2]])
        self.assertGreater(mesh_overlay(image, vertices, faces, camera).sum(), 0)
        vertices[:, 2] *= -1
        np.testing.assert_array_equal(mesh_overlay(image, vertices, faces, camera), image)


if __name__ == '__main__':
    unittest.main()
