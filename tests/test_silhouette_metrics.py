"""Shared observations, rasterized geometry, and missing/stale metric backfills."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from capture_processing.utils.fit_metrics import write_fit_metrics
from capture_processing.utils.silhouette_metrics import (binary_mask, media_path, render_silhouette,
                                         silhouette_counts)
from capture_processing.tools.evaluate_silhouette_fit import evaluate


class SilhouetteMetricsTests(unittest.TestCase):
    def test_iou_error_and_empty_convention(self):
        values = silhouette_counts([[1, 1], [0, 0]], [[1, 0], [1, 0]])
        self.assertAlmostEqual(values['iou'], 1/3)
        self.assertAlmostEqual(values['silhouette_error'], 2/3)
        self.assertEqual(values['xor'], 2)
        self.assertEqual(silhouette_counts(np.zeros((2, 2)), np.zeros((2, 2)))['iou'], 1.)
        self.assertEqual(silhouette_counts(np.ones((2, 2)), np.zeros((2, 2)))['iou'], 0.)
        np.testing.assert_array_equal(binary_mask(np.array([[0, 1]])), [[False, True]])
        np.testing.assert_array_equal(binary_mask(np.array([[0, 6, 127, 128, 255]])), [[False, False, False, True, True]])

    def test_rasterizer_unions_overlapping_faces_and_rejects_behind_camera(self):
        vertices = np.array([[2, 2, 1], [7, 2, 1], [2, 7, 1.]])
        faces = np.array([[0, 1, 2]])
        one = render_silhouette(vertices, faces, np.eye(3), np.eye(4), (10, 10))
        two = render_silhouette(vertices, np.repeat(faces, 2, axis=0), np.eye(3), np.eye(4), (10, 10))
        np.testing.assert_array_equal(one, two)
        self.assertTrue(one[3, 3])
        self.assertFalse(one[8, 8])
        self.assertFalse(render_silhouette(-vertices, faces, np.eye(3), np.eye(4), (10, 10)).any())
        crossing = vertices.copy()
        crossing[0, 2] = -1
        self.assertTrue(render_silhouette(crossing, faces, np.eye(3), np.eye(4), (10, 10)).any())

    def _fixture(self, root):
        calibration = {camera: {'K': [[20, 0, 8], [0, 20, 8], [0, 0, 1]],
                                'R': np.eye(3).reshape(-1).tolist(), 'T': [0, 0, 0]}
                       for camera in ['cam00', 'cam01']}
        (root / 'calibration.json').write_text(json.dumps(calibration))
        vertices = np.array([[0, 0, 2], [1, 0, 2], [0, 1, 2.]])
        faces = np.array([[0, 1, 2]])
        K = np.array(calibration['cam00']['K'], dtype=float)
        K[:2] /= 2  # Images are 32x32, masks are 16x16.
        target = render_silhouette(vertices, faces, K, np.eye(4), (16, 16))
        for camera in calibration:
            for kind in ['images', 'masks']:
                (root / kind / camera).mkdir(parents=True)
            for frame in range(3):
                Image.new('RGB', (32, 32)).save(root / 'images' / camera / f'{frame:08d}.jpg')
                Image.fromarray(target.astype(np.uint8) * 255).save(root / 'masks' / camera / f'{frame:08d}.png')
        np.savez(root / 'smpl_params.npz', global_orient=np.zeros((3, 3)), body_pose=np.zeros((3, 63)),
                 transl=np.zeros((3, 3)), betas=np.zeros((1, 10)))
        np.savez(root / 'mhr_params.npz', frame_ids=[0, 2], global_rot=np.zeros((2, 3)),
                 body_pose=np.zeros((2, 133)), translation=np.zeros((2, 3)), shape=np.zeros((2, 45)),
                 scale=np.zeros((2, 28)), hand_pose=np.zeros((2, 108)), expression=np.zeros((2, 72)),
                 reference_cam_names=['cam00', 'cam00'], num_views=[2, 2], cam_names=['cam00', 'cam01'])
        np.savez(root / 'mhr_from_smplx_params.npz', frame_ids=[0, 1, 2], lbs_model_params=np.zeros((3, 204)),
                 identity_coeffs=np.zeros((1, 45)), face_expr_coeffs=np.zeros((3, 72)), fitting_errors_cm=[0, 0, 0])
        return vertices, faces

    def test_all_three_fits_share_masks_and_cached_subjects_backfill_only_missing_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            vertices, faces = self._fixture(root)
            requests = []
            def worker(command, **kwargs):
                request = json.loads(Path(command[-1]).read_text())
                requests.append(request)
                result = evaluate(root, request, provider=lambda *args: (lambda frame: (vertices, faces), lambda: None))
                (root / f"silhouette_metrics.{request['kind']}.json").write_text(json.dumps(result))
            with patch('capture_processing.utils.silhouette_metrics.subprocess.run', side_effect=worker) as run, \
                 patch('capture_processing.utils.silhouette_metrics._repo_stamp', return_value={'revision': 'fixture'}):
                report = json.loads(write_fit_metrics(root).read_text())
                self.assertEqual(run.call_count, 3)
                self.assertTrue(report['silhouette_evaluation']['complete_comparison'])
                self.assertEqual(report['silhouette_evaluation']['frame_ids'], [0, 2])
                shared_views = [[(v['camera'], v['frame_id'], v['mask']) for v in request['views']] for request in requests]
                self.assertEqual(shared_views[0], shared_views[1])
                self.assertEqual(shared_views[1], shared_views[2])
                for fit in report['fits'].values():
                    self.assertEqual(fit['metrics']['silhouette_iou']['summary']['mean'], 1.)
                    self.assertEqual(fit['metrics']['silhouette_error']['summary']['mean'], 0.)
                # Removing the aggregate file reuses scores; no fitting/model work.
                (root / 'fit_metrics.json').unlink()
                write_fit_metrics(root)
                self.assertEqual(run.call_count, 3)
                cache = root / 'silhouette_metrics.smplx.json'
                contents = json.loads(cache.read_text())
                del contents['metrics']['silhouette_error']
                cache.write_text(json.dumps(contents))
                write_fit_metrics(root)
                self.assertEqual(run.call_count, 4)
                self.assertEqual(requests[-1]['kind'], 'smplx')
                # Changing a shared mask invalidates every fit's scores.
                Image.new('L', (16, 16), 255).save(root / 'masks/cam00/00000000.png')
                write_fit_metrics(root)
                self.assertEqual(run.call_count, 7)

    def test_adding_confidence_fit_only_scores_the_new_archive(self):
        import shutil
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            vertices, faces = self._fixture(root)
            kinds = []
            def worker(command, **kwargs):
                request_path = Path(command[command.index('--request') + 1])
                request = json.loads(request_path.read_text())
                kinds.append(request['kind'])
                if request['kind'] == 'smplx_confidence':
                    self.assertIn('capture_processing.tools.evaluate_confidence_silhouette', command)
                result = evaluate(root, request, provider=lambda *args: (
                    lambda frame: (vertices, faces), lambda: None))
                (root / f"silhouette_metrics.{request['kind']}.json").write_text(json.dumps(result))
            with patch('capture_processing.utils.silhouette_metrics.subprocess.run', side_effect=worker), \
                 patch('capture_processing.utils.silhouette_metrics._repo_stamp', return_value={'revision': 'fixture'}):
                write_fit_metrics(root)
                self.assertEqual(kinds, ['smplx', 'mhr', 'mhrfromsmplx'])
                shutil.copyfile(root / 'smpl_params.npz', root / 'smpl_params_confidence.npz')
                report = json.loads(write_fit_metrics(root).read_text())
                self.assertEqual(kinds, ['smplx', 'mhr', 'mhrfromsmplx', 'smplx_confidence'])
                self.assertEqual(report['fits']['smplx_confidence']['status'], 'complete')
                self.assertEqual(report['fits']['smplx_confidence']['metrics']['silhouette_iou']['summary']['mean'], 1.)
                signatures = {fit['silhouette_selection_signature'] for fit in report['fits'].values()}
                self.assertEqual(len(signatures), 1)
                (root / 'fit_metrics.json').unlink()
                write_fit_metrics(root)
                self.assertEqual(len(kinds), 4)
                # Updating only the extra fit leaves the original three scores cached.
                with np.load(root / 'smpl_params_confidence.npz') as data:
                    fields = {key: data[key] for key in data.files}
                fields['transl'][0, 0] += .1
                np.savez(root / 'smpl_params_confidence.npz', **fields)
                write_fit_metrics(root)
                self.assertEqual(kinds[-1], 'smplx_confidence')
                self.assertEqual(len(kinds), 5)

    def test_frame_gaps_and_missing_views_are_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            vertices, faces = self._fixture(root)
            self.assertEqual(media_path(root / 'masks', 'cam00', 0, 'mask').suffix, '.png')
            # An empty common intersection produces unavailable metrics, not a zero score.
            with np.load(root / 'mhr_params.npz') as data:
                parameters = {k: data[k] for k in data.files}
            parameters['frame_ids'] = np.array([9, 10])
            np.savez(root / 'mhr_params.npz', **parameters)
            with patch('capture_processing.utils.silhouette_metrics.subprocess.run') as run, \
                 patch('capture_processing.utils.silhouette_metrics._repo_stamp', return_value={'revision': 'fixture'}):
                report = json.loads(write_fit_metrics(root).read_text())
                run.assert_not_called()
                self.assertFalse(report['silhouette_evaluation']['complete_comparison'])
                for fit in report['fits'].values():
                    self.assertIn('silhouette', fit['unavailable'])
                    self.assertNotIn('silhouette_iou', fit['metrics'])


if __name__ == '__main__':
    unittest.main()
