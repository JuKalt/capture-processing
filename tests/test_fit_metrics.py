"""Protect residual geometry, frame coverage, missing values, and checkpoint metrics."""
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch

from capture_processing.utils.fit_metrics import reprojection_errors, root_motion, write_fit_metrics
from capture_processing.tools.run_mhr import _evaluate_mhr_fit, _save_mhr_params, _load_mhr_checkpoint
from capture_processing.utils.mhr import _merge_mhr_shards
from tests.test_mhr_visualization import arrays


class FitMetricsTests(unittest.TestCase):
    def test_projection_residual_masks_depth_visibility_and_invalid_observations(self):
        world = np.array([[[1, 0, 1], [0, 0, -1], [1, 1, 1], [2, 2, 1]]])
        observed = np.array([[[4, 4], [4, 4], [5, 5], [np.nan, 0]]])
        mean, count, total = reprojection_errors(world, observed, np.eye(3), np.eye(4), np.array([[1, 1, .2, 1]]))
        np.testing.assert_allclose(mean, [5])
        np.testing.assert_array_equal(count, [1])
        np.testing.assert_allclose(total, [5])

    def test_temporal_metrics_do_not_bridge_gaps(self):
        values = root_motion([[0, 0, 0], [1, 0, 0], [100, 0, 0], [101, 0, 0]],
                             np.array([0, 1, 4, 5]), np.ones(4, dtype=bool))
        self.assertEqual(values['root_step_m']['per_frame'], [None, 1., None, 1.])
        self.assertEqual(values['root_second_difference_m']['summary']['count'], 0)

    def test_report_distinguishes_missing_invalid_incomplete_and_finite_frames(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            np.savez(root / 'smpl_params.npz', global_orient=np.zeros((3, 3)),
                     body_pose=np.zeros((3, 63)), transl=np.zeros((3, 3)), betas=np.zeros((1, 10)))
            np.savez(root / 'mhr_from_smplx_params.npz', frame_ids=[0, 2], complete=False,
                     lbs_model_params=np.zeros((2, 204)), identity_coeffs=np.zeros((1, 45)),
                     face_expr_coeffs=np.zeros((2, 72)), fitting_errors_cm=[.5, np.nan], checkpointed_frames=1)
            output = write_fit_metrics(root)
            report = json.loads(output.read_text())['fits']
            self.assertEqual(report['mhr']['status'], 'missing')
            conversion = report['mhrfromsmplx']
            self.assertEqual(conversion['status'], 'incomplete')
            self.assertEqual(conversion['coverage_fraction'], 1/3)
            self.assertEqual(conversion['metrics']['conversion_vertex_error']['per_frame'], [.5, None])
            self.assertIn('reprojection_error', report['smplx']['unavailable'])
            self.assertNotIn('NaN', output.read_text())
            np.savez(root / 'mhr_params.npz', body_pose=np.zeros((2, 1)))
            self.assertEqual(json.loads(write_fit_metrics(root).read_text())['fits']['mhr']['status'], 'invalid')

    def test_mhr_metrics_use_final_model_parameters_and_camera_convention(self):
        class Head:
            def mhr_forward(self, **kwargs):
                self.arguments = kwargs
                # SAM3D flip converts raw [1,-1,-2] into camera [1,1,2].
                return None, torch.tensor([[[1., -1., -2.]] * 70]), None
        head = Head()
        calibration = {'cam00': {'world_to_cam': np.eye(4), 'K': np.eye(3)}}
        observed = np.array([[[.5, .5, 1.]] * 70])
        triangulated = np.array([[1., 1., 2., 1.]] * 70)
        metrics = _evaluate_mhr_fit(head, np.zeros(3), np.ones(133), np.zeros(3),
                                   np.zeros(45), np.zeros(28), np.zeros(108), np.zeros(72),
                                   calibration, ['cam00'], observed, triangulated, 'cpu')
        self.assertEqual(metrics['reprojection_error_px'], 0.)
        self.assertEqual(metrics['triangulated_keypoint_error_mm'], 0.)
        self.assertTrue(torch.all(head.arguments['body_pose_params'] == 1))

    def test_mhr_checkpoint_and_shard_merge_keep_diagnostics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shards = [root / 'one.npz', root / 'two.npz']
            for index, path in enumerate(shards):
                _save_mhr_params(path, frame_ids=[index], cam_names=['cam00'],
                                 faces=np.array([[0, 1, 2]]), **arrays(1),
                                 fit_diagnostics={'reprojection_error_px': np.array([index + 3.]),
                                                  'triangulated_keypoint_error_mm': np.array([index + 5.])},
                                 checkpointed_frames=1, complete=index == 1)
            restored = {'reprojection_error_px': np.array([np.nan])}
            _load_mhr_checkpoint(shards[0], frame_ids=[0], **arrays(1), fit_diagnostics=restored)
            np.testing.assert_array_equal(restored['reprojection_error_px'], [3.])
            # Mark checkpoint complete for the worker merge.
            with np.load(shards[0]) as data:
                saved = {k: data[k] for k in data.files}
            saved['complete'] = True
            np.savez(shards[0], **saved)
            output = root / 'mhr_params.npz'
            _merge_mhr_shards(output, shards, [0, 1])
            with np.load(output) as data:
                np.testing.assert_array_equal(data['reprojection_error_px'], [3., 4.])
                np.testing.assert_array_equal(data['triangulated_keypoint_error_mm'], [5., 6.])


if __name__ == '__main__':
    unittest.main()
