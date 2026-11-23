"""MHR reuse requires matching inputs, settings, frame IDs, and complete arrays."""
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from capture_processing.utils.mhr import run_mhr_parameterization, is_mhr_params_complete
from capture_processing.tools.run_mhr import _save_mhr_params, _load_mhr_checkpoint
from tests.test_mhr_visualization import arrays


class MHRResumeTests(unittest.TestCase):
    def test_changed_settings_and_inputs_recompute_but_identical_run_reuses(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'images/cam00').mkdir(parents=True)
            (root / 'masks').mkdir()
            (root / 'calibration.json').write_text('{"cam00": {}}')
            image = root / 'images/cam00/color_00000000.jpg'
            Image.new('RGB', (4, 4)).save(image)
            def estimate(command, **kwargs):
                with patch.dict(os.environ, kwargs['env']):
                    _save_mhr_params(root / 'mhr_params.npz', frame_ids=[0], cam_names=['cam00'],
                                     faces=np.array([[0, 1, 2]]), **arrays(1), checkpointed_frames=1, complete=True)
            options = dict(subject_root=root, repo=root, python_executable='python',
                           device=None, max_frames=None, iterations=100, force=False)
            with patch('capture_processing.utils.silhouette_metrics._repo_stamp', return_value={}), \
                 patch('capture_processing.utils.silhouette_metrics._sam_asset_stamps', return_value={}), \
                 patch('capture_processing.utils.mhr.run_mhr_visualization'), \
                 patch('capture_processing.utils.mhr.subprocess.run', side_effect=estimate) as fit:
                run_mhr_parameterization(**options)
                run_mhr_parameterization(**options)
                self.assertEqual(fit.call_count, 1)
                options['iterations'] = 50
                run_mhr_parameterization(**options)
                self.assertEqual(fit.call_count, 2)
                options['max_frames'] = 1
                run_mhr_parameterization(**options)
                self.assertEqual(fit.call_count, 3)
                Image.new('RGB', (4, 4), 'red').save(image)
                run_mhr_parameterization(**options)
                self.assertEqual(fit.call_count, 4)

    def test_legacy_or_malformed_archive_is_not_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'fit.npz'
            np.savez(path, body_pose=np.zeros((1, 133)))
            self.assertFalse(is_mhr_params_complete(path))
            np.savez(path, complete=True)
            self.assertFalse(is_mhr_params_complete(path))
            path.write_bytes(b'corrupt')
            self.assertFalse(is_mhr_params_complete(path))

    def test_complete_checkpoint_does_not_bypass_frame_or_signature_checks(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'fit.npz'
            with patch.dict(os.environ, {'VCI_MHR_ESTIMATION_SIGNATURE': 'old'}):
                _save_mhr_params(path, frame_ids=[0], cam_names=['cam00'], faces=np.array([[0, 1, 2]]),
                                 **arrays(1), checkpointed_frames=1, complete=True)
                self.assertEqual(_load_mhr_checkpoint(path, frame_ids=[1], **arrays(1)), 0)
                self.assertEqual(_load_mhr_checkpoint(path, frame_ids=[0], **arrays(1)), 1)
            with patch.dict(os.environ, {'VCI_MHR_ESTIMATION_SIGNATURE': 'new'}):
                self.assertEqual(_load_mhr_checkpoint(path, frame_ids=[0], **arrays(1)), 0)
