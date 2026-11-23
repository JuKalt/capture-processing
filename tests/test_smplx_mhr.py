"""Verify parameter/frame preservation, cache invalidation, and pipeline ordering."""
import argparse
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from capture_processing.tools.convert_smplx_to_mhr import load_parameters, save_conversion
from capture_processing.utils.smplx_mhr import (ConversionOptions, OUTPUT_NAME, add_conversion_arguments,
                                 convert_subject_smplx_to_mhr, options_from_args)


class OfficialConversionTests(unittest.TestCase):
    def test_loader_preserves_translation_and_noncontiguous_frames(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'smpl_params.npz'
            np.savez(source, global_orient=np.zeros((2, 3)), body_pose=np.zeros((2, 63)),
                     transl=[[1, 2, 3], [4, 5, 6]], betas=np.ones((1, 11)), frame_ids=[2, 8])
            params, ids = load_parameters(source)
            np.testing.assert_array_equal(ids, [2, 8])
            np.testing.assert_array_equal(params['transl'], [[1, 2, 3], [4, 5, 6]])
            self.assertEqual(params['betas'].shape, (2, 11))
            self.assertEqual(params['left_hand_pose'].shape, (2, 45))
            np.savez(source, global_orient=np.zeros((2, 3)), body_pose=np.zeros((2, 63)),
                     transl=np.full((2, 3), np.nan), betas=np.ones((1, 11)))
            with self.assertRaisesRegex(ValueError, 'Nonfinite.*transl'):
                load_parameters(source)

    def test_export_broadcasts_identity_and_retains_native_units(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / OUTPUT_NAME
            result = SimpleNamespace(result_parameters={
                'lbs_model_params': torch.ones(2, 204), 'identity_coeffs': torch.zeros(1, 45),
                'face_expr_coeffs': torch.zeros(2, 72)}, result_errors=np.array([.1, .2]))
            save_conversion(output, result, np.array([3, 7]), np.array([[0, 1, 2]]), 'signature')
            with np.load(output, allow_pickle=False) as data:
                self.assertEqual(data['identity_coeffs'].shape, (2, 45))
                self.assertEqual(data['units'].item(), 'centimeters')
                self.assertEqual(data['coordinate_system'].item(), 'world')
                np.testing.assert_array_equal(data['frame_ids'], [3, 7])
                self.assertTrue(data['complete'].item())
            result.result_parameters['lbs_model_params'][0, 0] = float('nan')
            original = output.read_bytes()
            with self.assertRaisesRegex(ValueError, 'Invalid converted'):
                save_conversion(output, result, np.array([3, 7]), np.array([[0, 1, 2]]), 'bad')
            self.assertEqual(original, output.read_bytes())

    def test_cache_invalidation_and_cpu_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'smpl_params.npz').write_bytes(b'source')
            model = root / 'model.npz'
            model.touch()
            (root / 'tools/mhr_smpl_conversion').mkdir(parents=True)
            (root / 'tools/mhr_smpl_conversion/conversion.py').touch()
            output = root / OUTPUT_NAME
            def converted(signature):
                np.savez(output, complete=True, conversion_signature=signature, frame_ids=np.arange(2),
                         lbs_model_params=np.zeros((2,204)), identity_coeffs=np.zeros((2,45)),
                         face_expr_coeffs=np.zeros((2,72)))
            converted('old')
            options = ConversionOptions(repo_path=root, model_path=model, python='python -u')
            with patch('capture_processing.utils.smplx_mhr.conversion_signature', return_value='old'), \
                 patch('capture_processing.utils.mhr.run_mhr_visualization') as render, \
                 patch('capture_processing.utils.smplx_mhr.subprocess.run') as run:
                self.assertEqual(convert_subject_smplx_to_mhr(root, options), output)
                run.assert_not_called()
                render.assert_called_once_with(root, root, 'python -u', 'cpu', kind='mhrfromsmplx')
            with patch('capture_processing.utils.smplx_mhr.conversion_signature', return_value='new'), \
                 patch('capture_processing.utils.mhr.run_mhr_visualization') as render, \
                 patch('capture_processing.utils.smplx_mhr.subprocess.run', side_effect=lambda *a, **kw: converted('new')) as run:
                convert_subject_smplx_to_mhr(root, options)
                self.assertEqual(run.call_args.kwargs['env']['CUDA_VISIBLE_DEVICES'], '')
                self.assertEqual(run.call_args.kwargs['env']['OMP_NUM_THREADS'], '1')
                self.assertEqual(run.call_args.args[0][:2], ['python', '-u'])
            self.assertIsNone(convert_subject_smplx_to_mhr(root, ConversionOptions(enabled=False)))

    def test_cli_default_and_opt_out(self):
        parser = argparse.ArgumentParser()
        add_conversion_arguments(parser)
        self.assertTrue(options_from_args(parser.parse_args([])).enabled)
        self.assertFalse(options_from_args(parser.parse_args(['--skip-smplx-to-mhr'])).enabled)

    def test_vci_runs_official_conversion_before_sam3d_even_when_sam3d_disabled(self):
        from capture_processing.core.vci_pipeline import _create_mhr_params_from_args
        args = SimpleNamespace(smplx_mhr_options=ConversionOptions(), mhr_source='none',
                               mamma_confidence_fit=False, semantic_parsing=False,
                               mhr_repo_path=None, mhr_python='python', device=None,
                               mhr_max_frames=None, mhr_iterations=100, mhr_force=False)
        calls = []
        with patch('capture_processing.core.vci_pipeline.convert_subject_smplx_to_mhr', side_effect=lambda *a: calls.append('official')), \
             patch('capture_processing.core.vci_pipeline.maybe_create_mhr_params', side_effect=lambda *a, **kw: calls.append('sam3d')), \
             patch('capture_processing.core.vci_pipeline.write_fit_metrics', side_effect=lambda *a: calls.append('metrics')), \
             patch('capture_processing.core.vci_pipeline.validate_fit_metrics'), \
             tempfile.TemporaryDirectory() as tmp:
            _create_mhr_params_from_args(args, Path(tmp))
        self.assertEqual(calls, ['official', 'metrics'])  # Disabled SAM3D is not invoked.


if __name__ == '__main__':
    unittest.main()
