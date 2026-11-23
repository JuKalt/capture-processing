"""Protect MAMMA launches from the export environment's Conda hooks."""

import os
import numpy as np
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from capture_processing.utils.mamma import expected_mamma_params_path, run_mamma, find_existing_mamma_params


class MammaRuntimeTests(unittest.TestCase):
    def test_fitting_and_preflights_receive_isolated_conda_state(self):
        parent = {
            "PATH": "/custom/bin:/conda/envs/export/bin:/conda/bin:/usr/bin",
            "CONDA_EXE": "/conda/bin/conda",
            "CONDA_PREFIX": "/conda/envs/export",
            "CONDA_PREFIX_1": "/conda",
            "CONDA_DEFAULT_ENV": "export",
            "CONDA_SHLVL": "2",
            "CONDA_PROMPT_MODIFIER": "(export) ",
            "CONDA_BACKUP_CXX": "/conda/bin/compiler",
            "MAMMA_MA2D_CHECKPOINT": "/models/checkpoint.ckpt",
            "DNA_RENDERING_SAMPLE_CODE": "/datasets/reader",
        }
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, parent, clear=True):
            root = Path(directory)
            repo = root / "mamma"
            preset = root / "preset.yaml"
            footage = root / "sequence"
            output = root / "output"
            preset.write_text("ma_cap:\n  enabled: true\n")
            expected = expected_mamma_params_path(output, "dna", footage, "images")
            expected.parent.mkdir(parents=True)
            np.savez(expected, smplx_pose=np.zeros((1, 165)), smplx_translation=np.zeros((1, 3)), smplx_betas=np.zeros((1, 10)))
            with patch("capture_processing.utils.mamma._check_mamma_python_dependencies") as dependencies, \
                 patch("capture_processing.utils.mamma._check_mamma_assets") as assets, \
                 patch("capture_processing.utils.mamma._run_mamma_with_progress") as pipeline:
                result = run_mamma(
                    repo=repo, preset=preset, footage_root=footage,
                    seq_name="images", calibration_path=root / "calibration.json",
                    output_root=output, tag="dna", python_executable="conda run -n mamma python",
                    force=False, device="7",
                )
            self.assertEqual(result, expected)
            child = pipeline.call_args.args[2]
            self.assertEqual(dependencies.call_args.args[2], child)
            self.assertEqual(assets.call_args.args[2], child)
            self.assertEqual(child["CONDA_SHLVL"], "0")
            self.assertFalse(any(k == "CONDA_PREFIX" or k.startswith(("CONDA_PREFIX_", "CONDA_BACKUP_")) for k in child))
            self.assertNotIn("CONDA_DEFAULT_ENV", child)
            self.assertNotIn("CONDA_PROMPT_MODIFIER", child)
            for name in ("CONDA_EXE", "PATH", "MAMMA_MA2D_CHECKPOINT", "DNA_RENDERING_SAMPLE_CODE"):
                self.assertEqual(child[name], parent[name])
            self.assertEqual(child["CUDA_VISIBLE_DEVICES"], "7")
            self.assertEqual(dict(os.environ), parent)


if __name__ == "__main__":
    unittest.main()


class MammaArtifactTests(unittest.TestCase):
    def test_corrupt_fit_clears_only_downstream_done_markers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output, footage = root / 'output', root / 'subject'
            preset = root / 'preset.yaml'
            preset.write_text('ma_cap:\n  enabled: true\n')
            params = expected_mamma_params_path(output, 'vci', footage, 'images')
            params.parent.mkdir(parents=True)
            params.write_bytes(b'corrupt')
            markers = {}
            for step in ('ma_cap', 'ma_masks', 'ma_2d', 'ma_3d', 'ma_vis'):
                markers[step] = output / step / 'vci' / 'subject' / 'images' / 'DONE'
                markers[step].parent.mkdir(parents=True, exist_ok=True)
                markers[step].touch()
            self.assertIsNone(find_existing_mamma_params(output, 'vci', footage, 'images'))
            def pipeline(*args):
                for step in ('ma_cap', 'ma_masks', 'ma_2d'):
                    self.assertTrue(markers[step].exists())
                for step in ('ma_3d', 'ma_vis'):
                    self.assertFalse(markers[step].exists())
                np.savez(params, smplx_pose=np.zeros((1, 165)), smplx_translation=np.zeros((1, 3)), smplx_betas=np.zeros((1, 10)))
            with patch('capture_processing.utils.mamma._check_mamma_python_dependencies'), \
                 patch('capture_processing.utils.mamma._check_mamma_assets'), \
                 patch('capture_processing.utils.mamma._run_mamma_with_progress', side_effect=pipeline):
                self.assertEqual(run_mamma(repo=root, preset=preset, footage_root=footage,
                    seq_name='images', calibration_path=root / 'calibration.json', output_root=output,
                    tag='vci', python_executable='python', force=False, device=None), params)

    def test_existing_fit_does_not_skip_unfinished_mamma_visualization(self):
        from argparse import Namespace
        from capture_processing.core.smplx_pipeline import create_smpl_params_with_mamma
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'images').mkdir()
            preset = root / 'preset.yaml'
            preset.write_text('ma_3d:\n  enabled: true\nma_vis:\n  enabled: true\n')
            params = expected_mamma_params_path(root / 'mamma_output', 'vci', root, 'images')
            params.parent.mkdir(parents=True)
            np.savez(params, smplx_pose=np.zeros((1, 165)), smplx_translation=np.zeros((1, 3)), smplx_betas=np.zeros((1, 10)))
            (params.parent / 'DONE').touch()
            args = Namespace(mamma_preset=preset, mamma_workers=1, mamma_force=False,
                             mamma_out_tag='vci', mamma_python='python', device=None)
            with patch('capture_processing.core.smplx_pipeline.prepare_mamma_runtime_preset', return_value=preset), \
                 patch('capture_processing.core.smplx_pipeline.write_mamma_calibration_from_thuman'), \
                 patch('capture_processing.core.smplx_pipeline.prepare_smplx_fit_diagnostics'), \
                 patch('capture_processing.core.smplx_pipeline.run_mamma', return_value=params) as run:
                create_smpl_params_with_mamma(args, root, root)
                run.assert_called_once()
                done = root / 'mamma_output/ma_vis/vci' / root.name / 'images/DONE'
                done.parent.mkdir(parents=True)
                done.touch()
                run.reset_mock()
                create_smpl_params_with_mamma(args, root, root)
                run.assert_not_called()
