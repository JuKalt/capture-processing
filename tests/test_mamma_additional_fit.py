"""Extra fit completion and repairs must never rerun the baseline stages."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np

from capture_processing.utils.mamma_additional_fit import ensure_confidence_fit, confidence_fit_complete, PARAMS, MANIFEST
from capture_processing.core.vci_resume import quick_subject_complete
from capture_processing.utils.stage_cache import fit_artifacts


class AdditionalMammaFitTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.subject = Path(temporary.name) / 'subject'
        self.subject.mkdir()
        self.repo = Path(temporary.name) / 'repo'
        self.repo.mkdir()
        self.args = SimpleNamespace(smplx_source='mamma', mamma_repo_path=None, mamma_out_tag='vci',
            mamma_python='python', mamma_force=False, mamma_confidence_fit=True, semantic_parsing=False, device='cpu', mhr_source='none')
        self.fields = {'global_orient': np.zeros((2, 3)), 'transl': np.zeros((2, 3)),
            'body_pose': np.zeros((2, 63)), 'jaw_pose': np.zeros((2, 3)),
            'left_hand_pose': np.zeros((2, 45)), 'right_hand_pose': np.zeros((2, 45)),
            'betas': np.zeros((1, 16)), 'expression': np.zeros((2, 10))}
        np.savez(self.subject / 'smpl_params.npz', **self.fields)
        self.baseline_hash = hashlib.sha256((self.subject / 'smpl_params.npz').read_bytes()).hexdigest()
        self.base = self.subject / 'baseline'
        self.base.mkdir()
        self.config = self.base / 'run_config.yaml'
        self.config.write_text('optim: {}\n')
        self.fit_calls = self.viz_calls = self.diagnostic_calls = 0
        self.fail_visualization = False
        self.signature = 'test-inputs-v1'
        def inputs(*args):
            return (self.base / 'params.npz', self.config, {'cam_names': ['cam00']},
                    self.base / 'cap', self.base / 'landmarks', self.signature)
        def worker(command, **kwargs):
            if 'capture_processing.tools.run_mamma_confidence' in command:
                self.fit_calls += 1
                folder = Path(command[command.index('--out_path') + 1]) / 'images'
                folder.mkdir(parents=True, exist_ok=True)
                np.savez(folder / 'smplx_params_body_id-00.npz', smplx_pose=np.zeros((2, 165)),
                         smplx_translation=np.zeros((2, 3)), smplx_betas=np.zeros((1, 16)),
                         triangulated_3d_pts=np.zeros((2, 512, 3)))
                np.savez(folder / 'verts_joints_body_id-00.npz',
                         pred_vertices=np.zeros((2, 3, 3)), pred_joints=np.zeros((2, 127, 3)))
            else:
                self.viz_calls += 1
                if self.fail_visualization:
                    raise subprocess.CalledProcessError(1, command)
                folder = Path(command[command.index('--out_path') + 1]) / 'images'
                (folder / 'overlay').mkdir(parents=True, exist_ok=True)
                for path in [folder / 'scene.rrd', folder / 'preview.mp4', folder / 'overlay/cam00.mp4']:
                    path.write_bytes(b'complete visualization fixture')
        def diagnostics(subject, params, python, kind):
            path = subject / f'{kind}_fit_diagnostics.json'
            if not path.exists():
                self.diagnostic_calls += 1
                path.write_text(json.dumps({'metrics': {'reprojection_error': {'summary': {'count': 2}}}}))
        for target, replacement in [('capture_processing.utils.mamma_additional_fit.inputs', inputs),
                ('capture_processing.utils.mamma.resolve_mamma_repo', lambda path: self.repo),
                ('capture_processing.utils.mamma_additional_fit.subprocess.run', worker),
                ('capture_processing.utils.fit_metrics.prepare_smplx_fit_diagnostics', diagnostics)]:
            mock = patch(target, side_effect=replacement)
            mock.start()
            self.addCleanup(mock.stop)
        quiet = contextlib.redirect_stdout(io.StringIO())
        quiet.__enter__()
        self.addCleanup(quiet.__exit__, None, None, None)

    def test_completed_fit_skips_and_missing_preview_only_rerenders(self):
        ensure_confidence_fit(self.subject, self.args)
        self.assertTrue(confidence_fit_complete(self.subject, self.repo))
        ensure_confidence_fit(self.subject, self.args)
        self.assertEqual((self.fit_calls, self.viz_calls), (1, 1))
        preview = next((self.subject / 'mamma_confidence_output/ma_vis').rglob('preview.mp4'))
        preview.unlink()
        self.assertFalse(confidence_fit_complete(self.subject, self.repo))
        ensure_confidence_fit(self.subject, self.args)
        self.assertEqual((self.fit_calls, self.viz_calls), (1, 2))
        self.assertTrue(confidence_fit_complete(self.subject, self.repo))
        self.assertEqual(hashlib.sha256((self.subject / 'smpl_params.npz').read_bytes()).hexdigest(), self.baseline_hash)

    def test_corrupt_export_and_missing_diagnostics_repair_without_fitting(self):
        ensure_confidence_fit(self.subject, self.args)
        (self.subject / PARAMS).write_bytes(b'corrupt zip')
        ensure_confidence_fit(self.subject, self.args)
        (self.subject / 'smplx_confidence_fit_diagnostics.json').unlink()
        ensure_confidence_fit(self.subject, self.args)
        self.assertEqual((self.fit_calls, self.viz_calls), (1, 1))
        self.assertTrue(confidence_fit_complete(self.subject, self.repo))

    def test_interrupted_visualization_reuses_completed_3d_fit(self):
        self.fail_visualization = True
        with self.assertRaisesRegex(RuntimeError, 'visualization.log'):
            ensure_confidence_fit(self.subject, self.args)
        self.assertFalse(confidence_fit_complete(self.subject, self.repo))
        self.fail_visualization = False
        ensure_confidence_fit(self.subject, self.args)
        self.assertEqual((self.fit_calls, self.viz_calls), (1, 2))
        self.assertTrue(confidence_fit_complete(self.subject, self.repo))

    def test_changed_inputs_and_damaged_geometry_refit_only_the_extra_stage(self):
        ensure_confidence_fit(self.subject, self.args)
        mesh = next((self.subject / 'mamma_confidence_output/ma_3d').rglob('verts_joints_body_id-00.npz'))
        mesh.write_bytes(b'damaged')
        ensure_confidence_fit(self.subject, self.args)
        self.assertEqual((self.fit_calls, self.viz_calls), (2, 2))
        self.signature = 'changed-2d-observations'
        self.assertFalse(confidence_fit_complete(self.subject, self.repo))
        ensure_confidence_fit(self.subject, self.args)
        self.assertEqual((self.fit_calls, self.viz_calls), (3, 3))
        self.assertEqual(hashlib.sha256((self.subject / 'smpl_params.npz').read_bytes()).hexdigest(), self.baseline_hash)

    def test_whole_subject_completion_requires_extra_stage_and_fresh_metrics(self):
        from capture_processing.utils.subject_manifest import SCHEMA_VERSION
        (self.subject / 'subject_manifest.json').write_text(json.dumps(
            {'schema_version': SCHEMA_VERSION, 'complete': True}))
        (self.subject / 'fit_metrics.json').write_text(json.dumps({'complete': True,
            'fits': {'smplx': {'status': 'complete'}}, 'cache_artifacts': {}}))
        with patch('capture_processing.core.vci_resume.quick_conversion_complete', return_value=True), \
             patch('capture_processing.utils.stage_cache.metrics_signature', return_value='fixture-metrics'):
            self.assertFalse(quick_subject_complete(self.args, self.subject))
            ensure_confidence_fit(self.subject, self.args)
            self.assertFalse(quick_subject_complete(self.args, self.subject))
            fit = {'status':'complete', 'metrics': {'silhouette_iou':{}, 'silhouette_error':{}}}
            report = {'complete': True, 'silhouette_evaluation': {'complete_comparison':True},
                'input_signature':'fixture-metrics',
                'fits': {'smplx':fit, 'smplx_confidence':fit}, 'cache_artifacts': {},
                'fit_artifacts': fit_artifacts(self.subject),
                'evaluation_options': {'cameras': None, 'max_frames': None}}
            (self.subject / 'fit_metrics.json').write_text(json.dumps(report))
            self.assertTrue(quick_subject_complete(self.args, self.subject))
            report['evaluation_options']['max_frames'] = 1
            (self.subject / 'fit_metrics.json').write_text(json.dumps(report))
            self.assertFalse(quick_subject_complete(self.args, self.subject))
            report['evaluation_options']['max_frames'] = None
            (self.subject / 'fit_metrics.json').write_text(json.dumps(report))
            (self.subject / 'smplx_confidence_fit_diagnostics.json').write_text('{}')
            self.assertFalse(quick_subject_complete(self.args, self.subject))

    def test_shared_smplx_helper_runs_extra_only_when_entrypoint_enables_it(self):
        from capture_processing.core.smplx_pipeline import ensure_subject_smplx
        arguments = dict(smplx_source='mamma', mamma_repo_path=None, mamma_preset=None,
                         mamma_out_tag='vci', mamma_python='python', mamma_force=False,
                         mamma_workers=1, device='cpu')
        with patch('capture_processing.core.smplx_pipeline.find_existing_mamma_params', return_value=None):
            ensure_subject_smplx(self.subject, **arguments)
            self.assertEqual(self.fit_calls, 0)
            ensure_subject_smplx(self.subject, mamma_confidence_fit=True, **arguments)
        self.assertEqual(self.fit_calls, 1)
        self.assertTrue(confidence_fit_complete(self.subject, self.repo))

    def test_asset_paths_reuse_saved_locations_and_honor_environment(self):
        from capture_processing.utils.mamma_additional_fit import _assets
        models = self.repo / 'custom-models'
        models.mkdir()
        matrix = self.repo / 'custom-verts.pkl'
        matrix.write_bytes(b'fixture')
        with patch.dict('os.environ', {'MAMMA_SMPLX_LOCKHEAD_MODELS': '',
                                      'MAMMA_DOWNSAMPLED_VERTS_PKL': ''}):
            self.assertEqual(_assets(self.repo, {'smplx_models': str(models),
                'downsampled_verts': str(matrix)}), (models, matrix))
        with patch.dict('os.environ', {'MAMMA_SMPLX_LOCKHEAD_MODELS': str(models),
                                      'MAMMA_DOWNSAMPLED_VERTS_PKL': str(matrix)}):
            self.assertEqual(_assets(self.repo, {}), (models, matrix))

    def test_metric_validation_requires_both_metrics_and_current_artifacts(self):
        from capture_processing.utils.mamma_additional_fit import validate_confidence_metrics
        ensure_confidence_fit(self.subject, self.args)
        path = self.subject / 'fit_metrics.json'
        report = {'complete': True, 'fit_artifacts': fit_artifacts(self.subject),
                  'fits': {'smplx_confidence': {'status': 'complete', 'metrics': {
                      'reprojection_error': {}, 'silhouette_iou': {}}}}}
        path.write_text(json.dumps(report))
        validate_confidence_metrics(self.subject, path)
        del report['fits']['smplx_confidence']['metrics']['silhouette_iou']
        path.write_text(json.dumps(report))
        with self.assertRaisesRegex(RuntimeError, 'missing metrics'):
            validate_confidence_metrics(self.subject, path)
        report['fits']['smplx_confidence']['metrics']['silhouette_iou'] = {}
        report['fit_artifacts'] = {}
        path.write_text(json.dumps(report))
        with self.assertRaisesRegex(RuntimeError, 'missing metrics'):
            validate_confidence_metrics(self.subject, path)

    def test_disabled_extra_fit_does_not_start_any_work(self):
        self.args.mamma_confidence_fit = False
        ensure_confidence_fit(self.subject, self.args)
        self.assertEqual((self.fit_calls, self.viz_calls), (0, 0))
        self.assertFalse((self.subject / MANIFEST).exists())


if __name__ == '__main__':
    unittest.main()
