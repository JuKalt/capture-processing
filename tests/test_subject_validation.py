import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from capture_processing.tools.evaluate_fit_metrics import main as metrics_main
from capture_processing.tools.validate_subject import main as validation_main
from capture_processing.utils.subject_validation import cached_metrics_complete
from tests import test_vci_resume as fixtures
from capture_processing.core.vci_pipeline import _create_mhr_params_from_args as run_postprocessing


class CompletionTests(unittest.TestCase):
    setUp = fixtures.VCIResumeTests.setUp

    def test_quick_rejects_changed_fits_and_limited_metrics_without_confidence_fit(self):
        from capture_processing.core.vci_pipeline import run_dataset_creation
        from capture_processing.core.vci_resume import quick_subject_complete
        run_dataset_creation(self.args)
        self.args.resume_validation = 'quick'
        path = fixtures.VCIResumeTests.complete_metrics(self)
        self.assertTrue(quick_subject_complete(self.args,self.output))
        data = json.loads(path.read_text())
        data['evaluation_options']['max_frames'] = 1
        path.write_text(json.dumps(data))
        self.assertFalse(quick_subject_complete(self.args,self.output))
        fixtures.VCIResumeTests.complete_metrics(self)
        (self.output/'smpl_params.npz').write_bytes(b'changed')
        self.assertFalse(cached_metrics_complete(self.output,['smplx']))

    def test_standalone_commands_fail_without_requested_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'smpl_params.npz').write_bytes(b'invalid')
            report = root/'fit_metrics.json'
            report.write_text('{"complete":false}')
            with patch('capture_processing.tools.evaluate_fit_metrics.write_fit_metrics',return_value=report):
                self.assertEqual(metrics_main(['--subject-root',tmp]),1)
            before = {p.name:p.read_bytes() for p in root.iterdir()}
            self.assertEqual(validation_main(['--subject-root',tmp]),1)
            self.assertEqual(before,{p.name:p.read_bytes() for p in root.iterdir()})

    def test_normal_pipeline_and_read_only_validator_agree_on_complete_subject(self):
        from capture_processing.core.vci_pipeline import run_dataset_creation
        from capture_processing.core.processing_log import get_missing_steps, load_processing_log
        self.args.mhr_source = 'none'
        self.args.smplx_mhr_options = None
        for key, value in dict(mhr_repo_path=None,mhr_python='python',device='cpu',
                               mhr_max_frames=None,mhr_iterations=100,mhr_force=False).items():
            setattr(self.args,key,value)
        def report(subject, options):
            return fixtures.VCIResumeTests.complete_metrics(self)
        with patch('capture_processing.core.vci_pipeline._create_mhr_params_from_args',side_effect=run_postprocessing), \
             patch('capture_processing.core.vci_pipeline.write_fit_metrics',side_effect=report):
            run_dataset_creation(self.args)
        self.assertEqual(get_missing_steps(load_processing_log(self.output),semantic_enabled=False),[])
        self.assertTrue(json.loads((self.output/'subject_manifest.json').read_text())['complete'])
        before = {p:p.stat().st_mtime_ns for p in self.output.rglob('*') if p.is_file()}
        self.assertEqual(validation_main(['--subject-root',str(self.output),'--deep']),0)
        self.assertEqual(before,{p:p.stat().st_mtime_ns for p in before})

    def test_media_validation_rejects_orphan_cameras_and_invalid_geometry(self):
        from capture_processing.core.vci_pipeline import run_dataset_creation
        from capture_processing.utils.subject_validation import validate_subject
        run_dataset_creation(self.args)
        orphan = self.output / 'masks' / 'orphan'
        orphan.mkdir()
        self.assertIn('media', validate_subject(self.output, ['media'])['errors'])
        orphan.rmdir()
        path = self.output / 'calibration.json'
        original = json.loads(path.read_text())
        for change in ('missing_camera', 'negative_focal', 'nonrigid_rotation'):
            data = json.loads(json.dumps(original))
            if change == 'missing_camera':
                data['cam01'] = data['cam00']
            elif change == 'negative_focal':
                data['cam00']['K'][0] = -1
            else:
                data['cam00']['R'][0] = 2
            path.write_text(json.dumps(data))
            with self.subTest(change=change):
                self.assertIn('media', validate_subject(self.output, ['media'])['errors'])

    def test_visualizations_alone_require_existing_fits_and_rendered_artifacts(self):
        from capture_processing.utils.subject_validation import validate_subject
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.assertIn('visualizations', validate_subject(root, ['visualizations'])['errors'])
            (root / 'mhr_params.npz').touch()
            self.assertIn('visualizations', validate_subject(root, ['visualizations'])['errors'])
            output = root / 'mhr_visualization'
            output.mkdir()
            (output / 'scene.rrd').write_bytes(b'rendering')
            from capture_processing.utils.stage_cache import file_stamp
            manifest = {'complete': True, 'signature': 'saved', 'cameras': [],
                        'artifacts': {'scene.rrd': file_stamp(output / 'scene.rrd')}}
            (output / 'manifest.json').write_text(json.dumps(manifest))
            self.assertTrue(validate_subject(root, ['visualizations'])['complete'])
            (output / 'scene.rrd').unlink()
            self.assertFalse(validate_subject(root, ['visualizations'])['complete'])

    def test_converted_fit_rejects_negative_frame_ids(self):
        import numpy as np
        from capture_processing.utils.subject_validation import converted_mhr_complete
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'fit.npz'
            for frame, complete in ((0, True), (-1, True), (0, 'false')):
                np.savez(path, complete=complete, frame_ids=np.array([frame]),
                         lbs_model_params=np.zeros((1, 204)), identity_coeffs=np.zeros((1, 45)),
                         face_expr_coeffs=np.zeros((1, 72)))
                self.assertEqual(converted_mhr_complete(path), frame == 0 and complete is True)
