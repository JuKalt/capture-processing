"""Successful stages survive reruns; damaged stages are repaired independently."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import contextlib
import io
import json
import os
import shutil
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from capture_processing.core.vci_pipeline import run_dataset_creation, _create_mhr_params_from_args
from capture_processing.core.processing_log import load_processing_log
from capture_processing.utils.transcode import transcode_pair
from capture_processing.core.vci_resume import quick_subject_complete


class VCIResumeTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        source = root / 'input' / 'subject'
        for fid in range(2):
            frame = source / f'frame_{fid:06d}'
            for folder, name in [('image', 'color_A.png'), ('mask', 'mask_A.png')]:
                (frame / folder).mkdir(parents=True)
                Image.new('RGB', (8, 8), 'white').save(frame / folder / name)
        (source / 'calibration.json').write_text(json.dumps({'cameras': [{
            'camera_id': 'A', 'intrinsics': {'camera_matrix': np.eye(3).tolist(), 'resolution': [8, 8]},
            'extrinsics': {'view_matrix': np.eye(4).tolist()}}]}))
        self.args = argparse.Namespace(input_path=source, out_dir=root / 'output',
            calibration='calibration.json', drop_cameras=None, resize_long_edge=None,
            jpeg_quality=90, workers=1, skip_images=False, mamma_force=False,
            mamma_confidence_fit=False, semantic_parsing=False)
        self.output = self.args.out_dir / source.name
        self.fit_count = 0
        def fit(args, output, frames):
            self.fit_count += 1
            shapes = {'global_orient': 3, 'transl': 3, 'body_pose': 63, 'jaw_pose': 3,
                      'left_hand_pose': 45, 'right_hand_pose': 45}
            np.savez(output / 'smpl_params.npz', betas=np.zeros((1, 10)),
                     **{key: np.zeros((max(fid for fid, _ in frames) + 1, size)) for key, size in shapes.items()})
        for target, kwargs in [
            ('capture_processing.core.vci_pipeline.ProcessPoolExecutor', {'new': ThreadPoolExecutor}),
            ('capture_processing.core.vci_pipeline.create_smpl_params', {'side_effect': fit}),
            ('capture_processing.core.vci_pipeline._create_mhr_params_from_args', {})]:
            mock = patch(target, **kwargs)
            mock.start()
            self.addCleanup(mock.stop)
        quiet = contextlib.redirect_stdout(io.StringIO())
        quiet.__enter__()
        self.addCleanup(quiet.__exit__, None, None, None)

    def complete_metrics(self):
        from capture_processing.utils.stage_cache import fit_artifacts, metric_artifacts, metrics_signature
        from capture_processing.utils.silhouette_metrics import options_from_pipeline
        path = self.output / 'fit_metrics.json'
        path.write_text(json.dumps({'complete': True,
            'silhouette_evaluation': {'complete_comparison': True},
            'fits': {'smplx': {'status':'complete', 'metrics': {'silhouette_iou':{}, 'silhouette_error':{}}}},
            'evaluation_options': {'cameras':None, 'max_frames':None},
            'fit_artifacts': fit_artifacts(self.output), 'cache_artifacts':metric_artifacts(self.output),
            'input_signature': metrics_signature(self.output, options_from_pipeline(self.args))}))
        return path

    def test_second_run_reuses_all_stages_and_repairs_only_corrupt_media(self):
        run_dataset_creation(self.args)
        image = self.output / 'images/cam00/color_00000000.jpg'
        original = image.stat().st_mtime_ns
        with patch('capture_processing.core.vci_pipeline.transcode_pair', wraps=transcode_pair) as transcode:
            run_dataset_creation(self.args)
            transcode.assert_not_called()
            (self.output / 'masks/cam00/mask_00000001.jpg').write_bytes(b'corrupt')
            run_dataset_creation(self.args)
            self.assertEqual(transcode.call_count, 1)
        self.assertEqual(image.stat().st_mtime_ns, original)
        self.assertEqual(self.fit_count, 1)

    def test_transcode_failure_is_not_complete_and_retry_reuses_successful_pairs(self):
        def failing(job):
            return (str(job[1]), 'failed') if '00000001' in job[1].name else transcode_pair(job)
        with patch('capture_processing.core.vci_pipeline.transcode_pair', side_effect=failing):
            with self.assertRaisesRegex(SystemExit, 'remains incomplete'):
                run_dataset_creation(self.args)
        self.assertNotIn('images_transcoded', load_processing_log(self.output)['steps_completed'])
        with patch('capture_processing.core.vci_pipeline.transcode_pair', wraps=transcode_pair) as transcode:
            run_dataset_creation(self.args)
            self.assertEqual(transcode.call_count, 1)

    def test_missing_calibration_and_smplx_are_repaired_despite_completion_markers(self):
        run_dataset_creation(self.args)
        (self.output / 'calibration.json').unlink()
        (self.output / 'smpl_params.npz').write_bytes(b'broken')
        with patch('capture_processing.core.vci_pipeline.transcode_pair') as transcode:
            run_dataset_creation(self.args)
            transcode.assert_not_called()
        self.assertEqual(self.fit_count, 2)
        self.assertIn('cam00', json.loads((self.output / 'calibration.json').read_text()))

    def test_changed_settings_invalidate_media_and_smplx_even_after_interruption(self):
        run_dataset_creation(self.args)
        self.args.jpeg_quality = 80
        with patch('capture_processing.core.vci_pipeline.create_smpl_params', side_effect=SystemExit('interrupted')):
            with self.assertRaises(SystemExit):
                run_dataset_creation(self.args)
        self.args.mamma_force = False
        run_dataset_creation(self.args)
        self.assertTrue(self.args.mamma_force)
        self.assertEqual(self.fit_count, 2)
        self.assertNotIn('smplx_force_pending', load_processing_log(self.output))

    def test_touching_one_source_only_checks_that_pair_and_preserves_fits(self):
        run_dataset_creation(self.args)
        before = {p: p.stat().st_mtime_ns for folder in ('images', 'masks')
                  for p in (self.output / folder).rglob('*.jpg')}
        source = self.args.input_path / 'frame_000000/image/color_A.png'
        stat = source.stat()
        os.utime(source, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
        with patch('capture_processing.core.vci_pipeline.transcode_pair', wraps=transcode_pair) as transcode:
            run_dataset_creation(self.args)
            self.assertEqual(transcode.call_count, 1)
        self.assertEqual(self.fit_count, 1)
        self.assertEqual(before, {p: p.stat().st_mtime_ns for p in before})

    def test_changed_image_or_mask_only_transcodes_its_pair(self):
        for folder, filename in (('image', 'color_A.png'), ('mask', 'mask_A.png')):
            with self.subTest(folder=folder):
                run_dataset_creation(self.args)
                self.args.mamma_force = False
                original_fits = self.fit_count
                untouched = {p: p.stat().st_mtime_ns for kind in ('images', 'masks')
                             for p in (self.output / kind).rglob('*00000001.jpg')}
                Image.new('RGB', (8, 8), 'black').save(self.args.input_path / 'frame_000000' / folder / filename)
                with patch('capture_processing.core.vci_pipeline.transcode_pair', wraps=transcode_pair) as transcode:
                    run_dataset_creation(self.args)
                    self.assertEqual(transcode.call_count, 1)
                self.assertEqual(self.fit_count, original_fits + 1)
                self.assertEqual(untouched, {p: p.stat().st_mtime_ns for p in untouched})
                self.args.mamma_force = False

    def test_settings_change_interruption_reuses_each_successful_pair(self):
        run_dataset_creation(self.args)
        self.args.jpeg_quality = 75
        def failing(job):
            return (str(job[1]), 'interrupted') if '00000001' in job[1].name else transcode_pair(job)
        with patch('capture_processing.core.vci_pipeline.transcode_pair', side_effect=failing):
            with self.assertRaisesRegex(SystemExit, 'remains incomplete'):
                run_dataset_creation(self.args)
        self.assertFalse(quick_subject_complete(self.args, self.output))
        successful = self.output / 'images/cam00/color_00000000.jpg'
        timestamp = successful.stat().st_mtime_ns
        with patch('capture_processing.core.vci_pipeline.transcode_pair', wraps=transcode_pair) as transcode:
            run_dataset_creation(self.args)
            self.assertEqual(transcode.call_count, 1)
        self.assertEqual(successful.stat().st_mtime_ns, timestamp)
        self.assertEqual(self.fit_count, 2)
        self.assertNotIn('media_update_pending', load_processing_log(self.output))

    def test_legacy_log_adopts_valid_media_without_reprocessing(self):
        run_dataset_creation(self.args)
        shutil.rmtree(self.output / '.media_state')
        path = self.output / '.processing_log.json'
        log = json.loads(path.read_text())
        del log['smplx_media_signature']
        path.write_text(json.dumps(log))
        with patch('capture_processing.core.vci_pipeline.transcode_pair', wraps=transcode_pair) as transcode:
            run_dataset_creation(self.args)
            transcode.assert_not_called()
        self.assertEqual(self.fit_count, 1)
        self.assertEqual(len(list((self.output / '.media_state').rglob('*.json'))), 2)

    def test_legacy_run_with_new_frame_invalidates_old_mamma_outputs(self):
        run_dataset_creation(self.args)
        shutil.rmtree(self.output / '.media_state')
        log_path = self.output / '.processing_log.json'
        log = json.loads(log_path.read_text())
        del log['smplx_media_signature']
        log_path.write_text(json.dumps(log))
        shutil.copytree(self.args.input_path / 'frame_000001', self.args.input_path / 'frame_000002')
        run_dataset_creation(self.args)
        self.assertTrue(self.args.mamma_force)
        self.assertEqual(self.fit_count, 2)
        with np.load(self.output / 'smpl_params.npz') as fit:
            self.assertEqual(len(fit['transl']), 3)

    def test_interrupted_pair_checkpoint_cannot_quick_skip_stale_fit(self):
        from capture_processing.core.vci_resume import record_media_pair
        run_dataset_creation(self.args)
        Image.new('RGB', (8, 8), 'black').save(self.args.input_path / 'frame_000000/image/color_A.png')
        def checkpoint(*args):
            record_media_pair(*args)
            if '00000000' in args[1][1].name:
                raise RuntimeError('interrupted after successful pair')
        with patch('capture_processing.core.vci_pipeline.record_media_pair', side_effect=checkpoint):
            with self.assertRaisesRegex(RuntimeError, 'interrupted after'):
                run_dataset_creation(self.args)
        self.args.resume_validation = 'quick'
        self.assertFalse(quick_subject_complete(self.args, self.output))
        with patch('capture_processing.core.vci_pipeline.transcode_pair', wraps=transcode_pair) as transcode:
            run_dataset_creation(self.args)
            transcode.assert_not_called()
        self.assertEqual(self.fit_count, 2)
        self.assertNotIn('media_update_pending', load_processing_log(self.output))

    def test_removed_source_pair_only_removes_stale_outputs(self):
        run_dataset_creation(self.args)
        (self.args.input_path / 'frame_000000/image/color_A.png').unlink()
        with patch('capture_processing.core.vci_pipeline.transcode_pair', wraps=transcode_pair) as transcode:
            run_dataset_creation(self.args)
            transcode.assert_not_called()
        self.assertFalse((self.output / 'images/cam00/color_00000000.jpg').exists())
        self.assertFalse((self.output / 'masks/cam00/mask_00000000.jpg').exists())
        self.assertFalse((self.output / '.media_state/cam00/color_00000000.json').exists())
        from capture_processing.utils.subject_validation import validate_subject
        self.assertTrue(validate_subject(self.output, ['media','smplx'], deep=True)['complete'])
        self.assertEqual(self.fit_count, 2)

    def test_metrics_failure_without_confidence_fitting_fails_and_cached_retry_succeeds(self):
        run_dataset_creation(self.args)
        for name, value in dict(mhr_repo_path=None, mhr_python=None, device='cpu',
                                mhr_max_frames=None, mhr_iterations=100, mhr_force=False).items():
            setattr(self.args, name, value)
        report = self.output / 'fit_metrics.json'
        report.write_text('{"complete": false}')
        with patch('capture_processing.core.vci_pipeline._create_mhr_params_from_args', side_effect=_create_mhr_params_from_args), \
             patch('capture_processing.core.vci_pipeline.convert_subject_smplx_to_mhr'), \
             patch('capture_processing.core.vci_pipeline.maybe_create_mhr_params'), \
             patch('capture_processing.core.vci_pipeline.write_fit_metrics', return_value=report), \
             patch('capture_processing.utils.semantic_parsing.ensure_semantic_parsing') as semantic:
            with self.assertRaisesRegex(RuntimeError, 'Required fit metrics are incomplete'):
                run_dataset_creation(self.args)
            semantic.assert_not_called()
            report.write_text(json.dumps({'complete': True,
                'silhouette_evaluation': {'complete_comparison': True},
                'fits': {'smplx': {'status': 'complete', 'metrics': {'silhouette_iou': {}, 'silhouette_error': {}}}}}))
            with patch('capture_processing.core.vci_pipeline.transcode_pair') as transcode:
                run_dataset_creation(self.args)
                transcode.assert_not_called()
            semantic.assert_called_once()
        self.assertEqual(self.fit_count, 1)

    def test_quick_resume_skips_media_scan_but_runs_postprocessing(self):
        run_dataset_creation(self.args)
        self.args.resume_validation = 'quick'
        with patch('capture_processing.core.vci_pipeline.build_transcoding_jobs', side_effect=AssertionError('media scan')), \
             patch('capture_processing.core.vci_pipeline._create_mhr_params_from_args') as postprocess:
            run_dataset_creation(self.args)
        postprocess.assert_called_once()

    def test_calibration_repair_invalidates_legacy_quick_resume_and_refits(self):
        from capture_processing.core.vci_resume import quick_conversion_complete
        run_dataset_creation(self.args)
        log_path = self.output / '.processing_log.json'
        log = json.loads(log_path.read_text())
        log.pop('quick_resume_settings', None)
        log.pop('smplx_media_signature', None)
        log_path.write_text(json.dumps(log))
        self.args.resume_validation = 'quick'
        self.args.jpeg_quality = 95
        source = self.args.input_path / self.args.calibration
        data = json.loads(source.read_text())
        data['cameras'][0]['intrinsics']['camera_matrix'][0][0] = 3.0
        source.write_text(json.dumps(data))
        self.assertFalse(quick_conversion_complete(self.args, self.output, log))
        run_dataset_creation(self.args)
        self.assertEqual(self.fit_count, 2)
        self.assertTrue(self.args.mamma_force)
        self.assertEqual(json.loads((self.output / 'calibration.json').read_text())['cam00']['K'][0], 3.0)
        self.args.mamma_force = False
        self.assertTrue(quick_conversion_complete(self.args, self.output, load_processing_log(self.output)))

    def test_obsolete_source_cannot_quick_skip_or_reprocess(self):
        run_dataset_creation(self.args)
        source = self.args.input_path / self.args.calibration
        data = json.loads(source.read_text())
        data['meta'] = {'name': '2026_10_01_cvpr-dataset_036',
                        'calibration': {'timestamp': '2025-10-22T15:50:00+02:00'}}
        source.write_text(json.dumps(data))
        self.args.resume_validation = 'quick'
        self.assertFalse(quick_subject_complete(self.args, self.output))
        with self.assertRaisesRegex(ValueError, 'Known obsolete'):
            run_dataset_creation(self.args)
        self.assertEqual(self.fit_count, 1)

    def test_quick_resume_changed_settings_fall_back_to_validation(self):
        run_dataset_creation(self.args)
        self.args.resume_validation = 'quick'
        self.args.jpeg_quality = 80
        with patch('capture_processing.core.vci_pipeline.transcode_pair', wraps=transcode_pair) as transcode:
            run_dataset_creation(self.args)
        self.assertEqual(transcode.call_count, 2)
        self.assertEqual(self.fit_count, 2)

    def test_quick_subject_requires_complete_metrics_and_requested_fits(self):
        run_dataset_creation(self.args)
        self.args.resume_validation = 'quick'
        self.assertFalse(quick_subject_complete(self.args, self.output))
        self.complete_metrics()
        self.assertTrue(quick_subject_complete(self.args, self.output))
        self.args.mhr_source = 'sam3d-body-rerun'
        self.assertFalse(quick_subject_complete(self.args, self.output))
        np.savez(self.output / 'mhr_params.npz', complete=np.asarray(False))
        self.assertFalse(quick_subject_complete(self.args, self.output))

    def test_quick_subject_requires_successful_current_manifest(self):
        run_dataset_creation(self.args)
        self.args.resume_validation = 'quick'
        self.complete_metrics()
        self.assertTrue(quick_subject_complete(self.args, self.output))
        path = self.output / 'subject_manifest.json'
        original = json.loads(path.read_text())
        for value in ([], dict(original, complete=False), dict(original, schema_version=0)):
            path.write_text(json.dumps(value))
            self.assertFalse(quick_subject_complete(self.args, self.output))
        path.unlink()
        self.assertFalse(quick_subject_complete(self.args, self.output))
        from capture_processing.core.vci_resume import quick_conversion_complete
        self.assertTrue(quick_conversion_complete(self.args, self.output, load_processing_log(self.output)))
        with patch('capture_processing.core.vci_pipeline.create_smpl_params', side_effect=AssertionError('refit')):
            run_dataset_creation(self.args)
        self.assertTrue(path.is_file())
        self.assertTrue(quick_subject_complete(self.args, self.output))

    def test_new_semantic_stage_is_required_without_invalidating_conversion(self):
        from capture_processing.core.vci_resume import quick_conversion_complete
        run_dataset_creation(self.args)
        self.args.resume_validation = 'quick'
        self.args.semantic_parsing = True
        self.complete_metrics()
        log = load_processing_log(self.output)
        self.assertTrue(quick_conversion_complete(self.args, self.output, log))
        self.assertFalse(quick_subject_complete(self.args, self.output))
        from capture_processing.core.processing_log import set_semantic_complete
        set_semantic_complete(self.output, True)
        # A stale checklist marker cannot substitute for validated artifacts.
        self.assertFalse(quick_subject_complete(self.args, self.output))
        with patch('capture_processing.utils.semantic_parsing.semantic_complete', return_value=True) as check:
            self.assertTrue(quick_subject_complete(self.args, self.output))
            check.assert_called_once()
        self.assertEqual(self.fit_count, 1)

    def test_legacy_process_dispatches_missing_semantics_without_reconversion(self):
        from capture_processing.core.processing_log import get_missing_steps, set_semantic_complete
        run_dataset_creation(self.args)
        self.assertEqual(get_missing_steps(load_processing_log(self.output)), ['semantic_parsing'])
        self.assertEqual(get_missing_steps(load_processing_log(self.output), semantic_enabled=False), [])
        self.args.resume_validation = 'quick'
        self.args.semantic_parsing = True
        for name, value in dict(mhr_repo_path=None, mhr_python=None, device='cpu',
                                mhr_max_frames=None, mhr_iterations=100, mhr_force=False).items():
            setattr(self.args, name, value)
        metric_report = self.output / 'fit_metrics.json'
        metric_report.write_text(json.dumps({'complete': True,
            'silhouette_evaluation': {'complete_comparison': True},
            'fits': {'smplx': {'status': 'complete', 'metrics': {'silhouette_iou': {}, 'silhouette_error': {}}}}}))
        before = {p: p.stat().st_mtime_ns for p in self.output.rglob('*') if p.is_file()}
        def semantics(subject, args):
            self.assertEqual(subject, self.output)
            self.assertTrue(args.semantic_parsing)
            set_semantic_complete(subject, True)
        with patch('capture_processing.core.vci_pipeline._create_mhr_params_from_args', side_effect=_create_mhr_params_from_args), \
             patch('capture_processing.core.vci_pipeline.build_transcoding_jobs', side_effect=AssertionError('media scan')), \
             patch('capture_processing.core.vci_pipeline.create_smpl_params', side_effect=AssertionError('default fit')), \
             patch('capture_processing.utils.mamma_additional_fit.ensure_confidence_fit'), \
             patch('capture_processing.core.vci_pipeline.convert_subject_smplx_to_mhr'), \
             patch('capture_processing.core.vci_pipeline.maybe_create_mhr_params'), \
             patch('capture_processing.core.vci_pipeline.write_fit_metrics', return_value=metric_report), \
             patch('capture_processing.utils.semantic_parsing.ensure_semantic_parsing', side_effect=semantics) as semantic:
            run_dataset_creation(self.args)
            semantic.assert_called_once()
        self.assertEqual(get_missing_steps(load_processing_log(self.output)), [])
        for path, timestamp in before.items():
            if path.name not in ('.processing_log.json', 'subject_manifest.json'):
                self.assertEqual(path.stat().st_mtime_ns, timestamp)
        self.assertEqual(self.fit_count, 1)

    def test_quick_subject_force_and_missing_calibration_prevent_skip(self):
        run_dataset_creation(self.args)
        self.args.resume_validation = 'quick'
        self.complete_metrics()
        self.args.mamma_force = True
        self.assertFalse(quick_subject_complete(self.args, self.output))
        self.args.mamma_force = False
        (self.output / 'calibration.json').unlink()
        self.assertFalse(quick_subject_complete(self.args, self.output))
