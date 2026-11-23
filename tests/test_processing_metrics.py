"""Successful processing requires every requested fit's silhouette metrics."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from capture_processing.utils.fit_metrics import validate_fit_metrics
from capture_processing.utils.semantic_parsing import SemanticOptions


class ProcessingMetricsTests(unittest.TestCase):
    def test_rejects_missing_invalid_and_incomplete_requested_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'fit_metrics.json'
            fit = {'status': 'complete', 'metrics': {'silhouette_iou': {}, 'silhouette_error': {}}}
            report = {'complete': True, 'silhouette_evaluation': {'complete_comparison': True},
                      'fits': {'smplx': fit}}
            path.write_text(json.dumps(report))
            validate_fit_metrics(path, ['smplx'])
            with self.assertRaisesRegex(RuntimeError, 'Required fit metrics'):
                validate_fit_metrics(path, ['smplx', 'mhr'])
            del fit['metrics']['silhouette_error']
            path.write_text(json.dumps(report))
            with self.assertRaises(RuntimeError):
                validate_fit_metrics(path, ['smplx'])
            for contents in ('[]', 'broken json', '{"complete": false}'):
                path.write_text(contents)
                with self.assertRaises(RuntimeError):
                    validate_fit_metrics(path, ['smplx'])
            path.unlink()
            with self.assertRaises(RuntimeError):
                validate_fit_metrics(path, ['smplx'])

    def test_thuman_metrics_failure_is_fatal_without_confidence_fit(self):
        from capture_processing.core.thuman_pipeline import _process_thuman_subject
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'source'
            source.mkdir()
            report = root / 'fit_metrics.json'
            report.write_text('{"complete": false}')
            kwargs = dict(smplx_source='existing', mamma_repo_path=None, mamma_preset=None,
                          mamma_out_tag='vci', mamma_python='python', mamma_force=False,
                          mhr_source='none', mamma_workers=1, mhr_repo_path=None, mhr_python='python',
                          mhr_max_frames=None, mhr_iterations=100, mhr_force=False, mhr_workers=1,
                          device='cpu', mamma_confidence_fit=False, semantic_options=SemanticOptions(enabled=False))
            with patch('capture_processing.core.thuman_pipeline.ensure_subject_smplx'), \
                 patch('capture_processing.core.thuman_pipeline.convert_subject_smplx_to_mhr'), \
                 patch('capture_processing.core.thuman_pipeline.maybe_create_mhr_params'), \
                 patch('capture_processing.core.thuman_pipeline.write_fit_metrics', return_value=report):
                with self.assertRaisesRegex(RuntimeError, 'Required fit metrics'):
                    _process_thuman_subject(source, root / 'target', **kwargs)
                report.write_text(json.dumps({'complete': True,
                    'silhouette_evaluation': {'complete_comparison': True},
                    'fits': {'smplx': {'status': 'complete', 'metrics': {'silhouette_iou': {}, 'silhouette_error': {}}}}}))
                self.assertEqual(_process_thuman_subject(source, root / 'target', **kwargs), root / 'target')
