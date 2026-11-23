"""Completion checks, recovery, and converted-fit visualization artifacts."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from capture_processing.utils.stage_cache import rendering_signature, rendering_complete, file_stamp
from capture_processing.utils.mhr import run_mhr_visualization
from capture_processing.utils.fit_metrics import write_fit_metrics
from capture_processing.utils.silhouette_metrics import SilhouetteOptions
from tests import test_silhouette_metrics as fixtures


class StageCacheTests(unittest.TestCase):
    def test_visualization_skips_complete_outputs_and_retries_missing_or_stale_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            np.savez(root / 'mhr_params.npz', complete=True)
            (root / 'calibration.json').write_text('{}')
            output = root / 'mhr_visualization'
            output.mkdir()
            settings = dict(cameras=None, max_frames=120, frame_step=1, fps=30., long_edge=1024, reference_camera=None)
            signature = rendering_signature(root, root, 'mhr', settings)
            artifacts = ['scene.rrd', 'cam00.mp4', 'cam00_preview.png']
            for name in artifacts:
                (output / name).write_bytes(b'output')
            manifest = dict(complete=True, signature=signature, cameras=['cam00'],
                            artifacts={name: file_stamp(output / name) for name in artifacts})
            (output / 'manifest.json').write_text(json.dumps(manifest))
            with patch('capture_processing.utils.silhouette_metrics._repo_stamp', return_value={'revision': None, 'assets': {}}), \
                 patch('capture_processing.utils.mhr.subprocess.run') as run:
                self.assertEqual(run_mhr_visualization(root, root, 'python', 'cpu'), output)
                self.assertFalse(any('-m' in call.args[0] for call in run.call_args_list))
            with patch('capture_processing.utils.silhouette_metrics._repo_stamp', return_value={'revision': None, 'assets': {}}), \
                 patch('capture_processing.utils.mhr.subprocess.run') as run:
                run_mhr_visualization(root, root, 'python', 'cpu', force=True)
                self.assertIn('--force', run.call_args.args[0])
            (output / 'cam00.mp4').unlink()
            self.assertFalse(rendering_complete(output, signature))
            (output / 'cam00.mp4').write_bytes(b'output')
            manifest['artifacts']['cam00.mp4'] = file_stamp(output / 'cam00.mp4')
            (output / 'manifest.json').write_text(json.dumps(manifest))
            np.savez(root / 'mhr_params.npz', complete=True, changed=True)
            self.assertFalse(rendering_complete(output, rendering_signature(root, root, 'mhr', settings)))

    def test_extra_smplx_archive_does_not_invalidate_mhr_rendering_or_estimation(self):
        from capture_processing.utils.mhr import estimation_signature
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            np.savez(root / 'mhr_params.npz', complete=True)
            (root / 'calibration.json').write_text('{}')
            with patch('capture_processing.utils.silhouette_metrics._repo_stamp', return_value={}), \
                 patch('capture_processing.utils.silhouette_metrics._sam_asset_stamps', return_value={}):
                render = rendering_signature(root, root, 'mhr', {})
                estimate = estimation_signature(root, root, None, 100)
                np.savez(root / 'smpl_params_confidence.npz', transl=np.zeros((2, 3)))
                (root / 'smplx_confidence_fit_diagnostics.json').write_text('{}')
                self.assertEqual(render, rendering_signature(root, root, 'mhr', {}))
                self.assertEqual(estimate, estimation_signature(root, root, None, 100))

    def test_semantic_outputs_do_not_invalidate_existing_fit_or_metric_caches(self):
        from capture_processing.utils.mhr import estimation_signature
        from capture_processing.utils.stage_cache import metrics_signature
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixtures.SilhouetteMetricsTests()._fixture(root)
            with patch('capture_processing.utils.silhouette_metrics._repo_stamp', return_value={}), \
                 patch('capture_processing.utils.silhouette_metrics._sam_asset_stamps', return_value={}):
                before = (rendering_signature(root, root, 'mhr', {}),
                          estimation_signature(root, root, None, 100),
                          metrics_signature(root, SilhouetteOptions()))
                folder = root / 'semantic_parsing/raw/cam00'
                folder.mkdir(parents=True)
                np.savez(folder / '00000000.npz', ids=[6], scores=[.9])
                (root / 'semantic_parsing/manifest.json').write_text('{"complete": true}')
                after = (rendering_signature(root, root, 'mhr', {}),
                         estimation_signature(root, root, None, 100),
                         metrics_signature(root, SilhouetteOptions()))
                self.assertEqual(before, after)

    def test_complete_metrics_skip_summaries_and_force_refreshes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixtures.SilhouetteMetricsTests()._fixture(root)
            def silhouettes(subject, fits, options):
                return {'complete_comparison': True}
            with patch('capture_processing.utils.silhouette_metrics.evaluate_shared_silhouettes', side_effect=silhouettes) as evaluate:
                output = write_fit_metrics(root)
                original = output.stat().st_mtime_ns
                with patch('capture_processing.utils.fit_metrics._fit', side_effect=AssertionError('cached summaries recomputed')):
                    write_fit_metrics(root)
                self.assertEqual(evaluate.call_count, 1)
                self.assertEqual(output.stat().st_mtime_ns, original)
                write_fit_metrics(root, SilhouetteOptions(force=True))
                self.assertEqual(evaluate.call_count, 2)
                output.write_text('[]')
                write_fit_metrics(root)
                self.assertEqual(evaluate.call_count, 3)

    def test_incomplete_metrics_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch('capture_processing.utils.silhouette_metrics.evaluate_shared_silhouettes', return_value={'complete_comparison': False}) as evaluate:
                write_fit_metrics(root)
                write_fit_metrics(root)
                self.assertEqual(evaluate.call_count, 2)

    def test_converted_fit_renders_separate_artifacts_and_skips_second_run(self):
        try:
            import cv2  # noqa: F401  # availability probe
            import rerun  # noqa: F401
        except ImportError:
            self.skipTest('Visualization runtime dependencies are unavailable')
        from capture_processing.tools.visualize_mhr import main
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixtures.SilhouetteMetricsTests()._fixture(root)
            vertices = np.array([[0, 0, 2], [1, 0, 2], [0, 1, 2.]])
            faces = np.array([[0, 1, 2]])
            args = ['visualize_mhr', '--subject-root', str(root), '--kind', 'mhrfromsmplx', '--repo-path', str(root), '--device', 'cpu', '--max-frames', '2']
            with patch('sys.argv', args), patch('capture_processing.tools.evaluate_silhouette_fit.mesh_provider', return_value=(lambda frame: (vertices, faces), lambda: None)) as provider:
                main()
                output = root / 'mhr_from_smplx_visualization'
                manifest = json.loads((output / 'manifest.json').read_text())
                self.assertTrue(rendering_complete(output, manifest['signature']))
                self.assertEqual(manifest['frame_ids'], [0, 1])
                self.assertFalse((root / 'mhr_visualization').exists())
                main()
                self.assertEqual(provider.call_count, 1)
                (output / 'cam00.mp4').unlink()
                main()
                self.assertEqual(provider.call_count, 2)
                (root / 'images/cam00/00000000.jpg').unlink()
                with self.assertRaisesRegex(RuntimeError, 'incomplete'):
                    main()
                self.assertFalse((output / 'manifest.json').exists())


class PortableSignatureTests(unittest.TestCase):
    def test_code_and_model_stamps_ignore_location_and_mtime(self):
        import os
        from capture_processing.utils.stage_cache import content_stamp
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'a').mkdir()
            (root / 'b').mkdir()
            for folder, mtime in (('a', 1_000_000), ('b', 2_000_000)):
                (root / folder / 'model.npz').write_bytes(b'weights')
                os.utime(root / folder / 'model.npz', (mtime, mtime))
            self.assertEqual(content_stamp(root / 'a/model.npz'), content_stamp(root / 'b/model.npz'))
            (root / 'b/model.npz').write_bytes(b'other!!')
            self.assertNotEqual(content_stamp(root / 'a/model.npz'), content_stamp(root / 'b/model.npz'))

    def test_fit_digest_ignores_embedded_signatures(self):
        from capture_processing.utils.stage_cache import fit_digest
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'mhr_params.npz'
            np.savez(path, body_pose=np.ones((2, 3)), estimation_signature=np.asarray('old'))
            before = fit_digest(path)
            np.savez(path, body_pose=np.ones((2, 3)), estimation_signature=np.asarray('new'))
            self.assertEqual(fit_digest(path), before)
            np.savez(path, body_pose=np.zeros((2, 3)), estimation_signature=np.asarray('new'))
            self.assertNotEqual(fit_digest(path), before)

    def test_subject_signature_has_no_absolute_paths(self):
        from capture_processing.utils.stage_cache import subject_signature
        with tempfile.TemporaryDirectory() as directory:
            signatures = []
            for name in ('first', 'second'):
                subject = Path(directory) / name / 'subject'
                (subject / 'images' / 'cam00').mkdir(parents=True)
                (subject / 'calibration.json').write_text('{}')
                image = subject / 'images' / 'cam00' / 'color_00000000.jpg'
                image.write_bytes(b'jpeg')
                import os
                os.utime(image, (5, 5))
                np.savez(subject / 'mhr_params.npz', body_pose=np.ones(3))
                signatures.append(subject_signature(subject, {}, media=('images',), fit_files=['mhr_params.npz']))
            self.assertEqual(*signatures)


if __name__ == '__main__':
    unittest.main()
