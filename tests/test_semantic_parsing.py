"""Semantic backfills never invoke upstream processing; repairs reuse per-frame inference."""
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image
from capture_processing.utils.semantic_parsing import (SemanticOptions, semantic_complete, inventory,
    ensure_semantic_parsing, options_from_args, add_arguments, MODEL_ID, MODEL_REVISION)
from capture_processing.utils.semantic_fusion import compact, expand, fuse, warp_checked
from capture_processing.utils.semantic_schema import UNKNOWN
from capture_processing.tools.run_semantic_parsing import run


@unittest.skipUnless(importlib.util.find_spec('cv2'), 'Run numerical semantic tests in the sapiens2 or mamma environment')
class SemanticParsingTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.subject = self.root / 'subject'
        self.model = self.root / 'model'
        self.model.mkdir()
        for name in ('config.json', 'preprocessor_config.json', 'model.safetensors'):
            (self.model / name).write_bytes(b'fixture-model')
        (self.model / 'config.json').write_text(json.dumps({
            'architectures': ['Sapiens2ForSemanticSegmentation'],
            'id2label': {i: f'LABEL_{i}' for i in range(29)}}))
        (self.model / 'vci_model.json').write_text(json.dumps({
            'model_id': MODEL_ID, 'revision': MODEL_REVISION, 'label_schema': 'sapiens2_dome29'}))
        self.calls = 0
        self.options = SemanticOptions(model=str(self.model), device='cpu', working_long_edge=64)
        self.image = np.zeros((64, 64, 3), dtype=np.uint8)
        self.image[8:56, 20:44] = 180
        self.mask = np.zeros((64, 64), dtype=np.uint8)
        self.mask[8:56, 20:44] = 255
        for camera in ('cam00', 'cam01'):
            for folder in ('images', 'masks'):
                (self.subject / folder / camera).mkdir(parents=True)
            for frame in range(3):
                # Exercise prefixed VCI images and numeric legacy mask names together.
                Image.fromarray(self.image).save(self.subject / 'images' / camera / f'color_{frame:08d}.png')
                Image.fromarray(self.mask).save(self.subject / 'masks' / camera / f'{frame:08d}.png')
        (self.subject / 'smpl_params.npz').write_bytes(b'untouched legacy fit')
        self.before = (self.subject / 'smpl_params.npz').read_bytes()

    def predictor(self):
        def predict(image, mask, shape):
            self.calls += 1
            p = np.full((*shape, 29), .001, dtype=np.float32)
            p[..., 23] = .972
            return p
        return predict

    def execute(self, options=None):
        return run(self.subject, options or self.options, self.predictor)

    def test_old_subject_backfill_skip_and_single_output_repair(self):
        self.assertFalse(semantic_complete(self.subject, self.options))
        report = self.execute()
        self.assertTrue(report['complete'])
        self.assertEqual(self.calls, 6)
        self.assertTrue(semantic_complete(self.subject, self.options))
        with patch('capture_processing.utils.semantic_parsing.subprocess.run', side_effect=AssertionError('no subprocess needed')):
            ensure_semantic_parsing(self.subject, self.options)
        path = self.subject / 'semantic_parsing/detailed/cam00/00000001.png'
        path.write_bytes(b'damaged PNG')
        self.assertFalse(semantic_complete(self.subject, self.options))
        from capture_processing.tools.run_semantic_parsing import load_raw
        with patch('capture_processing.tools.run_semantic_parsing.load_raw', wraps=load_raw) as read:
            self.execute()
        self.assertEqual(read.call_count, 3)  # damaged view plus its two temporal neighbours only
        self.assertEqual(self.calls, 6)
        self.assertTrue(semantic_complete(self.subject, self.options))
        self.assertEqual((self.subject / 'smpl_params.npz').read_bytes(), self.before)
        with Image.open(path) as im:
            labels = np.array(im)
        self.assertTrue((labels[self.mask == 0] == 0).all())
        self.assertTrue((labels[self.mask > 0] == 23).all())

    def test_checklist_tracks_validated_stage_failure_and_retry(self):
        import subprocess
        from capture_processing.core.processing_log import load_processing_log
        def marked():
            return 'semantic_parsing' in load_processing_log(self.subject)['steps_completed']
        def worker(*args, **kwargs):
            self.execute()
        self.assertFalse(marked())
        with patch('capture_processing.utils.semantic_parsing.subprocess.run', side_effect=worker):
            ensure_semantic_parsing(self.subject, self.options)
        self.assertTrue(marked())
        calls = self.calls
        report = self.subject / 'semantic_parsing/metrics.json'
        report.unlink()
        with patch('capture_processing.utils.semantic_parsing.subprocess.run', side_effect=subprocess.CalledProcessError(1, 'worker')):
            with self.assertRaisesRegex(RuntimeError, 'upstream stages remain reusable'):
                ensure_semantic_parsing(self.subject, self.options)
        self.assertFalse(marked())
        with patch('capture_processing.utils.semantic_parsing.subprocess.run', side_effect=worker):
            ensure_semantic_parsing(self.subject, self.options)
        self.assertTrue(marked())
        self.assertEqual(self.calls, calls)
        self.assertEqual((self.subject / 'smpl_params.npz').read_bytes(), self.before)

    def test_changed_frame_only_reinfers_that_frame_and_repairs_dependencies(self):
        self.execute()
        root = self.subject / 'semantic_parsing'
        untouched = (root / 'detailed/cam01/00000000.png').stat().st_mtime_ns
        Image.new('RGB', (64, 64), 'gray').save(self.subject / 'images/cam00/color_00000001.png')
        self.assertFalse(semantic_complete(self.subject, self.options))
        self.execute()
        self.assertEqual(self.calls, 7)
        # Temporal neighbouring frames in cam00 and same-time cross-view outputs update.
        self.assertEqual((root / 'detailed/cam01/00000000.png').stat().st_mtime_ns, untouched)
        self.assertTrue(semantic_complete(self.subject, self.options))

    def test_threshold_change_and_missing_report_never_repeat_inference(self):
        self.execute()
        options = replace(self.options, confidence=.6)
        self.assertFalse(semantic_complete(self.subject, options))
        self.execute(options)
        self.assertEqual(self.calls, 6)
        (self.subject / 'semantic_parsing/metrics.json').unlink()
        self.assertFalse(semantic_complete(self.subject, options))
        self.execute(options)
        self.assertEqual(self.calls, 6)
        self.assertTrue(semantic_complete(self.subject, options))

    def test_interrupted_export_reuses_inference_on_retry(self):
        with patch('capture_processing.tools.run_semantic_parsing.export', side_effect=RuntimeError('interrupted')):
            with self.assertRaisesRegex(RuntimeError, 'interrupted'):
                self.execute()
        self.assertFalse(semantic_complete(self.subject, self.options))
        self.execute()
        self.assertEqual(self.calls, 6)
        self.assertTrue(semantic_complete(self.subject, self.options))

    def test_corrupt_raw_cache_reinfers_only_one_frame(self):
        self.execute()
        (self.subject / 'semantic_parsing/raw/cam00/00000001.npz').write_bytes(b'bad cache')
        self.execute()
        self.assertEqual(self.calls, 7)
        self.assertTrue(semantic_complete(self.subject, self.options))

    def test_empty_capture_masks_skip_model_and_still_complete(self):
        for path in (self.subject / 'masks').rglob('*.png'):
            Image.new('L', (64, 64), 0).save(path)
        self.execute()
        self.assertEqual(self.calls, 0)
        self.assertTrue(semantic_complete(self.subject, self.options))

    def test_missing_masks_duplicates_and_bad_shapes_fail_explicitly(self):
        path = self.subject / 'masks/cam00/00000000.png'
        path.unlink()
        with self.assertRaisesRegex(ValueError, 'Missing image/mask'):
            self.execute()
        Image.new('L', (32, 32), 255).save(path)
        with self.assertRaisesRegex(ValueError, 'dimension mismatch'):
            self.execute()
        Image.fromarray(self.mask).save(path)
        Image.fromarray(self.mask).save(path.with_name('mask_00000000.png'))
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            inventory(self.subject)

    def test_deep_verification_repairs_invalid_labels_even_with_matching_stamps(self):
        from capture_processing.utils.stage_cache import file_stamp
        self.execute()
        root = self.subject / 'semantic_parsing'
        path = root / 'general/cam00/00000001.png'
        Image.new('L', (64, 64), 0).save(path)
        manifest = json.loads((root / 'manifest.json').read_text())
        manifest['final']['cam00/00000001']['artifacts']['general'] = file_stamp(path)
        (root / 'manifest.json').write_text(json.dumps(manifest))
        self.assertTrue(semantic_complete(self.subject, self.options))
        deep = replace(self.options, verify=True)
        self.assertFalse(semantic_complete(self.subject, deep))
        self.execute(deep)
        self.assertEqual(self.calls, 6)
        self.assertTrue(semantic_complete(self.subject, deep))

    def test_frame_id_gaps_break_temporal_propagation(self):
        for camera in ('cam00', 'cam01'):
            (self.subject / 'images' / camera / 'color_00000001.png').unlink()
            (self.subject / 'masks' / camera / '00000001.png').unlink()
        with patch('capture_processing.tools.run_semantic_parsing.propagate', side_effect=AssertionError('temporal gap')):
            report = self.execute()
        self.assertEqual(self.calls, 4)
        self.assertIsNone(report['raw_temporal_disagreement'])
        self.assertIsNotNone(report['temporal_unavailable'])

    def test_cross_view_hints_require_two_other_views_and_depth_visibility(self):
        from capture_processing.utils.semantic_geometry import GeometryHints
        geometry = self.subject / 'geometry.npz'
        verts = np.zeros((1, 33, 3), dtype=np.float32)
        verts[0, 0], verts[0, 16], verts[0, 32] = [0, 0, 2], [0, 0, 3], [.2, .2, 2]
        np.savez(geometry, pred_vertices=verts)
        camera = {'K': [[20, 0, 32], [0, 20, 32], [0, 0, 1]],
                  'R': np.eye(3).reshape(-1).tolist(), 'T': [0, 0, 0]}
        (self.subject / 'calibration.json').write_text(json.dumps({f'cam{i:02d}': camera for i in range(3)}))
        engine = GeometryHints(self.subject, geometry)
        p = np.zeros((64, 64, 29), dtype=np.float32)
        p[..., 6] = 1
        views = {f'cam{i:02d}': (p, np.ones((64, 64), dtype=bool), (64, 64)) for i in range(3)}
        hints, accepted = engine.frame(0, views)
        self.assertEqual(accepted, 6)  # 2 visible anchors per camera; rear anchor is excluded.
        self.assertEqual(hints['cam00'][0][32, 32], 6)
        hints, accepted = engine.frame(0, dict(list(views.items())[:2]))
        self.assertEqual(accepted, 0)
        self.assertTrue(all(not score.any() for _, score in hints.values()))
        (self.subject / 'run_args.json').write_text(json.dumps({'start_frame': 19}))
        engine = GeometryHints(self.subject, geometry)
        self.assertEqual(engine.frame(0, views), ({}, 0))
        self.assertEqual(engine.frame(19, views)[1], 6)
        geometry.write_bytes(b'corrupt mesh archive')
        engine = GeometryHints(self.subject, geometry)
        self.assertIsNotNone(engine.reason)
        self.assertEqual(engine.frame(19, views), ({}, 0))

    def test_lost_aggregate_manifest_recovers_per_frame_receipts(self):
        self.execute()
        (self.subject / 'semantic_parsing/manifest.json').unlink()
        self.assertFalse(semantic_complete(self.subject, self.options))
        self.execute()
        self.assertEqual(self.calls, 6)
        self.assertTrue(semantic_complete(self.subject, self.options))

    def test_removing_frame_drops_only_its_semantic_outputs(self):
        self.execute()
        for camera in ('cam00', 'cam01'):
            (self.subject / 'images' / camera / 'color_00000001.png').unlink()
            (self.subject / 'masks' / camera / '00000001.png').unlink()
        self.execute()
        self.assertEqual(self.calls, 6)
        self.assertFalse((self.subject / 'semantic_parsing/detailed/cam00/00000001.png').exists())
        self.assertTrue(semantic_complete(self.subject, self.options))

    def test_deep_checks_reinfer_source_edits_with_preserved_timestamps(self):
        import os
        self.execute()
        path = self.subject / 'images/cam00/color_00000001.png'
        stamp = path.stat()
        edited = self.image.copy()
        edited[edited == 180] = 181
        Image.fromarray(edited).save(path)
        os.utime(path, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
        deep = replace(self.options, verify=True)
        self.assertFalse(semantic_complete(self.subject, deep))
        self.execute(deep)
        self.assertEqual(self.calls, 7)
        self.assertTrue(semantic_complete(self.subject, deep))

    def test_completed_cache_and_repairs_do_not_require_local_model_weights(self):
        self.execute()
        elsewhere = replace(self.options, model=str(self.root / 'different-worker-model-location'))
        self.assertTrue(semantic_complete(self.subject, elsewhere))
        (self.subject / 'semantic_parsing/preview/cam00/00000001.png').unlink()
        self.execute(elsewhere)
        self.assertEqual(self.calls, 6)
        self.assertTrue(semantic_complete(self.subject, elsewhere))

    def test_mask_constraint_unknown_and_coarse_mapping(self):
        p = np.full((3, 3, 29), 1/29, dtype=np.float32)
        foreground = np.ones((3, 3), dtype=bool)
        foreground[0, 0] = False
        fine, coarse, confidence, _ = fuse(p, foreground)
        self.assertEqual(fine[0, 0], 0)
        self.assertTrue((fine[foreground] == UNKNOWN).all())
        self.assertTrue((coarse[foreground] == UNKNOWN).all())
        p[1, 1] = 0
        p[1, 1, 6] = 1  # anatomical left hand -> exposed body
        fine, coarse, _, _ = fuse(p, foreground)
        self.assertEqual(fine[1, 1], 6)
        self.assertEqual(coarse[1, 1], 2)

    def test_cycle_and_occlusion_checks_reject_wrong_propagation(self):
        image = np.zeros((32, 32), dtype=np.uint8)
        mask = np.ones_like(image, dtype=bool)
        p = np.zeros((32, 32, 29), dtype=np.float32)
        p[..., 6] = 1
        zero = np.zeros((32, 32, 2), dtype=np.float32)
        warped, valid = warp_checked(image, image, p, mask, mask, zero, zero)
        self.assertTrue(valid.all())
        self.assertTrue((warped.argmax(-1) == 6).all())
        _, valid = warp_checked(image, image, p, mask, mask, zero, zero + 10)
        self.assertFalse(valid.any())
        _, valid = warp_checked(image, image + 100, p, mask, mask, zero, zero)
        self.assertFalse(valid.any())
        _, valid = warp_checked(image, image, p, mask, ~mask, zero, zero)
        self.assertFalse(valid.any())

    def test_bidirectional_consensus_corrects_an_isolated_low_confidence_label(self):
        current = np.zeros((1, 1, 29), dtype=np.float32)
        current[..., 6], current[..., 15] = .45, .55
        neighbour = np.zeros_like(current)
        neighbour[..., 6] = 1
        good = np.ones((1, 1), dtype=bool)
        fine, _, _, _ = fuse(current, good, [(neighbour, good), (neighbour, good)])
        self.assertEqual(fine.item(), 6)

    def test_geometry_checks_report_conflicts_without_overriding_confident_labels(self):
        p = np.zeros((2, 2, 29), dtype=np.float32)
        p[..., 6] = 1
        mask = np.ones((2, 2), dtype=bool)
        hints = (np.full((2, 2), 15, dtype=np.uint8), np.ones((2, 2), dtype=np.float32))
        fine, _, _, diagnostics = fuse(p, mask, hints=hints)
        self.assertTrue((fine == 6).all())
        self.assertEqual(diagnostics['geometry_checked_pixels'], 4)
        self.assertEqual(diagnostics['geometry_disagreements'], 4)
        self.assertEqual(diagnostics['geometry_pixels'], 0)

    def test_compact_posteriors_preserve_uncertainty_and_reject_invalid_data(self):
        p = np.full((2, 2, 29), 1/29, dtype=np.float32)
        ids, scores = compact(p)
        np.testing.assert_allclose(expand(ids, scores), p, atol=1e-4)
        p[0, 0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, 'finite'):
            compact(p)
        scores[0, 0, 0] = np.inf
        with self.assertRaisesRegex(ValueError, 'Invalid cached'):
            expand(ids, scores)


class SemanticOptionsTests(unittest.TestCase):
    def test_default_opt_out_and_invalid_parameters(self):
        import argparse
        parser = argparse.ArgumentParser()
        add_arguments(parser)
        self.assertTrue(options_from_args(parser.parse_args([])).enabled)
        self.assertFalse(options_from_args(parser.parse_args(['--no-semantic-parsing'])).enabled)
        with self.assertRaises(ValueError):
            replace(SemanticOptions(), confidence=2).validate()
        with patch('capture_processing.utils.semantic_parsing.subprocess.run', side_effect=AssertionError('disabled')):
            ensure_semantic_parsing(Path('/nonexistent'), SemanticOptions(enabled=False))
