import json
import shutil
import unittest
from tests import test_vci_resume as fixtures
from capture_processing.core.vci_pipeline import run_dataset_creation
from capture_processing.utils.subject_manifest import write_subject_manifest


class SubjectManifestTests(unittest.TestCase):
    setUp = fixtures.VCIResumeTests.setUp

    def test_preserves_source_ids_timing_and_relative_products(self):
        source = self.args.input_path/'calibration.json'
        data = json.loads(source.read_text())
        data['meta'] = {'name':'original-recording','capture':{'fps':25}}
        source.write_text(json.dumps(data))
        run_dataset_creation(self.args)
        path = write_subject_manifest(self.output,self.args)
        manifest = json.loads(path.read_text())
        self.assertEqual(manifest['schema_version'],1)
        self.assertEqual(manifest['timeline']['fps'],25)
        self.assertEqual(manifest['timeline']['source_frame_ids'],[0,1])
        self.assertEqual(manifest['timeline']['fit_frame_ids'],[0,1])
        self.assertEqual(manifest['camera_id_mapping'],{'cam00':'A'})
        self.assertTrue(all(not value.startswith('/') for value in manifest['products'].values()))
        copied = self.output.parent/'portable-copy'
        shutil.copytree(self.output,copied)
        for value in manifest['products'].values():
            self.assertTrue((copied/value).exists())
        before = path.stat().st_mtime_ns
        write_subject_manifest(self.output,self.args)
        self.assertEqual(path.stat().st_mtime_ns,before)

    def test_unknown_metadata_and_failed_run_are_explicit(self):
        run_dataset_creation(self.args)
        path = write_subject_manifest(self.output,self.args,result='failed',error='interrupted')
        manifest = json.loads(path.read_text())
        self.assertIsNone(manifest['timeline']['fps'])
        self.assertFalse(manifest['complete'])
        self.assertEqual(manifest['last_run']['error'],'interrupted')

    def test_invalid_source_calibration_still_records_original_failure(self):
        source = self.args.input_path / 'calibration.json'
        for value in ([], None):
            source.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError, 'Source calibration must be a JSON object'):
                run_dataset_creation(self.args)
            manifest = json.loads((self.output / 'subject_manifest.json').read_text())
            self.assertFalse(manifest['complete'])
            self.assertEqual(manifest['last_run']['error'], 'Source calibration must be a JSON object')

    def test_corrupt_fit_archive_does_not_prevent_failure_manifest(self):
        self.output.mkdir(parents=True)
        (self.output / 'smpl_params.npz').write_bytes(b'PK\x03\x04truncated archive')
        path = write_subject_manifest(self.output, self.args, result='failed', error='fit interrupted')
        manifest = json.loads(path.read_text())
        self.assertFalse(manifest['complete'])
        self.assertFalse(manifest['provenance']['fit_archives']['smplx']['readable'])
        self.assertIsNone(manifest['timeline']['fit_frame_ids'])

    def test_backfill_marks_stage_and_refreshes_existing_manifest(self):
        from capture_processing.core.processing_log import load_processing_log, save_processing_log
        from capture_processing.utils.subject_manifest import record_backfill
        run_dataset_creation(self.args)
        log = load_processing_log(self.output)
        log['requested_steps'] = [*log.get('requested_steps', []), 'semantic_parsing']
        log['steps_completed'].pop('semantic_parsing', None)
        save_processing_log(self.output, log)
        write_subject_manifest(self.output, self.args)
        before = json.loads((self.output / 'subject_manifest.json').read_text())
        self.assertEqual(before['stages']['semantic_parsing']['status'], 'pending')
        self.assertFalse(before['complete'])
        record_backfill(self.output, ['semantic_parsing'], 'parse_subject_semantics')
        after = json.loads((self.output / 'subject_manifest.json').read_text())
        self.assertEqual(after['stages']['semantic_parsing']['status'], 'complete')
        self.assertEqual(after['complete'], all(stage['status'] == 'complete' for stage in after['stages'].values()))
        self.assertEqual(after['configuration'], before['configuration'])
        self.assertEqual(after['source'], before['source'])
        backfill, = after['provenance']['backfills']
        self.assertEqual((backfill['operation'], backfill['steps']), ('parse_subject_semantics', ['semantic_parsing']))
        self.assertIn('repository_dirty', after['provenance'])

    def test_backfill_without_manifest_only_marks_log(self):
        from capture_processing.core.processing_log import load_processing_log
        from capture_processing.utils.subject_manifest import record_backfill
        self.output.mkdir(parents=True)
        self.assertIsNone(record_backfill(self.output, ['smplx_confidence'], 'fit_mamma_confidence'))
        self.assertIn('smplx_confidence', load_processing_log(self.output)['steps_completed'])
        self.assertFalse((self.output / 'subject_manifest.json').exists())
