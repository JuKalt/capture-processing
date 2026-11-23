"""The CVPR geometry correction is reversible and does not falsify media history."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from capture_processing.utils.calibration_repair import (
    OLD_TIMESTAMP, REFERENCE_TIMESTAMP, ZOOM_CAMERAS,
    repair_calibrations, validate_source_calibration,
)


class CalibrationRepairTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.donor = {'meta': {'name': 'reference', 'calibration': {'timestamp': REFERENCE_TIMESTAMP}},
                      'cameras': [{'camera_id': cid, 'intrinsics': {
                          'resolution': [4608, 5328], 'camera_matrix': [12830, 0, 2304, 0, 12830, 2664, 0, 0, 1]}}
                                  for cid in sorted(ZOOM_CAMERAS)]}
        self.reference = self.root / 'reference.json'
        self.reference.write_text(json.dumps(self.donor))
        self.paths = []
        for n in range(36, 49):
            name = f'2026_10_08_cvpr-dataset_{n:03}'
            path = self.root / '2026_10_08' / name / 'calibration_dome.json'
            path.parent.mkdir(parents=True)
            data = copy.deepcopy(self.donor)
            data['meta'].update(name=name, capture={'number_frames': 252})
            data['meta']['calibration']['timestamp'] = OLD_TIMESTAMP
            data['raw_calibration'] = ['original calibration history']
            data['processing_applied'] = ['original undistortion history']
            for camera in data['cameras']:
                camera['intrinsics']['camera_matrix'][0] = 4570
            path.write_text(json.dumps(data))
            self.paths.append(path)
        self.originals = {p: p.read_bytes() for p in self.paths}

    def test_dry_run_apply_backup_and_idempotence(self):
        self.assertTrue(all(r['status'] == 'planned' for r in repair_calibrations(self.root, self.reference)))
        self.assertEqual(self.originals, {p: p.read_bytes() for p in self.paths})
        results = repair_calibrations(self.root, self.reference, apply=True)
        for result, path in zip(results, self.paths):
            updated = json.loads(path.read_text())
            original = json.loads(self.originals[path])
            self.assertEqual(updated['cameras'], self.donor['cameras'])
            for key in ('processing_applied', 'raw_calibration'):
                self.assertEqual(updated[key], original[key])
            self.assertEqual(updated['meta']['capture'], original['meta']['capture'])
            self.assertFalse(updated['calibration_repair']['media_reprocessed'])
            self.assertEqual(Path(result['backup']).read_bytes(), self.originals[path])
            validate_source_calibration(updated)
        repaired = {p: p.read_bytes() for p in self.paths}
        self.assertTrue(all(r['status'] == 'already_repaired' for r in repair_calibrations(self.root, self.reference, apply=True)))
        self.assertEqual(repaired, {p: p.read_bytes() for p in self.paths})

    def test_staging_keeps_originals_and_retains_reference_provenance(self):
        stage = self.root / 'staged'
        results = repair_calibrations(self.root, self.reference, stage_dir=stage)
        self.assertEqual(self.originals, {p: p.read_bytes() for p in self.paths})
        self.assertFalse(list((self.root / '2026_10_08').rglob('*.bak')))
        for row in results:
            data = json.loads(Path(row['staged_path']).read_text())
            self.assertEqual(row['status'], 'staged')
            self.assertEqual(data['cameras'], self.donor['cameras'])
            self.assertEqual(data['calibration_repair']['reference'], str(self.reference))

    def test_bad_target_aborts_entire_preflight(self):
        data = json.loads(self.paths[-1].read_text())
        data['cameras'][0]['intrinsics']['resolution'] = [1, 1]
        self.paths[-1].write_text(json.dumps(data))
        before = {p: p.read_bytes() for p in self.paths}
        with self.assertRaisesRegex(ValueError, 'resolution differs'):
            repair_calibrations(self.root, self.reference, apply=True)
        self.assertEqual(before, {p: p.read_bytes() for p in self.paths})
        self.assertFalse(list(self.root.rglob('*.bak')))

    def test_bad_reference_rejected(self):
        self.donor['meta']['calibration']['timestamp'] = OLD_TIMESTAMP
        self.reference.write_text(json.dumps(self.donor))
        with self.assertRaisesRegex(ValueError, 'July 2026'):
            repair_calibrations(self.root, self.reference, apply=True)

    def test_only_confirmed_obsolete_subjects_are_rejected(self):
        data = json.loads(self.paths[0].read_text())
        with self.assertRaisesRegex(ValueError, 'Known obsolete'):
            validate_source_calibration(data)
        data['meta']['name'] = '2026_09_17_cvpr-dataset_035'
        validate_source_calibration(data)
