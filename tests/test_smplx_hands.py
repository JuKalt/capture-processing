"""MAMMA hand poses are stored in the flat-hand convention without refitting."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from capture_processing.utils import smplx_hands
from capture_processing.utils.mamma_additional_fit import MANIFEST
from capture_processing.utils.stage_cache import file_stamp


class SmplxHandsTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        self.subject = root / 'subject'
        self.subject.mkdir()
        (self.subject / 'mamma_output').mkdir()
        self.repo = root / 'repo'
        models = self.repo / 'data/body_models/smplx_locked_head/smplx'
        models.mkdir(parents=True)
        self.left = np.linspace(-.5, .5, 45, dtype=np.float32)
        self.right = -self.left
        np.savez(models / 'SMPLX_NEUTRAL.npz', hands_meanl=self.left, hands_meanr=self.right)
        self.path = self.subject / 'smpl_params.npz'
        self.hands = np.random.default_rng(0).normal(size=(3, 45)).astype(np.float32)
        np.savez(self.path, transl=np.zeros((3, 3), np.float32), left_hand_pose=self.hands,
                 right_hand_pose=2 * self.hands, frame_ids=np.arange(3))
        self.quiet = contextlib.redirect_stdout(io.StringIO())
        self.quiet.__enter__()
        self.addCleanup(self.quiet.__exit__, None, None, None)

    def test_conversion_adds_mean_hand_once(self):
        mean = smplx_hands.mamma_hands_mean(self.repo)
        self.assertFalse(smplx_hands.hands_flat(self.path))
        self.assertTrue(smplx_hands.convert_mamma_hands(self.path, mean))
        self.assertFalse(smplx_hands.convert_mamma_hands(self.path, mean))
        with np.load(self.path, allow_pickle=False) as data:
            np.testing.assert_array_equal(data['left_hand_pose'], self.hands + self.left)
            np.testing.assert_array_equal(data['right_hand_pose'], 2 * self.hands + self.right)
            np.testing.assert_array_equal(data['frame_ids'], np.arange(3))
            self.assertEqual(data['left_hand_pose'].dtype, np.float32)
        self.assertTrue(smplx_hands.hands_flat(self.path))

    def write_manifest(self, signature):
        (self.subject / MANIFEST).write_text(json.dumps({'complete': True, 'signature': signature}))

    def signature(self, *args):
        # The real confidence signature includes the baseline export's file stamp.
        return ('baseline', 'config', {}, 'cap', 'landmarks', json.dumps(file_stamp(self.path)))

    def test_current_confidence_fit_is_resigned_not_refit(self):
        self.write_manifest(self.signature()[-1])
        with patch('capture_processing.utils.mamma_additional_fit.inputs', side_effect=self.signature):
            self.assertTrue(smplx_hands.normalize_subject_hands(self.subject, self.repo))
            manifest = json.loads((self.subject / MANIFEST).read_text())
            self.assertEqual(manifest['signature'], self.signature()[-1])
            self.assertFalse(smplx_hands.normalize_subject_hands(self.subject, self.repo))
        self.assertTrue(smplx_hands.hands_flat(self.path))

    def test_stale_confidence_fit_stays_stale(self):
        self.write_manifest('older-inputs')
        with patch('capture_processing.utils.mamma_additional_fit.inputs', side_effect=self.signature):
            smplx_hands.normalize_subject_hands(self.subject, self.repo)
        self.assertEqual(json.loads((self.subject / MANIFEST).read_text())['signature'], 'older-inputs')

    def test_flat_marker_is_numeric_for_array_consumers(self):
        smplx_hands.convert_mamma_hands(self.path, smplx_hands.mamma_hands_mean(self.repo))
        with np.load(self.path, allow_pickle=False) as data:
            # Avatar loaders cast every array in the file to float32.
            converted = {key: data[key].astype(np.float32) for key in data.files}
        self.assertEqual(converted[smplx_hands.MARKER], 1.)


if __name__ == '__main__':
    unittest.main()
