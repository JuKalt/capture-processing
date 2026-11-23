"""Verified model deployment and fail-closed worker readiness checks."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from scripts.setup.setup_smplx_model import install


class SmplxModelSetupTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = self.root / 'source.npz'
        self.source.write_bytes(b'controller model')
        self.expected = hashlib.sha256(self.source.read_bytes()).hexdigest()
        self.destination = self.root / 'worker' / 'model.npz'
        output = contextlib.redirect_stdout(io.StringIO())
        output.__enter__()
        self.addCleanup(output.__exit__, None, None, None)

    def test_copy_verifies_and_reuses_model_without_source(self):
        install(self.source, self.destination, self.expected)
        self.assertEqual(self.destination.read_bytes(), b'controller model')
        self.source.unlink()
        install(self.source, self.destination, self.expected)
        install(self.source, self.destination, self.expected, check_only=True)

    def test_bad_source_preserves_existing_destination_and_cleans_partial_copy(self):
        self.destination.parent.mkdir()
        self.destination.write_bytes(b'old model')
        self.source.write_bytes(b'incomplete transfer')
        with self.assertRaisesRegex(ValueError, 'source checksum mismatch'):
            install(self.source, self.destination, self.expected)
        self.assertEqual(self.destination.read_bytes(), b'old model')
        self.assertEqual(list(self.destination.parent.iterdir()), [self.destination])

    def test_readiness_rejects_missing_or_corrupt_model_without_copying(self):
        for content in [None, b'incomplete model']:
            with self.subTest(content=content):
                if content is not None:
                    self.destination.parent.mkdir(exist_ok=True)
                    self.destination.write_bytes(content)
                with self.assertRaisesRegex(ValueError, 'missing or checksum mismatch'):
                    install(self.source, self.destination, self.expected, check_only=True)
                if content is None:
                    self.assertFalse(self.destination.exists())
                else:
                    self.assertEqual(self.destination.read_bytes(), content)

    def test_missing_shared_source_reports_remedy(self):
        with self.assertRaisesRegex(ValueError, 'seed the shared model'):
            install(self.root / 'absent', self.destination, self.expected)
        self.assertFalse(self.destination.exists())

    def test_all_vci_worker_recipes_check_and_install_model(self):
        for path in Path('configs/distribution').glob('vci-2026_*.json'):
            config = json.loads(path.read_text())
            for machine in config['machines']:
                with self.subTest(config=path.name, machine=machine['name']):
                    recipe = machine.get('provision', config['provision'])
                    step = next(step for step in recipe if step['name'] == 'smplx-model')
                    self.assertEqual(step['check'], step['setup'] + ['--check'])


if __name__ == '__main__':
    unittest.main()
