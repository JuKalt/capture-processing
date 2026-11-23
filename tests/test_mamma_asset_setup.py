"""Worker readiness must reject missing and truncated MAMMA asset files."""
import contextlib
import hashlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.setup import setup_mamma_assets as assets


class MammaAssetSetupTests(unittest.TestCase):
    def test_checks_all_assets_repairs_corruption_and_reuses_verified_copies(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, destination = root / 'source', root / 'worker'
            manifest = {}
            for relative in ['weights/checkpoint.pt', 'body_models/vertices.pkl']:
                path = source / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(relative.encode())
                manifest[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
            with patch.object(assets, 'ASSETS', manifest), contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(ValueError, 'checkpoint.pt') as error:
                    assets.setup(source, destination, check_only=True)
                self.assertIn('vertices.pkl', str(error.exception))
                self.assertFalse(destination.exists())
                assets.setup(source, destination)
                (destination / 'weights/checkpoint.pt').write_bytes(b'truncated')
                with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
                    assets.setup(source, destination, check_only=True)
                assets.setup(source, destination)
                for relative in manifest:
                    self.assertEqual((destination / relative).read_bytes(), (source / relative).read_bytes())
                    (source / relative).unlink()
                assets.setup(source, destination)
                assets.setup(source, destination, check_only=True)

    def test_all_smplx_variants_are_required_and_repaired(self):
        models = {name: digest for name, digest in assets.ASSETS.items()
                  if name.startswith('data/body_models/smplx_locked_head/smplx/')}
        self.assertEqual({Path(name).name for name in models},
                         {'SMPLX_NEUTRAL.npz', 'SMPLX_MALE.npz', 'SMPLX_FEMALE.npz'})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, shared, worker = root / 'controller', root / 'shared', root / 'worker'
            manifest = {}
            for name in models:
                path = source / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(name.encode())
                manifest[name] = hashlib.sha256(path.read_bytes()).hexdigest()
            with patch.object(assets, 'ASSETS', manifest), contextlib.redirect_stdout(io.StringIO()):
                assets.setup(source, shared)
                assets.setup(shared, worker)
                for name in models:
                    with self.subTest(model=name):
                        (worker / name).unlink()
                        with self.assertRaisesRegex(ValueError, Path(name).name):
                            assets.setup(shared, worker, check_only=True)
                        assets.setup(shared, worker)
                        self.assertEqual((worker / name).read_bytes(), (source / name).read_bytes())
                        (worker / name).write_bytes(b'truncated')
                        with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
                            assets.setup(shared, worker, check_only=True)
                        assets.setup(shared, worker)
                assets.setup(shared, worker, check_only=True)

if __name__ == '__main__':
    unittest.main()
