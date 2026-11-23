"""A fully seeded runtime remains usable when the worker has no network."""

from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from capture_processing.tools.prepare_mhr_runtime import cached_model_paths


class CachedRuntimeTests(unittest.TestCase):
    def test_requires_all_local_files_and_never_queries_network(self):
        class CacheMiss(Exception):
            pass

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot = Mock(return_value=str(root))
            modules = {
                "huggingface_hub": SimpleNamespace(snapshot_download=snapshot),
                "huggingface_hub.errors": SimpleNamespace(LocalEntryNotFoundError=CacheMiss),
            }
            with patch.dict(sys.modules, modules):
                self.assertIsNone(cached_model_paths())
                for name in ("model.ckpt", "model_config.yaml", "assets/mhr_model.pt"):
                    path = root / name
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(b"fixture")
                self.assertEqual(cached_model_paths(), (root / "model.ckpt", root / "assets/mhr_model.pt"))
                snapshot.assert_called_with(repo_id="facebook/sam-3d-body-dinov3", local_files_only=True)
                snapshot.side_effect = CacheMiss
                self.assertIsNone(cached_model_paths())
