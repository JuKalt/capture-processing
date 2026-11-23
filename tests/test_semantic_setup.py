"""Setup checks verify pinned model hashes without updating existing environments."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from scripts.setup.setup_semantic_model import setup
from capture_processing.utils.semantic_parsing import MODEL_FILES, MODEL_ID, MODEL_REVISION


class SemanticSetupTests(unittest.TestCase):
    def test_model_check_detects_corrupt_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            hashes = {}
            for name in MODEL_FILES:
                path = root / name
                path.write_bytes(json.dumps({'architectures': ['Sapiens2ForSemanticSegmentation'],
                    'id2label': {str(i): str(i) for i in range(29)}}).encode() if name == 'config.json' else b'verified fixture')
                hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
            (root / 'vci_model.json').write_text(json.dumps({
                'model_id': MODEL_ID, 'revision': MODEL_REVISION, 'label_schema': 'sapiens2_dome29', 'sha256': hashes}))
            from unittest.mock import patch
            with patch('scripts.setup.setup_semantic_model.MODEL_HASHES', hashes):
                setup(root, check=True)
                (root / 'model.safetensors').write_bytes(b'corrupted')
                with self.assertRaisesRegex(ValueError, 'checksum'):
                    setup(root, check=True)

    def test_check_mode_never_installs_and_checks_model_assets(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            executable = root / 'conda'
            log = root / 'calls.jsonl'
            executable.write_text(f'#!{sys.executable}\nimport json,os,sys\n'
                'with open(os.environ["SEMANTIC_TEST_LOG"],"a") as f:f.write(json.dumps(sys.argv[1:])+"\\n")\n')
            executable.chmod(0o755)
            result = subprocess.run(['bash', 'scripts/setup/setup_semantic_parsing_env.sh', '--check',
                '--model-dir', str(root / 'model with spaces')], env={**os.environ,
                'PATH': str(root) + os.pathsep + os.environ['PATH'], 'SEMANTIC_TEST_LOG': str(log)},
                capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            calls = [json.loads(line) for line in log.read_text().splitlines()]
            self.assertFalse(any('create' in c or 'install' in c for c in calls))
            self.assertIn('--check', calls[-1])
            self.assertIn(str(root / 'model with spaces'), calls[-1])

    def test_thuman_copy_reuses_unchanged_media(self):
        from capture_processing.core.thuman_pipeline import _copy_if_changed
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, target = root / 'input', root / 'output'
            source.write_bytes(b'capture')
            _copy_if_changed(source, target)
            before = target.stat().st_mtime_ns
            from unittest.mock import patch
            with patch('capture_processing.core.thuman_pipeline.shutil.copy2', side_effect=AssertionError('unchanged media copied')):
                _copy_if_changed(source, target)
            self.assertEqual(target.stat().st_mtime_ns, before)
            source.write_bytes(b'updated capture')
            _copy_if_changed(source, target)
            self.assertEqual(target.read_bytes(), source.read_bytes())
