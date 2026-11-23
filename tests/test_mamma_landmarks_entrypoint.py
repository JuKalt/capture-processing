"""Exercise the wrapper against both upstream inference APIs without a GPU."""
import argparse
import contextlib
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from capture_processing.utils import mamma_landmarks


class MammaLandmarksEntrypointTests(unittest.TestCase):
    def test_both_upstream_entrypoints_use_patched_processing_and_one_gpu_model(self):
        for name in ['main', 'run_predictions']:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                namespace = {}
                exec('def predict(args, **kwargs):\n'
                     '    assert args.workers == 1\n'
                     '    assert DefaultPredictor_Lazy(None) is None\n'
                     '    return process_data("camera", None, "cuda", "model", "cfg",\n'
                     '                        kwargs["out_folder"], False, kwargs["masks_folder"])\n', namespace)
                args = argparse.Namespace(workers=8, mask_path='masks', seq_name='images',
                                          out_folder='out', img_folder='capture', save_cam_output=False)
                upstream = {name: namespace['predict'], 'parser': lambda: args, 'ViTDetDataset': object()}
                original_path = list(sys.path)
                try:
                    with patch.object(sys, 'argv', ['wrapper', '--workers', '6', '--batch-size', '3']), \
                         patch.object(Path, 'cwd', return_value=Path(directory)), \
                         patch.object(mamma_landmarks.runpy, 'run_path', return_value=upstream), \
                         patch.object(mamma_landmarks, 'process_masked_camera') as process, \
                         contextlib.redirect_stdout(io.StringIO()):
                        mamma_landmarks.main()
                    process.assert_called_once_with('camera', 'cuda', 'model', 'cfg', 'out/images',
                                                    'masks/images', upstream['ViTDetDataset'], 6, 3)
                finally:
                    sys.path[:] = original_path

    def test_unsupported_upstream_reports_actionable_error(self):
        original_path = list(sys.path)
        try:
            with patch.object(sys, 'argv', ['wrapper']), \
                 patch.object(mamma_landmarks.runpy, 'run_path', return_value={}):
                with self.assertRaisesRegex(RuntimeError, 'expected main or run_predictions'):
                    mamma_landmarks.main()
        finally:
            sys.path[:] = original_path


if __name__ == '__main__':
    unittest.main()
