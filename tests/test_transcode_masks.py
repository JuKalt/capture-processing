"""Capture masks retain their foreground through legacy-compatible JPEG output."""
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from capture_processing.core.vci_resume import valid_media_pair
from capture_processing.utils.silhouette_metrics import binary_mask
from capture_processing.utils.transcode import transcode_pair


class TranscodeMaskTests(unittest.TestCase):
    def test_binary_byte_and_empty_masks_survive_conversion(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new('RGB', (64, 64), 'white').save(root / 'image.png')
            foreground = np.zeros((64, 64), dtype=np.uint8)
            foreground[13:49, 17:43] = 1
            for value in (0, 1, 255):
                for quality in (90, 95):
                    with self.subTest(value=value, quality=quality):
                        Image.fromarray(foreground * value).save(root / 'mask.png')
                        job = (root / 'image.png', root / 'image.jpg', root / 'mask.png',
                               root / 'mask.jpg', None, quality)
                        self.assertIsNone(transcode_pair(job)[1])
                        with Image.open(root / 'mask.jpg') as image:
                            actual = binary_mask(np.array(image))
                        np.testing.assert_array_equal(actual, foreground.astype(bool) if value else False)
                        self.assertTrue(valid_media_pair(job))

    def test_full_validation_detects_and_repairs_legacy_low_amplitude_mask(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new('RGB', (64, 64), 'white').save(root / 'image.png')
            mask = np.zeros((64, 64), dtype=np.uint8)
            mask[13:49, 17:43] = 1
            Image.fromarray(mask).save(root / 'mask.png')
            Image.fromarray(mask).save(root / 'mask.jpg', quality=95)
            Image.new('RGB', (64, 64), 'white').save(root / 'image.jpg')
            job = (root / 'image.png', root / 'image.jpg', root / 'mask.png', root / 'mask.jpg', None, 95)
            self.assertFalse(valid_media_pair(job))
            self.assertIsNone(transcode_pair(job)[1])
            self.assertTrue(valid_media_pair(job))
            with Image.open(root / 'mask.jpg') as image:
                np.testing.assert_array_equal(binary_mask(np.array(image)), mask.astype(bool))
