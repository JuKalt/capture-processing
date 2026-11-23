import json
from PIL import Image
import unittest
from tests import test_vci_resume as fixtures
from capture_processing.core.vci_pipeline import run_dataset_creation


class ProvenanceTests(unittest.TestCase):
    setUp = fixtures.VCIResumeTests.setUp

    def test_preset_content_change_refits_and_identical_content_reuses(self):
        preset = self.args.input_path / 'preset.yaml'
        preset.write_text('ma_3d:\n  enabled: true\n')
        self.args.mamma_preset = preset
        run_dataset_creation(self.args)
        self.args.mamma_force = False
        run_dataset_creation(self.args)
        self.assertEqual(self.fit_count, 1)
        preset.write_text('ma_3d:\n  enabled: true\n  flags: [--different]\n')
        run_dataset_creation(self.args)
        self.assertEqual(self.fit_count, 2)
        self.assertTrue(self.args.mamma_force)

    def test_legacy_missing_export_forces_upstream_after_source_changes(self):
        run_dataset_creation(self.args)
        path = self.output / '.processing_log.json'
        log = json.loads(path.read_text())
        for key in ('smplx_media_signature', 'smplx_dependency_signature'):
            log.pop(key)
        path.write_text(json.dumps(log))
        (self.output / 'smpl_params.npz').unlink()
        Image.new('RGB',(8,8),'black').save(self.args.input_path/'frame_000000/image/color_A.png')
        run_dataset_creation(self.args)
        self.assertTrue(self.args.mamma_force)
        self.assertEqual(self.fit_count,2)

    def test_canonical_export_preserves_ids_and_rejects_bad_archive_atomically(self):
        import numpy as np
        from capture_processing.utils.smplx_utils import save_mamma_smpl_params
        source = self.args.input_path/'mamma.npz'
        self.output.mkdir(parents=True)
        fields = dict(smplx_pose=np.zeros((2,165)),smplx_betas=np.zeros((1,10)),
                      smplx_translation=np.zeros((2,3)),frame_ids=np.array([2,8]))
        np.savez(source,**fields)
        save_mamma_smpl_params(self.output,source)
        output = self.output/'smpl_params.npz'
        with np.load(output) as data:
            np.testing.assert_array_equal(data['frame_ids'],[2,8])
        previous = output.read_bytes()
        fields['frame_ids'] = np.array([2,2])
        np.savez(source,**fields)
        with self.assertRaisesRegex(ValueError,'frame IDs'):
            save_mamma_smpl_params(self.output,source)
        self.assertEqual(output.read_bytes(),previous)
