"""Verify the adapter's input filtering, visibility weights, and masked 3D loss."""
import sys
from types import ModuleType
import unittest
from unittest.mock import patch
import numpy as np
import torch
from capture_processing.tools.run_mamma_confidence import install_adapters


class ConfidenceAdapterTests(unittest.TestCase):
    def fixture(self):
        fitting, optimization = ModuleType('utils.fitting'), ModuleType('utils.optimization')
        package = ModuleType('utils')
        package.fitting, package.optimization = fitting, optimization
        def source(*args):
            return (np.array([[[10., 10., .1], [20., 20., .1], [200., 10., .1]]]),
                    np.array([[.9, .01, .9]]), None, None)
        fitting.process_2d_pred = source
        fitting.extract_data_from_metadata = lambda parallel=True: parallel
        optimization.triang_funcs = ModuleType('triangulation')
        self.floors = []
        optimization.proj_pts_loss = lambda *args: self.floors.append(args[-2])
        class Fitter:
            def triangulate_3d_from_2d_points(self):
                return {0: torch.zeros(1, 2, 3)}, {0: torch.tensor([[True, False]])}, {}, {}
            def loss_functions(self, pts2d, K, extrinsics, pts3d, joints3d_pred, joint_angles,
                               body_shape, losses_names, weights_cfg=None, triangulated_3d_points=None):
                assert 'l2_loss_3d_points' not in losses_names
                return {}
        optimization.OptimizeSMPLX = Fitter
        return package, fitting, optimization

    def test_masks_low_visibility_and_out_of_image_without_mutating_inputs(self):
        package, fitting, optimization = self.fixture()
        with patch.dict(sys.modules, {'utils': package, 'utils.fitting': fitting, 'utils.optimization': optimization}):
            install_adapters()
            self.assertFalse(fitting.extract_data_from_metadata(True))
            points, vis, _, _ = fitting.process_2d_pred(0, None, None,
                {'cam_img_w': 100, 'cam_img_h': 100}, 1, 0)
            np.testing.assert_array_equal(vis, [[.9, 0, 0]])
            np.testing.assert_array_equal(points[0, 1:, :2], 0)
            optimization.proj_pts_loss(None, None, None, None, 'mse', None, .8, 1.)
            self.assertEqual(self.floors, [0.])

    def test_invalid_triangulation_targets_do_not_contribute_to_loss_or_gradient(self):
        package, fitting, optimization = self.fixture()
        with patch.dict(sys.modules, {'utils': package, 'utils.fitting': fitting, 'utils.optimization': optimization}):
            install_adapters()
            fitter = optimization.OptimizeSMPLX()
            target = fitter.triangulate_3d_from_2d_points()[0][0]
            self.assertTrue(torch.isnan(target[0, 1]).all())
            predicted = torch.ones(1, 2, 3, requires_grad=True)
            result = fitter.loss_functions(None, None, None, predicted, None, None, None,
                ['l2_loss_3d_points'], weights_cfg={'l2_loss_3d_points': {'weight': 6}},
                triangulated_3d_points=target)
            loss = result['l2_loss_3d_points']
            self.assertEqual(loss.item(), 6.)
            loss.backward()
            self.assertFalse(predicted.grad[0, 1].any())
            self.assertTrue((predicted.grad[0, 0] == 4).all())


if __name__ == '__main__':
    unittest.main()
