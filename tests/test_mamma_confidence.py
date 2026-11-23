"""Confidence filtering must reject bad views without inventing 3D constraints."""
import unittest
import torch
from capture_processing.utils.mamma_confidence import triangulate_confident


class ConfidenceTriangulationTests(unittest.TestCase):
    def cameras(self, world, shifts=(-1., 0., 1.)):
        K = torch.tensor([[500., 0, 250.], [0, 500., 250.], [0, 0, 1.]], dtype=torch.float64)[None]
        matrices, observations = [], []
        for shift in shifts:
            E = torch.eye(4, dtype=torch.float64)[None]
            E[:, 0, 3] = shift
            camera = world @ E[0, :3, :3].T + E[0, :3, 3]
            projected = camera @ K[0].T
            xy = projected[..., :2] / projected[..., 2:]
            matrices.append(E)
            observations.append(torch.cat([xy, torch.full_like(xy[..., :1], .02)], dim=-1))
        return observations, [K] * len(shifts), matrices

    def test_low_visibility_outlier_does_not_move_the_solution(self):
        world = torch.tensor([[[.1, .2, 3.], [.2, -.1, 4.]]], dtype=torch.float64)
        points, K, E = self.cameras(world)
        points[-1][..., :2] += 1000
        visible = [torch.ones(1, 2), torch.ones(1, 2), torch.full((1, 2), .01)]
        result, valid = triangulate_confident(points, K, E, visible)
        self.assertTrue(valid.all())
        torch.testing.assert_close(result, world, atol=1e-8, rtol=1e-8)

    def test_insufficient_views_and_nonfinite_points_are_invalid(self):
        world = torch.tensor([[[.1, .2, 3.]]], dtype=torch.float64)
        points, K, E = self.cameras(world)
        points[1][...] = float('nan')
        visible = [torch.ones(1, 1), torch.ones(1, 1), torch.zeros(1, 1)]
        result, valid = triangulate_confident(points, K, E, visible)
        self.assertFalse(valid.any())
        self.assertTrue(torch.isfinite(result).all())
        self.assertFalse(result.any())

    def test_geometric_disagreement_cannot_produce_a_valid_target(self):
        world = torch.tensor([[[.1, .2, 3.]]], dtype=torch.float64)
        points, K, E = self.cameras(world)
        for i, p in enumerate(points):
            p[..., 1] += i * 500
        result, valid = triangulate_confident(points, K, E, [torch.ones(1, 1)] * 3,
                                               max_reprojection_px=10.)
        self.assertFalse(valid.any())
        self.assertFalse(result.any())

    def test_behind_camera_solution_is_not_valid(self):
        world = torch.tensor([[[.1, .2, -3.]]], dtype=torch.float64)
        points, K, E = self.cameras(world)
        _, valid = triangulate_confident(points, K, E, [torch.ones(1, 1)] * 3)
        self.assertFalse(valid.any())


if __name__ == '__main__':
    unittest.main()
