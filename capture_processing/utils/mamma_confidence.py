"""Confidence-aware triangulation for the additional MAMMA fit only."""
from __future__ import annotations


def triangulate_confident(pts2d_list, Ks_list, extrs_list, vis_list=None,
                         img_size_px=512, channel_is_logvar=False,
                         visibility_threshold=.5, max_reprojection_px=50., **kwargs):
    """Weighted DLT with finite/depth checks and geometric outlier rejection.

    Preserve MAMMA's (zero-filled points, explicit validity mask) contract. The
    adapter masks invalid targets out of the optimizer's 3D initialization loss.
    """
    import torch
    points = torch.stack(pts2d_list, dim=1)
    dtype, device = points.dtype, points.device
    visibility = (torch.stack(vis_list, dim=1) if vis_list is not None
                  else torch.ones_like(points[..., 0]))
    valid = (torch.isfinite(points).all(-1) & torch.isfinite(visibility)
             & (visibility >= visibility_threshold) & (points[..., :2].sum(-1) > 0))
    projection = torch.stack([K @ E[:, :3] for K, E in zip(Ks_list, extrs_list)], dim=1)
    if not torch.isfinite(projection).all():
        raise ValueError('Nonfinite MAMMA camera matrices')
    xy = torch.nan_to_num(points[..., :2])
    weight = torch.nan_to_num(visibility).clamp(0, 1)
    if points.shape[-1] == 3:
        sigma = (torch.exp(.5 * points[..., 2]) if channel_is_logvar else points[..., 2])
        sigma = torch.nan_to_num(sigma * img_size_px / 2, nan=50., posinf=50.).clamp(1, 50)
        weight = weight / sigma.square()

    def solve(mask):
        scale = torch.sqrt(torch.where(mask, weight, torch.zeros_like(weight)))
        rows = [(xy[..., axis, None] * projection[:, :, None, 2]
                 - projection[:, :, None, axis]) * scale[..., None] for axis in (0, 1)]
        matrix = torch.cat(rows, dim=1).permute(0, 2, 1, 3).contiguous()
        # The pinned MAMMA implementation also uses CPU SVD for this small system.
        _, _, vh = torch.linalg.svd(matrix.cpu(), full_matrices=False)
        homogeneous = vh[..., -1, :].to(device)
        denom = homogeneous[..., 3:]
        ok = (mask.sum(1) >= 2) & (denom[..., 0].abs() > 1e-8)
        world = homogeneous[..., :3] / torch.where(denom.abs() > 1e-8, denom, torch.ones_like(denom))
        ok &= torch.isfinite(world).all(-1)
        return world, ok

    for _ in range(3):
        world, ok = solve(valid)
        homogeneous = torch.cat([world, torch.ones_like(world[..., :1])], dim=-1)
        projected = torch.einsum('fcij,fnj->fcni', projection, homogeneous)
        depth = projected[..., 2]
        uv = projected[..., :2] / torch.where(depth.abs() > 1e-8, depth, torch.ones_like(depth))[..., None]
        residual = torch.linalg.vector_norm(uv - xy, dim=-1)
        retained = valid & (depth > 1e-6) & (residual <= max_reprojection_px) & ok[:, None]
        if torch.equal(retained, valid):
            break
        valid = retained
    world, ok = solve(valid)
    return torch.where(ok[..., None], world, torch.zeros_like(world)).to(dtype), ok
