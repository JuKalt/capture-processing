"""Run the pinned upstream 3D fitter with process-local confidence corrections.

No upstream checkout edits; the baseline fitting subprocess uses its original
implementation. This adapter only runs for the independently cached extra fit.
"""
from __future__ import annotations
import argparse
import inspect
from pathlib import Path
import runpy
import sys


def install_adapters(threshold=.5, max_reprojection_px=50.):
    import numpy as np
    import torch
    import utils.fitting as fitting
    import utils.optimization as optimization
    from capture_processing.utils.mamma_confidence import triangulate_confident

    original_data = fitting.process_2d_pred
    def filtered_data(body_id, pred_fns, smplx_out, cam_metadata, batch_size, i):
        points, visibility, contacts, floor = original_data(body_id, pred_fns, smplx_out, cam_metadata, batch_size, i)
        points = points.copy()
        visibility = np.ones(points.shape[:2]) if visibility is None else visibility.copy()
        xy = points[..., :2]
        valid = (np.isfinite(points).all(-1) & np.isfinite(visibility) & (visibility >= threshold)
                 & (xy[..., 0] >= 0) & (xy[..., 0] < int(cam_metadata['cam_img_w']))
                 & (xy[..., 1] >= 0) & (xy[..., 1] < int(cam_metadata['cam_img_h'])))
        points[~valid, :2] = 0
        if points.shape[-1] == 3:
            points[~valid, 2] = 1
        visibility[~valid] = 0
        return points, visibility, contacts, floor
    fitting.process_2d_pred = filtered_data

    # Upstream uses spawn for its camera pool; spawned interpreters would lose
    # process-local adapters. Loading cached arrays in this process preserves
    # filtering and avoids launching another GPU/import stack per camera worker.
    original_extract = fitting.extract_data_from_metadata
    extract_signature = inspect.signature(original_extract)
    def extract(*args, **kwargs):
        bound = extract_signature.bind(*args, **kwargs)
        bound.arguments['parallel'] = False
        return original_extract(*bound.args, **bound.kwargs)
    fitting.extract_data_from_metadata = extract

    def triangulate(*args, **kwargs):
        kwargs.update(visibility_threshold=threshold, max_reprojection_px=max_reprojection_px)
        return triangulate_confident(*args, **kwargs)
    optimization.triang_funcs.triangulate_batch = triangulate

    original_projection = optimization.proj_pts_loss
    def projection(pts2d, pts3d_pred, intrinsics, extrinsics, loss='mse',
                   per_point_weight=None, vis_clip_value=.8, weight=1.):
        # Upstream also passes these arguments positionally through its registry.
        return original_projection(pts2d, pts3d_pred, intrinsics, extrinsics,
                                   loss, per_point_weight, 0., weight)
    optimization.proj_pts_loss = projection

    cls = optimization.OptimizeSMPLX
    original_tri = cls.triangulate_3d_from_2d_points
    def triangulated(self, *args, **kwargs):
        points, valid, floor, contact = original_tri(self, *args, **kwargs)
        for body, mask in valid.items():
            points[body] = points[body].masked_fill(~mask[..., None], float('nan'))
        return points, valid, floor, contact
    cls.triangulate_3d_from_2d_points = triangulated

    original_loss = cls.loss_functions
    signature = inspect.signature(original_loss)
    def loss(self, *args, **kwargs):
        bound = signature.bind(self, *args, **kwargs)
        bound.apply_defaults()
        names = list(bound.arguments['losses_names'])
        bound.arguments['losses_names'] = [name for name in names if name != 'l2_loss_3d_points']
        result = original_loss(*bound.args, **bound.kwargs)
        if 'l2_loss_3d_points' in names:
            target = bound.arguments['triangulated_3d_points']
            predicted = bound.arguments['pts3d']
            if target is not None:
                valid = torch.isfinite(target).all(-1)
                residual = predicted[valid] - target[valid]
                value = residual.square().mean() if residual.numel() else predicted.sum() * 0
                result['l2_loss_3d_points'] = float(bound.arguments['weights_cfg']['l2_loss_3d_points']['weight']) * value
        return result
    cls.loss_functions = loss


def main():
    import os
    if os.environ.get('VCI_MAMMA_TRACEBACK_SECONDS'):
        import faulthandler
        faulthandler.dump_traceback_later(int(os.environ['VCI_MAMMA_TRACEBACK_SECONDS']), repeat=True)
    print('Installing confidence-filtered MAMMA adapters', flush=True)
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--confidence-threshold', type=float, default=.5)
    parser.add_argument('--confidence-max-reprojection-px', type=float, default=50.)
    args, remaining = parser.parse_known_args()
    if not 0 < args.confidence_threshold <= 1 or args.confidence_max_reprojection_px <= 0:
        raise SystemExit('Confidence threshold must be in (0,1] and reprojection limit must be positive')
    sys.path.insert(0, str(Path.cwd()))
    install_adapters(args.confidence_threshold, args.confidence_max_reprojection_px)
    sys.argv = [str(Path.cwd() / 'run_ma_3d.py'), *remaining]
    print('Starting upstream 3D fitting with confidence-filtered observations', flush=True)
    runpy.run_path(sys.argv[0], run_name='__main__')


if __name__ == '__main__':
    main()
