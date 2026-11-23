"""Conservative sparse cross-view semantic hints from existing fitted geometry.

These hints cannot change the capture foreground. Vertex depth splats reject
self-occluded anchors; only agreement from at least two OTHER cameras is used.
"""
from __future__ import annotations
import json
import zipfile
from pathlib import Path
import numpy as np


def geometry_path(subject):
    for name in ('mamma_confidence_output', 'mamma_output'):
        root = Path(subject) / name / 'ma_3d'
        files = sorted(root.glob('*/*/images/verts_joints_body_id-00.npz'))
        if len(files) == 1:
            return files[0]
    return None


class GeometryHints:
    def __init__(self, subject, path):
        self.reason = None
        self.vertices = None
        try:
            calibration = Path(subject) / 'calibration.json'
            if path is None or not calibration.is_file():
                raise ValueError('No existing MAMMA mesh/calibration; using temporal fusion only')
            self.calibration = json.loads(calibration.read_text())
            if not isinstance(self.calibration, dict):
                raise ValueError('Invalid existing calibration; using temporal fusion only')
            for camera in self.calibration.values():
                for key, count in (('K', 9), ('R', 9), ('T', 3)):
                    values = np.asarray(camera[key], dtype=float)
                    if values.size != count or not np.isfinite(values).all():
                        raise ValueError('Invalid existing calibration; using temporal fusion only')
            with np.load(path, allow_pickle=False) as data:
                self.vertices = data['pred_vertices']
                ids = data['frame_ids'] if 'frame_ids' in data else None
            if (self.vertices.ndim != 3 or self.vertices.shape[-1] != 3
                    or len(self.vertices) == 0 or self.vertices.shape[1] < 3
                    or not np.isfinite(self.vertices).all()):
                raise ValueError('Existing geometry is invalid; using temporal fusion only')
            if ids is None:
                arguments = Path(path).parent / 'run_args.json'
                offset = int(json.loads(arguments.read_text()).get('start_frame') or 0) if arguments.is_file() else 0
                ids = np.arange(len(self.vertices)) + offset
            if np.asarray(ids).shape != (len(self.vertices),) or len(np.unique(ids)) != len(ids):
                raise ValueError('Invalid existing geometry frame IDs')
            self.indices = {int(frame): i for i, frame in enumerate(ids)}
        except (OSError, ValueError, KeyError, TypeError, zipfile.BadZipFile) as exc:
            self.vertices = None
            self.reason = str(exc)

    def frame(self, frame_id, views):
        import cv2
        if self.vertices is None or frame_id not in self.indices:
            return {}, 0
        vertices = self.vertices[self.indices[frame_id]]
        anchors = np.arange(0, len(vertices), 16)
        samples = {}
        for camera, (posterior, foreground, native_shape) in views.items():
            if camera not in self.calibration:
                continue
            c = self.calibration[camera]
            K = np.asarray(c['K'], dtype=float).reshape(3, 3).copy()
            h, w = foreground.shape
            nh, nw = native_shape
            K[0] *= w / nw
            K[1] *= h / nh
            points = vertices @ np.asarray(c['R']).reshape(3, 3).T + np.asarray(c['T']).reshape(3)
            projection = points @ K.T
            depth = points[:, 2]
            uv = projection[:, :2] / np.where(np.abs(depth) > 1e-8, depth, 1)[:, None]
            pixels = np.rint(uv).clip(-1e8, 1e8).astype(np.int64)
            inside = ((depth > 1e-5) & (pixels[:, 0] >= 0) & (pixels[:, 0] < w)
                      & (pixels[:, 1] >= 0) & (pixels[:, 1] < h))
            zbuffer = np.full((h, w), np.inf, dtype=np.float32)
            np.minimum.at(zbuffer, (pixels[inside, 1], pixels[inside, 0]), depth[inside])
            zbuffer = cv2.erode(zbuffer, np.ones((3, 3), np.uint8))
            x, y = pixels[anchors, 0].clip(0, w - 1), pixels[anchors, 1].clip(0, h - 1)
            visible = (inside[anchors] & foreground[y, x]
                       & (np.abs(depth[anchors] - zbuffer[y, x]) < .03))
            scores = posterior[y, x]
            labels, confidence = scores.argmax(-1), scores.max(-1)
            visible &= (labels > 0) & (confidence >= .7)
            samples[camera] = (x, y, labels, confidence, visible, foreground.shape)
        result, accepted = {}, 0
        for camera, (x, y, labels, confidence, visible, shape) in samples.items():
            votes = np.zeros((len(anchors), 29), dtype=np.float32)
            count = np.zeros(len(anchors), dtype=int)
            for other, (_, _, ol, oc, ov, _) in samples.items():
                if other == camera:
                    continue
                ii = np.flatnonzero(ov)
                votes[ii, ol[ii]] += oc[ii]
                count += ov
            total = votes.sum(-1)
            consensus = votes.max(-1) / np.maximum(total, 1e-8)
            chosen = visible & (count >= 2) & (consensus >= .8)
            lab = np.zeros(shape, dtype=np.uint8)
            support = np.zeros(shape, dtype=np.float32)
            # Keep hints sparse: loose garments are not the fitted body surface.
            for i in np.flatnonzero(chosen):
                if consensus[i] > support[y[i], x[i]]:
                    lab[y[i], x[i]] = votes[i].argmax()
                    support[y[i], x[i]] = consensus[i]
            accepted += int(chosen.sum())
            result[camera] = (lab, support)
        return result, accepted
