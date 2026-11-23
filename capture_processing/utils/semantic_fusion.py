"""Mask-constrained semantic fusion with occlusion-aware, bidirectional motion checks."""
from __future__ import annotations
import numpy as np
from capture_processing.utils.semantic_schema import FINE, FINE_TO_GENERAL, UNKNOWN


def compact(probabilities):
    p = np.asarray(probabilities, dtype=np.float32)
    if (p.ndim != 3 or p.shape[-1] != len(FINE) or not np.isfinite(p).all()
            or (p < 0).any() or not np.allclose(p.sum(-1), 1, atol=.01)):
        raise ValueError('Parser must return finite, normalized dome29 probabilities')
    ids = np.argsort(p, axis=-1)[..., -3:][..., ::-1].astype(np.uint8)
    return ids, np.take_along_axis(p, ids, axis=-1).astype(np.float16)


def expand(ids, scores):
    if (ids.ndim != 3 or ids.shape[-1] != 3 or ids.shape != scores.shape
            or ids.dtype != np.uint8 or (ids >= len(FINE)).any()
            or not np.isfinite(scores).all() or (scores < 0).any() or (scores > 1).any()
            or (scores.sum(-1) > 1.002).any()
            or (np.sort(ids, axis=-1)[..., 1:] == np.sort(ids, axis=-1)[..., :-1]).any()):
        raise ValueError('Invalid cached semantic posterior')
    remainder = np.maximum(0, 1 - scores.astype(np.float32).sum(-1)) / (len(FINE) - 3)
    p = np.repeat(remainder[..., None], len(FINE), axis=-1)
    np.put_along_axis(p, ids, scores.astype(np.float32), axis=-1)
    return p / p.sum(-1, keepdims=True)


def propagate(target_gray, source_gray, source_prob, target_mask, source_mask,
              max_roundtrip=1.5, max_photometric=30.):
    """Warp source -> target using target->source sampling and cycle validation."""
    import cv2
    if target_gray.shape != source_gray.shape:
        raise ValueError('Temporal propagation requires matching image dimensions')
    # DIS is used only for label propagation, independently of the frozen parser.
    tracker = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    backward = tracker.calc(target_gray, source_gray, None)
    forward = tracker.calc(source_gray, target_gray, None)
    return warp_checked(target_gray, source_gray, source_prob, target_mask, source_mask,
                        backward, forward, max_roundtrip, max_photometric)


def warp_checked(target_gray, source_gray, source_prob, target_mask, source_mask,
                 backward, forward, max_roundtrip=1.5, max_photometric=30.):
    import cv2
    h, w = target_gray.shape
    y, x = np.mgrid[:h, :w].astype(np.float32)
    mx, my = x + backward[..., 0], y + backward[..., 1]
    def sample(a, interpolation=cv2.INTER_LINEAR):
        return cv2.remap(a.astype(np.float32), mx, my, interpolation,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    cycle = backward + sample(forward)
    valid = ((mx >= 0) & (mx <= w - 1) & (my >= 0) & (my <= h - 1)
             & np.isfinite(cycle).all(-1) & (np.linalg.norm(cycle, axis=-1) <= max_roundtrip)
             & (np.abs(sample(source_gray) - target_gray) <= max_photometric)
             & target_mask & (sample(source_mask, cv2.INTER_NEAREST) > .5))
    warped = sample(source_prob)
    valid &= warped[..., 1:].max(-1) >= .5
    return warped, valid


def fuse(current, foreground, neighbours=(), hints=None, threshold=.5, margin=.1):
    total = current.copy()
    denominator = np.ones(foreground.shape, dtype=np.float32)
    diagnostics = {'flow_pixels': 0, 'raw_temporal_disagreements': 0, 'geometry_pixels': 0,
                   'geometry_checked_pixels': 0, 'geometry_disagreements': 0}
    for warped, valid in neighbours:
        weight = .75 * valid * warped.max(-1)
        total += weight[..., None] * warped
        denominator += weight
        diagnostics['flow_pixels'] += int(valid.sum())
        diagnostics['raw_temporal_disagreements'] += int(((current.argmax(-1) != warped.argmax(-1)) & valid).sum())
    if hints is not None:
        # Sparse hints only refine uncertain predictions; never force the mesh silhouette.
        label, support = hints
        checked = (support >= .8) & (label > 0) & (label < len(FINE)) & foreground
        diagnostics['geometry_checked_pixels'] = int(checked.sum())
        diagnostics['geometry_disagreements'] = int(((current.argmax(-1) != label) & checked).sum())
        valid = checked & (current.max(-1) < .8)
        yy, xx = np.nonzero(valid)
        total[yy, xx, label[valid]] += .35 * support[valid]
        denominator[valid] += .35 * support[valid]
        diagnostics['geometry_pixels'] = int(valid.sum())
    posterior = total / denominator[..., None]
    labels = posterior.argmax(-1).astype(np.uint8)
    confidence = posterior.max(-1)
    ordered = np.partition(posterior, -2, axis=-1)
    unknown = foreground & ((labels == 0) | (confidence < threshold)
                            | ((ordered[..., -1] - ordered[..., -2]) < margin))
    labels[unknown] = UNKNOWN
    labels[~foreground] = 0
    confidence[unknown] = 0
    confidence[~foreground] = 1
    lut = np.full(256, UNKNOWN, dtype=np.uint8)
    lut[:len(FINE)] = FINE_TO_GENERAL
    return labels, lut[labels], confidence, diagnostics
