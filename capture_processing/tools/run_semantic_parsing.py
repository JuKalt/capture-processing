"""Run frozen Sapiens2 inference and independently resumable semantic fusion."""
from __future__ import annotations
import argparse
import fcntl
import json
import zipfile
from pathlib import Path
import numpy as np
from PIL import Image
from capture_processing.utils.semantic_schema import FINE, FINE_TO_GENERAL, UNKNOWN, schema
from capture_processing.utils.semantic_fusion import compact, expand, fuse, propagate
from capture_processing.utils.semantic_geometry import GeometryHints
from capture_processing.utils.semantic_parsing import (ROOT, SemanticOptions, atomic_json, read_json, inventory,
    signatures, input_signature, raw_path, final_paths, final_signature, frame_groups, source_hashes, file_digest)
from capture_processing.utils.stage_cache import file_stamp


def load_view(view, edge):
    import cv2
    with Image.open(view['image']) as im:
        image = np.asarray(im.convert('RGB'))
    with Image.open(view['mask']) as im:
        mask = np.asarray(im.convert('L'))
    if mask.shape != image.shape[:2]:
        raise ValueError(f'Capture image/mask dimension mismatch: {view["image"]}')
    mask = mask > (0 if mask.max(initial=0) <= 1 else 127)
    h, w = mask.shape
    scale = min(1, edge / max(h, w))
    shape = (max(1, round(h * scale)), max(1, round(w * scale)))
    small = cv2.resize(image, (shape[1], shape[0]), interpolation=cv2.INTER_AREA)
    fg = cv2.resize(mask.astype(np.uint8), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
    return image, mask, gray, fg


def save_raw(path, p, gray, foreground, native_shape):
    path.parent.mkdir(parents=True, exist_ok=True)
    ids, scores = compact(p)
    tmp = path.with_suffix('.tmp')
    with tmp.open('wb') as stream:
        np.savez_compressed(stream, ids=ids, scores=scores, gray=gray,
                            foreground=foreground, native_shape=native_shape)
    tmp.replace(path)


def load_raw(path):
    with np.load(path, allow_pickle=False) as d:
        ids, scores, gray, fg, native = (d[name] for name in ('ids', 'scores', 'gray', 'foreground', 'native_shape'))
    p = expand(ids, scores)
    if (gray.dtype != np.uint8 or fg.dtype != bool or gray.shape != p.shape[:2]
            or fg.shape != gray.shape or native.shape != (2,) or (native <= 0).any()):
        raise ValueError(f'Invalid cached image grid: {path}')
    return p, gray, fg, tuple(int(v) for v in native)


def save_png(path, array):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    Image.fromarray(array).save(tmp, format='PNG')
    tmp.replace(path)


def validate_labels(fine, general, confidence, foreground):
    if not (fine.shape == general.shape == confidence.shape == foreground.shape):
        raise ValueError('Semantic outputs must match capture dimensions')
    lut = np.full(256, UNKNOWN, dtype=np.uint8)
    lut[:len(FINE)] = FINE_TO_GENERAL
    if (fine.dtype != np.uint8 or general.dtype != np.uint8 or confidence.dtype != np.uint16
            or not np.all((fine < len(FINE)) | (fine == UNKNOWN))
            or not np.array_equal(general, lut[fine])
            or (fine[~foreground] != 0).any() or (fine[foreground] == 0).any()
            or (confidence[fine == UNKNOWN] != 0).any()):
        raise ValueError('Invalid labels, coarse mapping, confidence, or capture-mask constraint')


def valid_export(paths, view):
    try:
        arrays = {}
        for name, path in paths.items():
            with Image.open(path) as image:
                if image.format != 'PNG':
                    return False
                image.load()
                if name != 'preview':
                    arrays[name] = np.array(image)
        with Image.open(view['mask']) as image:
            pixels = np.asarray(image.convert('L'))
        mask = pixels > (0 if pixels.max(initial=0) <= 1 else 127)
        with Image.open(view['image']) as image:
            if image.size[::-1] != mask.shape:
                return False
        score = arrays['confidence']
        if not np.issubdtype(score.dtype, np.integer) or (score < 0).any() or (score > 65535).any():
            return False
        validate_labels(arrays['detailed'], arrays['general'], score.astype(np.uint16), mask)
        return True
    except (OSError, ValueError, KeyError):
        return False


def export(paths, image, mask, labels, general, confidence):
    import cv2
    h, w = mask.shape
    fine = cv2.resize(labels, (w, h), interpolation=cv2.INTER_NEAREST)
    coarse = cv2.resize(general, (w, h), interpolation=cv2.INTER_NEAREST)
    score = cv2.resize(confidence, (w, h), interpolation=cv2.INTER_LINEAR)
    # Restore exact capture boundaries after working-grid propagation/upsampling.
    fine[~mask] = coarse[~mask] = 0
    fine[mask & (fine == 0)] = UNKNOWN
    lut = np.full(256, UNKNOWN, dtype=np.uint8)
    lut[:len(FINE)] = FINE_TO_GENERAL
    coarse = lut[fine]
    score[fine == UNKNOWN] = 0
    score[~mask] = 1
    score = np.rint(np.clip(score, 0, 1) * 65535).astype(np.uint16)
    validate_labels(fine, coarse, score, mask)
    save_png(paths['detailed'], fine)
    save_png(paths['general'], coarse)
    save_png(paths['confidence'], score)
    # Deterministic colors shared across all frames, cameras and subjects.
    palette = np.array([[(i * 67) % 256, (i * 131) % 256, (i * 197) % 256] for i in range(256)], dtype=np.uint8)
    palette[UNKNOWN] = 255
    scale = min(1, 512 / max(h, w))
    size = (max(1, round(w * scale)), max(1, round(h * scale)))
    rgb = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    colored = palette[cv2.resize(fine, size, interpolation=cv2.INTER_NEAREST)]
    overlay = np.rint(.55 * rgb + .45 * colored).astype(np.uint8)
    save_png(paths['preview'], overlay)
    return {'foreground_pixels': int(mask.sum()), 'unknown_pixels': int((fine == UNKNOWN).sum()),
            'class_pixels': np.bincount(fine[fine != UNKNOWN], minlength=len(FINE)).tolist()}


def run(subject, options, predictor_factory=None):
    import cv2
    cv2.setNumThreads(2)
    from capture_processing.utils.mamma_additional_fit import _avoid_hugepage_stalls
    _avoid_hugepage_stalls()
    options.validate()
    subject = Path(subject).resolve()
    root = subject / ROOT
    root.mkdir(parents=True, exist_ok=True)
    with (root / '.lock').open('w') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f'Another semantic worker is already processing {subject}') from exc
        return _run_locked(subject, options, predictor_factory)


def _run_locked(subject, options, predictor_factory):
    root = subject / ROOT
    raw_sig, fusion_sig, mesh = signatures(subject, options)
    views = inventory(subject)
    manifest = read_json(root / 'manifest.json')
    if options.force:
        import shutil
        shutil.rmtree(root / 'state', ignore_errors=True)
    elif not manifest.get('complete'):
        # Each atomic receipt checkpoints one completed unit, even if interrupted
        # before the final aggregate manifest is written. Avoid quadratic writes.
        for kind in ('raw', 'final'):
            records = manifest.get(kind, {})
            records = records if isinstance(records, dict) else {}
            for path in (root / 'state' / kind).glob('*/*.json'):
                key = str(path.relative_to(root / 'state' / kind).with_suffix(''))
                record = read_json(path)
                if record:
                    records[key] = record
            manifest[kind] = records
    if options.force or manifest.get('version') not in (None, 1):
        manifest = {}
    for kind in ('raw', 'final'):
        value = manifest.get(kind, {})
        manifest[kind] = {k: v for k, v in value.items() if isinstance(v, dict)} if isinstance(value, dict) else {}
    stale = (set(manifest['raw']) | set(manifest['final'])) - set(views)
    for key in stale:
        parts = Path(key).parts
        if len(parts) != 2 or '..' in parts or Path(key).is_absolute():
            continue
        for path in [raw_path(root, key), *final_paths(root, key).values(),
                     root / 'state/raw' / (key + '.json'), root / 'state/final' / (key + '.json')]:
            path.unlink(missing_ok=True)
    manifest.update(complete=False, version=1, subject=subject.name,
                    raw={k: v for k, v in manifest.get('raw', {}).items() if k in views},
                    final={k: v for k, v in manifest.get('final', {}).items() if k in views})
    def persist():
        atomic_json(root / 'manifest.json', manifest)
    def save_record(kind, key):
        path = root / 'state' / kind / (key + '.json')
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(path, manifest[kind][key])
    persist()
    predictor = None
    inferred = 0
    for index, (key, view) in enumerate(views.items()):
        path = raw_path(root, key)
        signature = input_signature(view, raw_sig)
        previous = manifest['raw'].get(key, {})
        reusable = previous.get('signature') == signature and previous.get('stamp') == file_stamp(path)
        hashes = source_hashes(view) if options.verify or not previous.get('source_sha256') else previous['source_sha256']
        if options.verify and previous.get('source_sha256') and previous['source_sha256'] != hashes:
            reusable = False
        if options.verify and reusable and previous.get('sha256') and previous['sha256'] != file_digest(path):
            reusable = False
        if reusable and options.verify:
            try:
                load_raw(path)
            except (OSError, ValueError, KeyError, zipfile.BadZipFile):
                reusable = False
        if not reusable:
            image, mask, gray, fg = load_view(view, options.working_long_edge)
            p = np.zeros((*fg.shape, len(FINE)), dtype=np.float32)
            p[..., 0] = 1
            if mask.any():
                if predictor is None:
                    if predictor_factory is None:
                        from capture_processing.tools.sapiens2_predictor import Sapiens2Predictor
                        device = 'cpu' if options.device == 'cpu' else None
                        predictor = Sapiens2Predictor(options.model, device)
                    else:
                        predictor = predictor_factory()
                p = predictor(image, mask, fg.shape)
                if p.shape != (*fg.shape, len(FINE)):
                    raise ValueError('Parser posterior dimensions do not match the requested working grid')
                p[~fg] = 0
                p[~fg, 0] = 1
                inferred += 1
            save_raw(path, p, gray, fg, mask.shape)
            manifest['raw'][key] = {'signature': signature, 'stamp': file_stamp(path),
                                    'source_sha256': source_hashes(view), 'sha256': file_digest(path)}
            save_record('raw', key)
        elif not previous.get('source_sha256') or not previous.get('sha256'):
            manifest['raw'][key]['source_sha256'] = hashes
            manifest['raw'][key]['sha256'] = file_digest(path)
            save_record('raw', key)
        if index == 0 or (index + 1) % 50 == 0:
            print(f'Semantic inference {index + 1}/{len(views)}; recomputed {inferred}', flush=True)
    # No GPU model needs to be retained during CPU fusion/export.
    del predictor
    geometry = GeometryHints(subject, mesh)
    repaired = 0
    groups = frame_groups(views)
    multiview = mesh is not None and (subject / 'calibration.json').is_file()
    frame_ids = sorted(groups)
    for index, frame in enumerate(frame_ids):
        keys = groups[frame]
        dirty = []
        for key in keys:
            previous = manifest['final'].get(key, {})
            signature = final_signature(key, views, manifest['raw'], fusion_sig, groups, multiview)
            if (previous.get('signature') != signature or any(
                    file_stamp(path) is None or previous.get('artifacts', {}).get(name) != file_stamp(path)
                    for name, path in final_paths(root, key).items())
                    or (options.verify and (not valid_export(final_paths(root, key), views[key])
                        or any(previous.get('sha256', {}).get(name) != file_digest(path)
                               for name, path in final_paths(root, key).items())))):
                dirty.append((key, signature))
        if not dirty:
            continue
        needed = keys if geometry.vertices is not None else [key for key, signature in dirty]
        cache = {views[k]['camera']: load_raw(raw_path(root, k)) for k in needed}
        hints, _ = geometry.frame(frame, {c: (p, fg, native) for c, (p, gray, fg, native) in cache.items()})
        for key, signature in dirty:
            view = views[key]
            current, gray, foreground, native = cache[view['camera']]
            neighbours = []
            for delta in (-1, 1):
                neighbour = f'{view["camera"]}/{frame + delta:08d}'
                if neighbour not in views:
                    continue
                source, source_gray, source_mask, _ = load_raw(raw_path(root, neighbour))
                if source_gray.shape != gray.shape or min(gray.shape) < 16:
                    continue
                neighbours.append(propagate(gray, source_gray, source, foreground, source_mask))
            labels, general, confidence, diagnostics = fuse(current, foreground, neighbours,
                hints.get(view['camera']), options.confidence, options.margin)
            diagnostics['raw_temporal_disagreements_all'] = diagnostics['raw_temporal_disagreements']
            diagnostics['evaluated_temporal_pixels'] = sum(int((valid & (labels != UNKNOWN)).sum()) for p, valid in neighbours)
            diagnostics['raw_temporal_disagreements'] = sum(int(((current.argmax(-1) != p.argmax(-1)) & valid
                & (labels != UNKNOWN)).sum()) for p, valid in neighbours)
            diagnostics['fused_temporal_disagreements'] = sum(int(((labels != p.argmax(-1)) & valid
                & (labels != UNKNOWN)).sum()) for p, valid in neighbours)
            image, mask, _, _ = load_view(view, options.working_long_edge)
            paths = final_paths(root, key)
            diagnostics.update(export(paths, image, mask, labels, general, confidence))
            manifest['final'][key] = {'signature': signature, 'artifacts': {
                name: file_stamp(path) for name, path in paths.items()},
                'sha256': {name: file_digest(path) for name, path in paths.items()}, 'diagnostics': diagnostics}
            repaired += 1
            save_record('final', key)
        if index == 0 or (index + 1) % 25 == 0:
            print(f'Semantic fusion {index + 1}/{len(frame_ids)} frames; repaired {repaired} views', flush=True)
    atomic_json(root / 'schema.json', schema())
    rows = {k: v['diagnostics'] for k, v in manifest['final'].items()}
    foreground = sum(r['foreground_pixels'] for r in rows.values())
    unknown = sum(r['unknown_pixels'] for r in rows.values())
    if foreground > 0 and unknown == foreground:
        raise ValueError('Parser assigned no confident labels anywhere in this subject; inspect the checkpoint/thresholds')
    flow = sum(r.get('evaluated_temporal_pixels', r['flow_pixels']) for r in rows.values())
    checked = sum(r.get('geometry_checked_pixels', 0) for r in rows.values())
    conflicts = sum(r.get('geometry_disagreements', 0) for r in rows.values())
    report = {'complete': len(rows) == len(views), 'subject': subject.name, 'view_count': len(rows),
        'frame_count': len(frame_ids), 'camera_count': len({v['camera'] for v in views.values()}),
        'evaluated_temporal_pixels': flow,
        'foreground_pixels': foreground, 'unknown_foreground_fraction': unknown / max(1, foreground),
        'raw_temporal_disagreement': sum(r['raw_temporal_disagreements'] for r in rows.values()) / flow if flow else None,
        'fused_temporal_disagreement': sum(r['fused_temporal_disagreements'] for r in rows.values()) / flow if flow else None,
        'temporal_unavailable': 'No jointly known, flow-valid adjacent pixels' if not flow else None,
        'cross_view_checked_pixels': checked, 'cross_view_disagreement': conflicts / checked if checked else None,
        'geometry_hint_pixels': sum(r['geometry_pixels'] for r in rows.values()),
        'geometry_unavailable': geometry.reason if options.multiview else 'Disabled by option',
        'flagged_views': [k for k, r in rows.items() if r['unknown_pixels'] / max(1, r['foreground_pixels']) > .25],
        'per_view': rows, 'ground_truth_accuracy': 'Unavailable: automatic pseudo-labels; no manual ground truth supplied',
        'notes': ['Top-3 posteriors cached; remaining probability mass distributed uniformly.',
                  'DIS optical flow with forward/backward cycle and photometric checks; no propagation across frame-ID gaps.',
                  'Raw/fused temporal disagreement uses the same known, flow-valid target pixels; it measures consistency, not accuracy.',
                  'Cross-view hints use approximate vertex depth visibility and two other agreeing cameras.']}
    atomic_json(root / 'metrics.json', report)
    manifest.update(complete=report['complete'], schema=file_stamp(root / 'schema.json'),
                    metrics=file_stamp(root / 'metrics.json'), inference_signature=raw_sig, fusion_signature=fusion_sig)
    persist()
    print(f'Semantic parsing ready: {root}; inferred {inferred}, repaired {repaired}', flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--subject-root', type=Path, required=True)
    parser.add_argument('--request', type=Path, required=True)
    args = parser.parse_args()
    run(args.subject_root, SemanticOptions(**json.loads(args.request.read_text())))


if __name__ == '__main__':
    main()
