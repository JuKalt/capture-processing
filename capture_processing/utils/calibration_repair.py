"""Reversible geometry repair for the CVPR recordings exported with old calibration."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import os
import uuid

OLD_TIMESTAMP = '2025-10-22T15:50:00+02:00'
REFERENCE_TIMESTAMP = '2026-07-29T17:35:40.922576+02:00'
ZOOM_CAMERAS = {'C0005', 'C0008', 'C0018', 'C0019', 'C0029', 'C0034', 'C0038'}


def affected_subject(name):
    match = re.fullmatch(r'(?:vci__)?2026_\d{2}_\d{2}_cvpr-dataset_(\d{3})', name)
    return bool(match and 36 <= int(match[1]) <= 48)


def validate_source_calibration(data):
    """Reject the confirmed obsolete export before processing or quick reuse."""
    meta = data.get('meta', {})
    if (affected_subject(meta.get('name', ''))
            and meta.get('calibration', {}).get('timestamp') == OLD_TIMESTAMP):
        raise ValueError('Known obsolete CVPR calibration for subjects 036–048. '
                         'Repair the source with python -m capture_processing.tools.repair_vci_calibration; '
                         'all seven zoomed cameras, including C0019, are affected.')


def repair_calibrations(root, reference, *, apply=False, stage_dir=None):
    """Preflight every target, then back up and atomically replace geometry only.

    Capture metadata and processing history remain unchanged. This cannot undo
    undistortion already applied to the exported images.
    """
    if apply and stage_dir is not None:
        raise ValueError("Choose apply or stage_dir, not both")
    root, reference = Path(root), Path(reference)
    donor_bytes = reference.read_bytes()
    donor = json.loads(donor_bytes)
    if donor['meta']['calibration']['timestamp'] != REFERENCE_TIMESTAMP:
        raise ValueError('Reference must use the verified July 2026 calibration.')
    donor_cams = {c['camera_id']: c for c in donor['cameras']}
    for cid in ZOOM_CAMERAS:
        if donor_cams[cid]['intrinsics']['camera_matrix'][0] < 12000:
            raise ValueError(f'Reference lacks zoom calibration for {cid}')
    paths = sorted(p for p in root.glob('*/*/calibration_dome.json')
                   if affected_subject(p.parent.name))
    if len(paths) != 13 or len({p.parent.name for p in paths}) != 13:
        raise ValueError(f'Expected exactly 13 source subjects; found {len(paths)}')
    plan = []
    for path in paths:
        original = path.read_bytes()
        data = json.loads(original)
        backup = path.with_name(path.name + '.before-2026-11-09-repair.bak')
        if data.get('calibration_repair', {}).get('reference_sha256') == hashlib.sha256(donor_bytes).hexdigest():
            if data['cameras'] != donor['cameras'] or data['meta']['calibration'] != donor['meta']['calibration']:
                raise ValueError(f'Repaired geometry has changed: {path}')
            if not backup.is_file() or hashlib.sha256(backup.read_bytes()).hexdigest() != data['calibration_repair']['original_sha256']:
                raise ValueError(f'Missing or changed repair backup: {backup}')
            plan.append((path, original, None, backup))
            continue
        if data['meta']['calibration']['timestamp'] != OLD_TIMESTAMP:
            raise ValueError(f'Unexpected calibration timestamp: {path}')
        current = {c['camera_id']: c for c in data['cameras']}
        if current.keys() != donor_cams.keys():
            raise ValueError(f'Camera IDs differ: {path}')
        for cid in current:
            if current[cid]['intrinsics']['resolution'] != donor_cams[cid]['intrinsics']['resolution']:
                raise ValueError(f'Image resolution differs: {path}: {cid}')
        if backup.exists() and backup.read_bytes() != original:
            raise ValueError(f'Backup would be overwritten: {backup}')
        updated = deepcopy(data)
        updated['cameras'] = deepcopy(donor['cameras'])
        updated['meta']['calibration'] = deepcopy(donor['meta']['calibration'])
        updated['calibration_repair'] = {
            'version': 1, 'date': '2026-11-09', 'reference': str(reference.resolve()),
            'reference_sha256': hashlib.sha256(donor_bytes).hexdigest(),
            'original_sha256': hashlib.sha256(original).hexdigest(),
            'backup': backup.name, 'affected_zoom_cameras': sorted(ZOOM_CAMERAS),
            'scope': 'All camera geometry replaced together; capture metadata, raw_calibration and processing_applied preserved as original export history.',
            'media_reprocessed': False,
            'limitation': 'Previously applied undistortion is not undone by this geometry repair.'}
        plan.append((path, original, json.dumps(updated, indent=2).encode() + b'\n', backup))
    result = []
    for path, original, replacement, backup in plan:
        if apply and replacement is not None:
            # Recheck before mutation; retain the original bytes without reserialization.
            if path.read_bytes() != original:
                raise ValueError(f'Source changed during preflight: {path}')
            source_stat = path.stat()
            def preserve_permissions(destination):
                destination.chmod(source_stat.st_mode & 0o777)
                if os.geteuid() == 0:
                    os.chown(destination, source_stat.st_uid, source_stat.st_gid)
            if not backup.exists():
                with backup.open('xb') as stream:
                    stream.write(original)
                preserve_permissions(backup)
            temporary = path.with_name(f'.{path.name}.{uuid.uuid4().hex}.repair.tmp')
            try:
                with temporary.open('xb') as stream:
                    stream.write(replacement)
                preserve_permissions(temporary)
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)
        staged = None
        if stage_dir is not None:
            staged = Path(stage_dir) / path.relative_to(root)
            staged.parent.mkdir(parents=True, exist_ok=True)
            staged.write_bytes(replacement if replacement is not None else original)
        result.append({'subject': path.parent.name, 'path': str(path),
                       'backup': str(backup), 'staged_path': str(staged) if staged else None, 'status': 'already_repaired' if replacement is None else 'repaired' if apply else 'staged' if staged else 'planned',
                       'original_sha256': hashlib.sha256(backup.read_bytes() if replacement is None else original).hexdigest()})
    return result
