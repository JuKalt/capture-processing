"""Independent semantic preprocessing lifecycle, compatible with old processed subjects."""
from __future__ import annotations

from capture_processing.utils.subject_lock import subject_locked
from dataclasses import asdict, dataclass, field
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import zipfile
from capture_processing.utils.stage_cache import file_stamp

PROJECT = Path(__file__).resolve().parents[2]
MODEL_ID = 'facebook/sapiens2-seg-1b'
MODEL_REVISION = '60cf0b3584d68dccca9adb3b59677b72a5611378'
MODEL_FILES = ('config.json', 'preprocessor_config.json', 'model.safetensors')
MODEL_HASHES = {
    'config.json': '1a951172d547eb508eec825e4258a24b8b1170df3e9bd9084b1b2a792fb82249',
    'preprocessor_config.json': 'c63123e8d940368bb27f2011cfb45e23f5add5d0a73c10b717f114b2ec1dc858',
    'model.safetensors': '4b73c44963b377e93fcb4c4053f72a189836a22d05e12c30383046b9cd3c5bd4',
}
ROOT = 'semantic_parsing'


def default_semantic_python():
    if os.environ.get('VCI_SEMANTIC_PYTHON'):
        return os.environ['VCI_SEMANTIC_PYTHON']
    prefix = Path(sys.prefix)
    environments = prefix.parent if prefix.parent.name == 'envs' else prefix / 'envs'
    interpreter = environments / os.environ.get('VCI_SEMANTIC_ENV', 'sapiens2') / 'bin/python'
    if interpreter.is_file():
        return shlex.quote(str(interpreter))
    return 'conda run -n sapiens2 python'


@dataclass(frozen=True)
class SemanticOptions:
    enabled: bool = True
    python: str = field(default_factory=default_semantic_python)
    model: str = str(PROJECT / 'models/sapiens2-seg-1b')
    device: str | None = None
    working_long_edge: int = 1024
    confidence: float = .5
    margin: float = .1
    multiview: bool = True
    force: bool = False
    verify: bool = False

    def validate(self):
        if self.working_long_edge < 32 or not 0 < self.confidence <= 1 or not 0 <= self.margin <= 1:
            raise ValueError('Semantic working edge must be >=32; confidence in (0,1], margin in [0,1]')


def options_from_args(args):
    from dataclasses import replace
    existing = getattr(args, 'semantic_options', None)
    if existing is not None:
        return replace(existing, device=getattr(args, 'device', existing.device))
    return SemanticOptions(enabled=getattr(args, 'semantic_parsing', True),
        python=getattr(args, 'semantic_python', default_semantic_python()),
        model=str(getattr(args, 'semantic_model', None) or os.environ.get('VCI_SAPIENS2_MODEL', SemanticOptions.model)),
        device=getattr(args, 'device', None),
        working_long_edge=getattr(args, 'semantic_working_long_edge', 1024),
        confidence=getattr(args, 'semantic_confidence', .5),
        margin=getattr(args, 'semantic_margin', .1),
        multiview=getattr(args, 'semantic_multiview', True),
        force=getattr(args, 'semantic_force', False),
        verify=getattr(args, 'semantic_verify', False))


def add_arguments(parser):
    import argparse
    parser.add_argument('--semantic-parsing', action=argparse.BooleanOptionalAction, default=True,
                        help='Add general/detailed semantic labels using existing capture masks; enabled by default.')
    parser.add_argument('--semantic-python', default=default_semantic_python())
    parser.add_argument('--semantic-model', type=Path, default=None,
                        help='Local Sapiens2 HF snapshot; defaults to VCI_SAPIENS2_MODEL or models/sapiens2-seg-1b.')
    parser.add_argument('--semantic-working-long-edge', type=int, default=1024,
                        help='Temporal fusion grid; exported labels retain the original image resolution.')
    parser.add_argument('--semantic-confidence', type=float, default=.5)
    parser.add_argument('--semantic-margin', type=float, default=.1)
    parser.add_argument('--semantic-multiview', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--semantic-verify', action='store_true',
                        help='Deep-check cached probability arrays and label PNGs against capture masks; repair only failures.')
    parser.add_argument('--semantic-force', action='store_true', help='Refresh semantic outputs only.')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    import uuid
    tmp = path.with_name(f'.{path.name}.{uuid.uuid4().hex}.tmp')
    try:
        tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def read_json(path):
    try:
        value = json.loads(Path(path).read_text())
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def inventory(subject):
    """Pair legacy numeric and prefixed VCI names by camera and numeric frame ID."""
    subject = Path(subject)
    def files(folder):
        result = {}
        for path in sorted(folder.iterdir()) if folder.is_dir() else []:
            if path.suffix.lower() not in ('.jpg', '.jpeg', '.png', '.webp'):
                continue
            match = re.search(r'(\d+)$', path.stem)
            if match is None:
                raise ValueError(f'No numeric frame ID: {path}')
            frame = int(match.group(1))
            if frame in result:
                raise ValueError(f'Duplicate camera/frame ID: {path}')
            result[frame] = path
        return result
    image_root, mask_root = subject / 'images', subject / 'masks'
    image_cameras = {path.name for path in image_root.iterdir() if path.is_dir()}
    mask_cameras = {path.name for path in mask_root.iterdir() if path.is_dir()} if mask_root.is_dir() else set()
    if image_cameras != mask_cameras:
        raise ValueError('Image/mask camera directories differ: '
                         f'images without masks={sorted(image_cameras - mask_cameras)}, '
                         f'masks without images={sorted(mask_cameras - image_cameras)}')
    views = {}
    for directory in sorted(image_root.iterdir()):
        if not directory.is_dir():
            continue
        images, masks = files(directory), files(subject / 'masks' / directory.name)
        if set(images) != set(masks):
            raise ValueError(f'Missing image/mask pairs in {directory.name}: '
                             f'images without masks={sorted(images.keys()-masks.keys())[:10]}, '
                             f'masks without images={sorted(masks.keys()-images.keys())[:10]}')
        for frame, image in images.items():
            views[f'{directory.name}/{frame:08d}'] = {
                'camera': directory.name, 'frame': frame, 'image': str(image), 'mask': str(masks[frame])}
    if not views:
        raise ValueError(f'No existing image/mask pairs under {subject}')
    return views


def verify_model_assets(model_path):
    """Only inference needs model files; cache checks/repairs remain portable across workers."""
    model = Path(model_path).expanduser().resolve()
    record = read_json(model / 'vci_model.json')
    if (record.get('model_id') != MODEL_ID or record.get('revision') != MODEL_REVISION
            or record.get('label_schema') != 'sapiens2_dome29' or record.get('sha256') != MODEL_HASHES):
        raise ValueError(f'Unverified/missing Sapiens2 assets at {model}. '
                         'Run bash scripts/setup/setup_semantic_parsing_env.sh; upstream stages are untouched.')
    for name in MODEL_FILES:
        path = model / name
        stamp = file_stamp(path)
        if stamp is None or stamp[1] <= 0:
            raise ValueError(f'Missing Sapiens2 asset: {path}')
        if record.get('verified_files', {}).get(name) != stamp[1:] and file_digest(path) != MODEL_HASHES[name]:
            raise ValueError(f'Corrupted Sapiens2 asset: {path}; rerun model setup')


def signatures(subject, options):
    from capture_processing.utils.semantic_geometry import geometry_path
    options.validate()
    def code(name):
        return hashlib.sha256((PROJECT / name).read_bytes()).hexdigest()
    raw = digest({'version': 1, 'model': {'id': MODEL_ID, 'revision': MODEL_REVISION, 'sha256': MODEL_HASHES}, 'edge': options.working_long_edge,
        'adapter': code('capture_processing/tools/sapiens2_predictor.py'), 'schema': code('capture_processing/utils/semantic_schema.py')})
    geometry = geometry_path(subject) if options.multiview else None
    fusion = digest({'version': 1, 'confidence': options.confidence, 'margin': options.margin,
        'multiview': options.multiview, 'geometry': file_stamp(geometry) if geometry else None,
        'geometry_frames': file_stamp(geometry.parent / 'run_args.json') if geometry else None,
        'calibration': file_stamp(Path(subject) / 'calibration.json') if geometry else None,
        'fusion': code('capture_processing/utils/semantic_fusion.py'), 'geometry_code': code('capture_processing/utils/semantic_geometry.py'),
        'worker': code('capture_processing/tools/run_semantic_parsing.py'), 'schema': code('capture_processing/utils/semantic_schema.py')})
    return raw, fusion, geometry


def file_digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def source_hashes(view):
    return [file_digest(view[name]) for name in ('image', 'mask')]


def input_signature(view, raw_signature):
    return digest([raw_signature, file_stamp(view['image']), file_stamp(view['mask'])])


def raw_path(root, key):
    return root / 'raw' / (key + '.npz')


def final_paths(root, key):
    return {name: root / name / (key + '.png') for name in ('detailed', 'general', 'confidence', 'preview')}


def frame_groups(views):
    groups = {}
    for key, view in views.items():
        groups.setdefault(view['frame'], []).append(key)
    return groups


def final_signature(key, views, raw_records, fusion_signature, groups, multiview):
    camera, frame = views[key]['camera'], views[key]['frame']
    dependencies = set(groups[frame] if multiview else [key])
    dependencies.update(f'{camera}/{frame + delta:08d}' for delta in (-1, 1)
                        if f'{camera}/{frame + delta:08d}' in views)
    return digest([fusion_signature, [(k, raw_records.get(k)) for k in sorted(dependencies)]])


def semantic_complete(subject, options=None):
    options = options or SemanticOptions()
    if not options.enabled:
        return True
    if options.force:
        return False
    if options.verify:
        from capture_processing.utils.mamma_additional_fit import _avoid_hugepage_stalls
        _avoid_hugepage_stalls()
    try:
        subject = Path(subject).resolve()
        root = subject / ROOT
        manifest = read_json(root / 'manifest.json')
        if manifest.get('complete') is not True:
            return False
        raw_sig, fusion_sig, geometry = signatures(subject, options)
        views = inventory(subject)
        raw, final = manifest['raw'], manifest['final']
        groups = frame_groups(views)
        multiview = geometry is not None and (subject / 'calibration.json').is_file()
        if set(views) != set(raw) or set(views) != set(final):
            return False
        for key, view in views.items():
            if (raw[key]['signature'] != input_signature(view, raw_sig)
                    or raw[key]['stamp'] != file_stamp(raw_path(root, key))
                    or final[key]['signature'] != final_signature(key, views, raw, fusion_sig, groups, multiview)):
                return False
            if any(file_stamp(path) is None or file_stamp(path) != final[key]['artifacts'].get(name)
                   for name, path in final_paths(root, key).items()):
                return False
            if options.verify:
                from capture_processing.tools.run_semantic_parsing import load_raw, valid_export
                load_raw(raw_path(root, key))
                if raw[key].get('sha256') != file_digest(raw_path(root, key)):
                    return False
                if any(final[key].get('sha256', {}).get(name) != file_digest(path)
                       for name, path in final_paths(root, key).items()):
                    return False
                if raw[key].get('source_sha256') != source_hashes(view):
                    return False
                if not valid_export(final_paths(root, key), view):
                    return False
        return (manifest['schema'] == file_stamp(root / 'schema.json')
                and manifest['metrics'] == file_stamp(root / 'metrics.json')
                and read_json(root / 'metrics.json').get('complete') is True
                and read_json(root / 'metrics.json').get('view_count') == len(views))
    except (OSError, ValueError, KeyError, TypeError, IndexError, zipfile.BadZipFile):
        return False


def ensure_semantic_parsing(subject, args):
    options = args if isinstance(args, SemanticOptions) else options_from_args(args)
    if not options.enabled:
        return
    return _ensure_semantic_parsing(subject, args)


@subject_locked
def _ensure_semantic_parsing(subject, args):
    options = args if isinstance(args, SemanticOptions) else options_from_args(args)
    if not options.enabled:
        return
    from capture_processing.utils.mamma_additional_fit import _avoid_hugepage_stalls
    _avoid_hugepage_stalls()
    subject = Path(subject).resolve()
    from capture_processing.core.processing_log import set_semantic_complete
    if semantic_complete(subject, options):
        set_semantic_complete(subject, True)
        print(f'Using complete semantic parsing: {subject / ROOT}', flush=True)
        return
    set_semantic_complete(subject, False)
    # Validate prerequisites before launching a GPU process. No upstream stage is called.
    signatures(subject, options)
    inventory(subject)
    root = subject / ROOT
    root.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(PYTHONPATH=str(PROJECT) + os.pathsep + env.get('PYTHONPATH', ''),
               PYTHONUNBUFFERED='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
    if options.device and options.device not in ('cpu', 'cuda'):
        from capture_processing.utils.mamma import _apply_mamma_device_env
        _apply_mamma_device_env(env, options.device)
    import uuid
    request = root / f'request.{uuid.uuid4().hex}.json'
    # Concurrent launches get distinct requests; the worker locks stage outputs.
    atomic_json(request, asdict(options))
    print(f'Semantic parsing: {root / "run.log"}', flush=True)
    try:
        with (root / 'run.log').open('a') as output:
            subprocess.run([*shlex.split(options.python), '-m', 'capture_processing.tools.run_semantic_parsing',
                            '--subject-root', str(subject), '--request', str(request)],
                           cwd=PROJECT, env=env, stdout=output, stderr=subprocess.STDOUT, check=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f'Semantic parsing failed; completed upstream stages remain reusable. '
                           f'See {root / "run.log"}; setup: bash scripts/setup/setup_semantic_parsing_env.sh') from exc
    finally:
        request.unlink(missing_ok=True)
    from dataclasses import replace
    if not semantic_complete(subject, replace(options, force=False)):
        raise RuntimeError(f'Semantic outputs failed completion validation: {root}')
    set_semantic_complete(subject, True)
