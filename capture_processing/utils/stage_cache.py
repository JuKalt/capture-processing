"""Input signatures and completion checks for subject processing stages.

Signatures must be identical on every worker that sees the same shared subject
directory: they never contain checkout/model paths or checkout mtimes. Code and
model files are identified by content; fits by their array data (excluding the
embedded signatures, so re-signing a fit never invalidates its consumers);
subject media by subject-relative path, size and mtime on the shared storage.
"""
import hashlib
import json
import os
from pathlib import Path
import socket

PROJECT = Path(__file__).resolve().parents[2]
SIGNATURE_VERSION = 2
# Fit archive keys that record provenance rather than fit content.
SIGNATURE_KEYS = frozenset({'estimation_signature', 'conversion_signature'})
_DIGESTS = {}


def file_stamp(path):
    """Change detection for artifacts that never leave the shared subject directory."""
    path = Path(path)
    if not path.is_file():
        return None
    stat = path.stat()
    return [str(path.resolve()), stat.st_size, stat.st_mtime_ns]


def _digest_cache():
    root = Path(os.environ.get('XDG_CACHE_HOME') or Path.home() / '.cache')
    return root / 'vci-dataset' / 'content-digests.json'


def content_digest(path):
    """SHA-256 of a file, memoized per (path, size, mtime) across processes."""
    resolved = Path(path).resolve()
    # Hugging Face's blob store is content-addressed (sha256 for LFS files).
    if resolved.parent.name == 'blobs' and len(resolved.name) in (40, 64):
        return resolved.name
    stat = resolved.stat()
    key = f'{resolved}:{stat.st_size}:{stat.st_mtime_ns}'
    if key in _DIGESTS:
        return _DIGESTS[key]
    cache = _digest_cache()
    try:
        stored = json.loads(cache.read_text())
    except (OSError, ValueError):
        stored = {}
    if not isinstance(stored, dict):
        stored = {}
    if key not in stored:
        digest = hashlib.sha256()
        with resolved.open('rb') as stream:
            for block in iter(lambda: stream.read(1 << 24), b''):
                digest.update(block)
        stored[key] = digest.hexdigest()
        try:
            cache.parent.mkdir(parents=True, exist_ok=True)
            temporary = cache.with_name(f'{cache.name}.{socket.gethostname()}.{os.getpid()}')
            temporary.write_text(json.dumps(stored))
            temporary.replace(cache)
        except OSError:
            pass
    _DIGESTS[key] = stored[key]
    return stored[key]


def content_stamp(path):
    """Machine-independent identity of a code or model file."""
    path = Path(path)
    if not path.is_file():
        return None
    return [path.name, path.stat().st_size, content_digest(path)]


def code_stamp(relative):
    return content_stamp(PROJECT / relative)


def fit_digest(path):
    """Hash of a fit archive's arrays, excluding its embedded stage signatures."""
    import numpy as np
    digest = hashlib.sha256()
    import zipfile

    try:
        with np.load(path, allow_pickle=False) as data:
            for key in sorted(set(data.files) - SIGNATURE_KEYS):
                value = data[key]
                digest.update(f'{key}:{value.dtype.str}:{value.shape};'.encode())
                digest.update(np.ascontiguousarray(value).tobytes())
    except (OSError, ValueError, TypeError, zipfile.BadZipFile):
        # Record damaged archives as inputs so metrics can report their failure.
        return content_digest(path)
    return digest.hexdigest()


def subject_inputs(subject, *, media=('images', 'masks'), fit_files=None):
    subject = Path(subject)
    files = (sorted(subject.glob('*params.npz')) + [subject / 'smpl_params_confidence.npz', subject / 'smplx_fit_diagnostics.json', subject / 'smplx_confidence_fit_diagnostics.json']
             if fit_files is None else [subject / name for name in fit_files])
    files += [subject / 'calibration.json']
    for name in media:
        directory = subject / name
        if directory.is_dir():
            files.extend(sorted(path for path in directory.rglob('*') if path.is_file()))
    return files


def subject_signature(subject, options, *, media=('images', 'masks'), dependencies=(), fit_files=None):
    subject = Path(subject)
    inputs = []
    for path in subject_inputs(subject, media=media, fit_files=fit_files):
        name = path.relative_to(subject).as_posix()
        if not path.is_file():
            inputs.append([name, None])
        elif path.suffix == '.npz':
            inputs.append([name, fit_digest(path)])
        elif path.parent == subject:
            inputs.append([name, hashlib.sha256(path.read_bytes()).hexdigest()])
        else:
            # Media stamps avoid reading every image on a cached run.
            stat = path.stat()
            inputs.append([name, stat.st_size, stat.st_mtime_ns])
    payload = {'version': SIGNATURE_VERSION, 'inputs': inputs, 'options': options,
               'dependencies': dependencies}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def rendering_signature(subject, repo, kind, settings):
    from capture_processing.utils.silhouette_metrics import _repo_stamp, _sam_asset_stamps
    return subject_signature(subject, {'kind': kind, **settings}, media=('images',),
                             dependencies=[_repo_stamp(Path(repo)), _sam_asset_stamps() if kind == 'mhr' else {},
                                           code_stamp('capture_processing/tools/visualize_mhr.py'),
                                           code_stamp('capture_processing/tools/evaluate_silhouette_fit.py')],
                             fit_files=['mhr_params.npz' if kind == 'mhr' else 'mhr_from_smplx_params.npz'])


def rendering_complete(output, signature):
    output = Path(output)
    try:
        manifest = json.loads((output / 'manifest.json').read_text())
        if not isinstance(manifest, dict):
            return False
        artifacts = manifest.get('artifacts')
        if not isinstance(artifacts, dict) or not artifacts:
            return False
        expected = {'scene.rrd'}
        for camera in manifest.get('cameras', []):
            expected.update(camera + suffix for suffix in ('.mp4', '_preview.png'))
        return (manifest.get('complete') is True and manifest.get('signature') == signature
                and expected <= artifacts.keys()
                and all(stamp is not None and stamp[1] > 0
                        and file_stamp(output / name) == stamp for name, stamp in artifacts.items()))
    except (OSError, ValueError, TypeError, KeyError, IndexError):
        return False


def metrics_signature(subject, options):
    from capture_processing.utils.silhouette_metrics import resolve_paths, _repo_stamp, _sam_asset_stamps, VERSION
    official, sam3d, model = resolve_paths(options)
    # Interpreter and checkout paths differ per worker; the repositories and
    # model they select are captured by content below.
    settings = {'cameras': options.cameras, 'max_frames': options.max_frames}
    return subject_signature(subject, settings, dependencies=[VERSION, content_stamp(model),
                             _repo_stamp(official), _repo_stamp(sam3d), _sam_asset_stamps(),
                             code_stamp('capture_processing/utils/fit_metrics.py'),
                             code_stamp('capture_processing/utils/silhouette_metrics.py')])


def metric_artifacts(subject):
    return {path.name: file_stamp(path) for path in sorted(Path(subject).glob('silhouette_metrics.*.json'))}


def fit_artifacts(subject):
    """Lightweight completion evidence for canonical fits and their diagnostics."""
    from capture_processing.utils.fit_metrics import FIT_FILES
    names = [*FIT_FILES.values(), 'smplx_fit_diagnostics.json', 'smplx_confidence_fit_diagnostics.json']
    return {name: file_stamp(Path(subject) / name) for name in names
            if (Path(subject) / name).is_file()}
