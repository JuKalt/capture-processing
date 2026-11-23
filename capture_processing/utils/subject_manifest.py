"""Versioned, portable provenance for saved VCI preprocessing products."""
from dataclasses import asdict, is_dataclass
from importlib.metadata import PackageNotFoundError, version
import json
import math
from pathlib import Path
import subprocess
import zipfile

from capture_processing.core.processing_log import CONVERSION_STEPS, load_processing_log
from multiview_data.processed import ANNOTATION_PATHS
from capture_processing.utils.preprocessing_provenance import content_identity
from capture_processing.core.processing_log import mark_step_complete, save_processing_log
from capture_processing.utils.subject_lock import subject_locked

SCHEMA_VERSION = 1


def _json_value(value):
    if is_dataclass(value):
        return _json_value(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _fps(meta):
    capture = meta.get('capture', {})
    for values in (capture, meta):
        if not isinstance(values, dict):
            continue
        for key in ('fps', 'frame_rate', 'framerate'):
            try:
                fps = float(values[key])
                if math.isfinite(fps) and fps > 0:
                    return fps
            except (KeyError, ValueError, TypeError):
                continue
    return None


def _derived_fields(subject, log):
    """Manifest fields recomputed from saved outputs, independent of CLI arguments."""
    import numpy as np
    requested = log.get('requested_steps', list(CONVERSION_STEPS))
    stages = {name: {'status': 'complete' if name in log['steps_completed'] else 'pending',
                     'completed_at': log['steps_completed'].get(name)} for name in requested}
    products = {name: relative for name, relative in {
        'images':'images', 'masks':'masks', 'calibration':'calibration.json',
        'camera_id_mapping':'camera_id_mapping.json', 'smplx':'smpl_params.npz',
        'mhr':'mhr_params.npz', **ANNOTATION_PATHS}.items()
        if name != 'subject_manifest' and (subject / relative).exists()}
    try:
        mapping = json.loads((subject / 'camera_id_mapping.json').read_text())
    except (OSError, ValueError):
        mapping = {}
    try:
        metrics = json.loads((subject / 'fit_metrics.json').read_text())
        coverage = {name: {key: fit.get(key) for key in ('status','finite_frames','total_frames',
                    'expected_frames','coverage_fraction')} for name, fit in metrics.get('fits', {}).items()}
    except (OSError, ValueError, TypeError, AttributeError):
        coverage = {}
    project = Path(__file__).resolve().parents[2]
    try:
        revision = subprocess.check_output(['git','-C',str(project),'rev-parse','HEAD'],
                                           text=True,stderr=subprocess.DEVNULL).strip()
        # A revision alone does not identify uncommitted tracked-code changes.
        dirty = bool(subprocess.check_output(['git','-C',str(project),'status','--porcelain',
                                              '--untracked-files=no'],
                                             text=True,stderr=subprocess.DEVNULL).strip())
    except (OSError, subprocess.CalledProcessError):
        revision = dirty = None
    packages = {}
    for name in ('numpy','Pillow','pillow-avif-plugin','PyYAML','torch','scipy'):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    from capture_processing.utils.fit_metrics import FIT_FILES
    fit_provenance = {}
    for kind, filename in FIT_FILES.items():
        path = subject / filename
        if not path.is_file():
            continue
        record = {'sha256': content_identity(path)}
        try:
            with np.load(path, allow_pickle=False) as data:
                for key in ('conversion_signature','estimation_signature','source','units','coordinate_system'):
                    if key in data:
                        record[key] = str(data[key].item())
        except (OSError, ValueError, KeyError, zipfile.BadZipFile):
            record['readable'] = False
        fit_provenance[kind] = record
    from capture_processing.utils.semantic_parsing import MODEL_ID, MODEL_REVISION, MODEL_HASHES
    semantic_provenance = ({'model_id': MODEL_ID, 'revision': MODEL_REVISION,
                            'sha256': MODEL_HASHES} if (subject / 'semantic_parsing').is_dir() else None)
    pipeline_code = {name: content_identity(project / name) for name in (
        'capture_processing/core/vci_pipeline.py', 'capture_processing/core/vci_resume.py',
        'capture_processing/utils/preprocessing_validation.py', 'capture_processing/utils/subject_validation.py',
        'capture_processing/utils/subject_manifest.py')}
    return {'stages': stages, 'products': products, 'camera_id_mapping': mapping,
            'fit_coverage': coverage, 'revision': revision, 'dirty': dirty, 'packages': packages,
            'pipeline_code': pipeline_code, 'fit_provenance': fit_provenance,
            'semantic_provenance': semantic_provenance}


@subject_locked
def write_subject_manifest(subject, args, *, result='complete', error=None):
    import numpy as np
    from capture_processing.utils.frames import discover_frames
    subject = Path(subject).resolve()
    source = args.input_path.expanduser().resolve()
    calibration_path = source / args.calibration
    try:
        calibration = json.loads(calibration_path.read_text())
    except (OSError, ValueError):
        calibration = {}
    calibration = calibration if isinstance(calibration, dict) else {}
    meta = calibration.get('meta', {})
    meta = meta if isinstance(meta, dict) else {}
    try:
        source_ids = [fid for fid, _ in discover_frames(source)]
    except (OSError, ValueError):
        source_ids = []
    fit_ids = None
    try:
        with np.load(subject / 'smpl_params.npz', allow_pickle=False) as data:
            fit_ids = (data['frame_ids'] if 'frame_ids' in data else np.arange(len(data['transl']))).tolist()
    except (OSError, ValueError, KeyError, zipfile.BadZipFile):
        pass
    fps = _fps(meta)
    log = load_processing_log(subject)
    derived = _derived_fields(subject, log)
    configuration = {key: _json_value(value) for key, value in vars(args).items()
                     if key not in ('input_path','out_dir','mamma_force')}
    configuration['mamma_force_requested_or_invalidated'] = bool(getattr(args,'mamma_force',False))
    source_id_set = set(source_ids)
    manifest = {
        'schema_version': SCHEMA_VERSION, 'kind':'processed-vci', 'subject_id':subject.name,
        'last_run': {'result':result, 'error':error},
        'complete': result == 'complete' and all(stage['status']=='complete' for stage in derived['stages'].values()),
        'source': {'recording_id':source.name, 'calibration_file':args.calibration,
                   'calibration_sha256':content_identity(calibration_path), 'metadata':meta,
                   'calibration_repair':calibration.get('calibration_repair')},
        'timeline': {'source_frame_ids':source_ids, 'fit_frame_ids':fit_ids, 'fps':fps,
                     'fps_source':'source calibration metadata' if fps else None,
                     'timestamps_seconds':None, 'timestamps_note':'No per-frame timestamps exported by this input contract.',
                     'missing_source_frame_ids':[fid for fid in range(max(source_ids)+1) if fid not in source_id_set] if source_ids else []},
        'coordinates': {'extrinsics':'world_to_camera', 'projection':'K @ (R @ X + T)',
                        'world_units':'meters', 'converted_mhr_units':'centimeters',
                        'distortion':'Inputs must already be undistorted; conversion does not undistort.',
                        'unit_provenance':'VCI input contract; units are not inferred from matrix magnitudes.'},
        'camera_id_mapping':derived['camera_id_mapping'], 'configuration':configuration, 'stages':derived['stages'],
        'products':derived['products'], 'fit_coverage':derived['fit_coverage'],
        'provenance': {'repository_revision':derived['revision'], 'repository_dirty':derived['dirty'],
                       'baseline_fit_dependencies':log.get('smplx_dependencies'),
                       'baseline_fit_signature':log.get('smplx_dependency_signature'),
                       'fitted_media_signature':log.get('smplx_media_signature'),
                       'converter_packages':derived['packages'], 'pipeline_code_sha256':derived['pipeline_code'],
                       'fit_archives':derived['fit_provenance'], 'semantic_model':derived['semantic_provenance']},
        'limitations':['Body fits and semantic labels are automatic annotations, not independent ground truth.',
                       'Calibration metadata repair cannot undo prior image undistortion.',
                       'Absent timing or coverage information is unknown, not zero.'],
    }
    return _write(subject, manifest)


def _write(subject, manifest):
    output = subject / 'subject_manifest.json'
    text = json.dumps(manifest, indent=2, allow_nan=False) + '\n'
    if not output.is_file() or output.read_text() != text:
        from capture_processing.utils.semantic_parsing import atomic_json
        atomic_json(output, manifest)
    return output


@subject_locked
def record_backfill(subject, steps, operation):
    """Mark stages completed by a standalone backfill tool and refresh the manifest.

    Source, timeline and configuration fields describe the last full pipeline run and
    are kept; fields derived from saved outputs are recomputed. Subjects without a
    manifest get one from their next normal pipeline run.
    """
    from datetime import datetime
    subject = Path(subject).resolve()
    log = load_processing_log(subject)
    requested = log.get('requested_steps')
    for step in steps:
        mark_step_complete(log, step)
        if requested is not None and step not in requested:
            requested.append(step)
    save_processing_log(subject, log)
    path = subject / 'subject_manifest.json'
    try:
        manifest = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(manifest, dict) or manifest.get('schema_version') != SCHEMA_VERSION:
        return None
    derived = _derived_fields(subject, log)
    manifest.update(stages=derived['stages'], products=derived['products'],
                    camera_id_mapping=derived['camera_id_mapping'], fit_coverage=derived['fit_coverage'])
    manifest['complete'] = (manifest.get('last_run', {}).get('result') == 'complete'
                            and all(stage['status'] == 'complete' for stage in derived['stages'].values()))
    provenance = manifest.setdefault('provenance', {})
    provenance.update(fit_archives=derived['fit_provenance'], semantic_model=derived['semantic_provenance'])
    provenance.setdefault('backfills', []).append({
        'operation': operation, 'steps': list(steps), 'at': datetime.now().isoformat(),
        'repository_revision': derived['revision'], 'repository_dirty': derived['dirty'],
        'pipeline_code_sha256': derived['pipeline_code']})
    return _write(subject, manifest)
