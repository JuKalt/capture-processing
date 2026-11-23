"""Read-only completion contracts for preprocessing products and cached reports."""
import json
from pathlib import Path
import zipfile


def requested_fits(args):
    from capture_processing.utils.mamma_additional_fit import requested
    fits = ['smplx']
    if requested(args):
        fits.append('smplx_confidence')
    if getattr(args, 'mhr_source', 'none') != 'none':
        fits.append('mhr')
    conversion = getattr(args, 'smplx_mhr_options', None)
    if conversion and conversion.enabled:
        fits.append('mhrfromsmplx')
    return fits


def converted_mhr_complete(path, signature=None):
    import numpy as np
    try:
        with np.load(path, allow_pickle=False) as data:
            ids = data['frame_ids']
            return (data['complete'].item() is True and ids.ndim == 1 and len(ids) > 0
                    and ids.dtype.kind in 'iu' and (ids >= 0).all()
                    and len(np.unique(ids)) == len(ids)
                    and all(data[name].shape == (len(ids), width) and np.isfinite(data[name]).all()
                            for name, width in [('lbs_model_params',204), ('identity_coeffs',45),
                                                ('face_expr_coeffs',72)])
                    and (signature is None or str(data['conversion_signature'].item()) == signature))
    except (OSError, ValueError, KeyError, TypeError, zipfile.BadZipFile):
        return False


def cached_metrics_complete(subject, fits, options=None):
    from capture_processing.utils.fit_metrics import validate_fit_metrics
    from capture_processing.utils.stage_cache import metric_artifacts, fit_artifacts, metrics_signature
    try:
        path = Path(subject) / 'fit_metrics.json'
        validate_fit_metrics(path, fits)
        report = json.loads(path.read_text())
        expected = {'cameras': list(options.cameras) if options and options.cameras else None,
                    'max_frames': options.max_frames if options else None}
        return (report.get('fit_artifacts') == fit_artifacts(subject)
                and report.get('cache_artifacts') == metric_artifacts(subject)
                and report.get('evaluation_options') == expected
                and (options is None or report.get('input_signature') == metrics_signature(subject, options)))
    except (RuntimeError, OSError, ValueError, TypeError, AttributeError):
        return False


def validate_subject(subject, required=('media', 'smplx', 'metrics'), *, deep=False):
    """Return named failures without changing outputs or launching inference.

    Portable checks validate artifacts. Runtime entry points additionally compare
    model/code dependency signatures before reusing cached stages.
    """
    import numpy as np
    from capture_processing.core.vci_resume import valid_smplx
    from capture_processing.utils.fit_metrics import FIT_FILES
    from capture_processing.utils.mhr import is_mhr_params_complete
    from capture_processing.utils.semantic_parsing import inventory, SemanticOptions, semantic_complete
    subject = Path(subject).expanduser().resolve()
    errors = {}
    required = list(dict.fromkeys(required))
    for stage in required:
        try:
            if stage == 'media':
                calibration = json.loads((subject / 'calibration.json').read_text())
                views = inventory(subject)
                if not isinstance(calibration, dict) or not calibration:
                    raise ValueError('Missing camera calibration')
                media_cameras = {path.name for path in (subject / 'images').iterdir() if path.is_dir()}
                if set(calibration) != media_cameras:
                    raise ValueError('Calibration and media camera directories differ')
                for camera in calibration.values():
                    k = np.asarray(camera['K'], dtype=float).reshape(3, 3)
                    r = np.asarray(camera['R'], dtype=float).reshape(3, 3)
                    t = np.asarray(camera['T'], dtype=float).reshape(3)
                    if (not all(np.isfinite(value).all() for value in (k, r, t))
                            or k[0, 0] <= 0 or k[1, 1] <= 0
                            or not np.allclose(k[2], [0, 0, 1])
                            or not np.allclose(r @ r.T, np.eye(3), atol=1e-3)
                            or not np.isclose(np.linalg.det(r), 1, atol=1e-3)):
                        raise ValueError('Invalid camera matrices')
                for view in views.values():
                    if deep:
                        from PIL import Image
                        with Image.open(view['image']) as image, Image.open(view['mask']) as mask:
                            image.load()
                            mask.load()
                            if image.size != mask.size:
                                raise ValueError('Image/mask dimensions differ')
                if deep:
                    from capture_processing.core.vci_resume import _file_digest
                    for path in (subject / '.media_state').rglob('*.json'):
                        receipt = json.loads(path.read_text())
                        for name, record in receipt['outputs'].items():
                            output = (subject / name).resolve()
                            if not output.is_relative_to(subject) or _file_digest(output) != record['sha256']:
                                raise ValueError('Media receipt content mismatch')
            elif stage in ('smplx', 'smplx_confidence'):
                with np.load(subject / FIT_FILES[stage], allow_pickle=False) as data:
                    count = len(data['transl'])
                if count <= 0 or not valid_smplx(subject / FIT_FILES[stage], count):
                    raise ValueError('Invalid SMPL-X parameters')
                if stage == 'smplx_confidence':
                    from capture_processing.utils.mamma_additional_fit import confidence_fit_complete
                    if not confidence_fit_complete(subject):
                        raise ValueError('Incomplete confidence fit artifacts')
            elif stage == 'mhr':
                if not is_mhr_params_complete(subject / FIT_FILES[stage]):
                    raise ValueError('Incomplete MHR parameters')
            elif stage == 'mhrfromsmplx':
                if not converted_mhr_complete(subject / FIT_FILES[stage]):
                    raise ValueError('Incomplete converted MHR parameters')
            elif stage == 'metrics':
                fits = [name for name in required if name in FIT_FILES]
                if not cached_metrics_complete(subject, fits or ['smplx']):
                    raise ValueError('Incomplete, stale, or limited fit metrics')
            elif stage == 'semantics':
                if not semantic_complete(subject, SemanticOptions(verify=deep)):
                    raise ValueError('Incomplete semantic labels or incompatible semantic settings')
            elif stage == 'visualizations':
                from capture_processing.utils.stage_cache import rendering_complete
                kinds = [kind for kind in ('mhr', 'mhrfromsmplx') if kind in required]
                if not kinds:
                    kinds = [kind for kind in ('mhr', 'mhrfromsmplx') if (subject / FIT_FILES[kind]).is_file()]
                if not kinds:
                    raise ValueError('No MHR fits selected or present for visualization validation')
                for kind in kinds:
                    folder = 'mhr_visualization' if kind == 'mhr' else 'mhr_from_smplx_visualization'
                    manifest = json.loads((subject / folder / 'manifest.json').read_text())
                    if not rendering_complete(subject / folder, manifest['signature']):
                        raise ValueError(f'Incomplete {folder}')
            else:
                raise ValueError(f'Unknown stage: {stage}')
        except (OSError, ValueError, KeyError, TypeError, IndexError, RuntimeError,
                AttributeError, zipfile.BadZipFile) as exc:
            errors[stage] = str(exc)
    return {'subject': subject.name, 'required': required, 'complete': not errors, 'errors': errors}
