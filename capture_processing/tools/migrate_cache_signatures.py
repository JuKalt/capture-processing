"""Re-sign completed stages with worker-independent (version 2) cache signatures.

Earlier signatures embedded each worker's checkout/model paths and file mtimes,
so a subject finished on one machine looked stale on every other one. Since
the format changed, every existing signature differs once. This tool adopts a
complete earlier result instead of recomputing it when its artifact is at least
as new as every input it was computed from (and, for the SMPL-X -> MHR
conversion, when the recorded source hash and MHR revision still match).
Results that fail these checks are left alone and recomputed by the pipeline.

Run it once per output root, on a machine with the standard checkouts (MAMMA,
external/MHR, external/sam3d-body-rerun, Sapiens/SAM3D model caches), while no
preprocessing job is running. Without --apply nothing is written.

    python -m capture_processing.tools.migrate_cache_signatures --output-root /mnt/.../processed
    python -m capture_processing.tools.migrate_cache_signatures --output-root /mnt/.../processed --apply
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
from pathlib import Path
import sys
import zipfile

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from capture_processing.utils.stage_cache import subject_inputs  # noqa: E402
from capture_processing.utils.subject_lock import SubjectBusyError, subject_lock  # noqa: E402

RENDERINGS = (('mhr_visualization', 'mhr', 'mhr_params.npz'),
              ('mhr_from_smplx_visualization', 'mhrfromsmplx', 'mhr_from_smplx_params.npz'))


def newest(paths):
    return max((path.stat().st_mtime_ns for path in paths if path.is_file()), default=0)


def not_older(artifact, inputs):
    return artifact.stat().st_mtime_ns >= newest(inputs)


def write_json(path, value, indent=2):
    """Atomic rewrite that keeps the artifact's mtime (its freshness evidence)."""
    stat = path.stat()
    temporary = path.with_name(path.name + '.migrate-tmp')
    temporary.write_text(json.dumps(value, indent=indent, allow_nan=False) + '\n')
    temporary.replace(path)
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))


def resign_npz(path, key, signature):
    import numpy as np
    stat = path.stat()
    with np.load(path, allow_pickle=False) as data:
        arrays = {name: data[name] for name in data.files}
    arrays[key] = np.asarray(signature)
    temporary = path.with_name(path.stem + '.migrate-tmp.npz')
    np.savez(temporary, **arrays)
    temporary.replace(path)
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))


class Migration:
    def __init__(self, apply):
        from capture_processing.utils.silhouette_metrics import SilhouetteOptions, resolve_paths
        from capture_processing.utils.mamma import resolve_mamma_repo
        self.apply = apply
        self.options = SilhouetteOptions()
        self.official, self.sam3d, self.model = resolve_paths(self.options)
        self.mamma = resolve_mamma_repo(None)
        self.counts = {}

    def record(self, subject, stage, outcome, detail=''):
        self.counts[(stage, outcome)] = self.counts.get((stage, outcome), 0) + 1
        if outcome != 'current':
            print(f'{subject.name}: {stage}: {outcome}{" (" + detail + ")" if detail else ""}', flush=True)

    def decide(self, subject, stage, stored, new, fresh, detail='inputs changed after it was written'):
        """current, adopt or stale; the caller rewrites only for adopt with --apply."""
        if stored == new:
            self.record(subject, stage, 'current')
            return 'current'
        if not fresh:
            self.record(subject, stage, 'stale, will recompute', detail)
            return 'stale'
        self.record(subject, stage, 'adopted' if self.apply else 'would adopt')
        return 'adopt'

    def adopt(self, subject, stage, stored, new, fresh, detail='inputs changed after it was written'):
        return self.decide(subject, stage, stored, new, fresh, detail) == 'adopt' and self.apply

    def subject(self, subject):
        stages = (self.mhr, self.conversion, self.renderings, self.confidence, self.metrics)
        try:
            # A report-only run writes nothing, not even lock metadata.
            with subject_lock(subject, 'cache signature migration') if self.apply else contextlib.nullcontext():
                for stage in stages:
                    try:
                        stage(subject)
                    except (OSError, ValueError, KeyError, TypeError, IndexError, zipfile.BadZipFile) as exc:
                        self.record(subject, stage.__name__, 'skipped', str(exc))
        except SubjectBusyError as exc:
            self.record(subject, 'subject', 'skipped', str(exc))

    def mhr(self, subject):
        import numpy as np
        from capture_processing.utils.mhr import DEFAULT_MHR_ITERATIONS, estimation_signature, is_mhr_params_complete
        path = subject / 'mhr_params.npz'
        if not is_mhr_params_complete(path):
            return
        with np.load(path, allow_pickle=False) as data:
            stored = str(data['estimation_signature'].item()) if 'estimation_signature' in data.files else ''
        new = estimation_signature(subject, self.sam3d, None, DEFAULT_MHR_ITERATIONS)
        fresh = not_older(path, subject_inputs(subject, fit_files=[]))
        if self.adopt(subject, 'mhr', stored, new, fresh):
            resign_npz(path, 'estimation_signature', new)

    def conversion(self, subject):
        import numpy as np
        from capture_processing.utils.smplx_mhr import conversion_signature
        from capture_processing.utils.subject_validation import converted_mhr_complete
        path = subject / 'mhr_from_smplx_params.npz'
        if not converted_mhr_complete(path):
            return
        with np.load(path, allow_pickle=False) as data:
            stored = str(data['conversion_signature'].item())
        new = conversion_signature(subject / 'smpl_params.npz', self.model, self.official)
        old, current = json.loads(stored), json.loads(new)
        # The old record names its exact source fit and MHR revision.
        same = (old.get('source_sha256') == current['source_sha256'] and old.get('revision') == current['revision']
                and Path(str(old.get('model', ''))).name == current['model']
                and old.get('model_size') in (None, self.model.stat().st_size))
        if self.adopt(subject, 'mhrfromsmplx', stored, new, same, 'source fit, MHR revision or model differ'):
            resign_npz(path, 'conversion_signature', new)

    def renderings(self, subject):
        from capture_processing.utils.mhr import VISUALIZATION_SETTINGS
        from capture_processing.utils.stage_cache import rendering_complete, rendering_signature
        for folder, kind, fit in RENDERINGS:
            path = subject / folder / 'manifest.json'
            if not path.is_file():
                continue
            manifest = json.loads(path.read_text())
            stored = manifest.get('signature')
            if not rendering_complete(subject / folder, stored):
                continue
            repo = self.sam3d if kind == 'mhr' else self.official
            new = rendering_signature(subject, repo, kind, dict(VISUALIZATION_SETTINGS))
            inputs = subject_inputs(subject, media=('images',), fit_files=[fit])
            if self.adopt(subject, folder, stored, new, not_older(path, inputs)):
                manifest['signature'] = new
                write_json(path, manifest)

    def confidence(self, subject):
        from capture_processing.utils.mamma_additional_fit import MANIFEST, confidence_fit_complete, inputs
        path = subject / MANIFEST
        if not path.is_file() or not confidence_fit_complete(subject):
            return
        manifest = json.loads(path.read_text())
        baseline, config, _, cap, landmarks, new = inputs(subject, self.mamma, 'vci')
        observed = [subject / 'smpl_params.npz', config, baseline.parent / 'run_args.json',
                    *cap.glob('images/gt/*.npz'), *landmarks.glob('images/*.npz')]
        if self.adopt(subject, 'smplx_confidence', manifest.get('signature'), new, not_older(path, observed)):
            manifest['signature'] = new
            write_json(path, manifest)

    def metrics(self, subject):
        from capture_processing.utils.fit_metrics import FIT_FILES, _expected_ids, _fit
        from capture_processing.utils.silhouette_metrics import _cache_matches, silhouette_signatures
        from capture_processing.utils.stage_cache import fit_artifacts, metric_artifacts, metrics_signature
        path = subject / 'fit_metrics.json'
        if not path.is_file():
            return
        report = json.loads(path.read_text())
        if report.get('complete') is not True or report.get('evaluation_options') != {'cameras': None, 'max_frames': None}:
            return
        expected = _expected_ids(subject)
        fits = {kind: _fit(subject, kind, expected) for kind in FIT_FILES
                if kind != 'smplx_confidence' or (subject / FIT_FILES[kind]).is_file()
                or (subject / 'mamma_confidence_fit.json').is_file()}
        views, _, reason, selection, signatures = silhouette_signatures(subject, fits, self.options)
        if reason:
            return
        adopted_all = True
        for kind, signature in signatures.items():
            cache = subject / f'silhouette_metrics.{kind}.json'
            if not cache.is_file():
                adopted_all = False
                continue
            candidate = json.loads(cache.read_text())
            if not _cache_matches(candidate, candidate.get('signature'), views):
                adopted_all = False
                continue
            fresh = not_older(cache, [subject / FIT_FILES[kind]])
            outcome = self.decide(subject, f'silhouette:{kind}', candidate['signature'], signature, fresh)
            if outcome == 'stale':
                adopted_all = False
            elif outcome == 'adopt' and self.apply:
                candidate['signature'] = signature
                write_json(cache, candidate)
        new = metrics_signature(subject, self.options)
        # Every per-fit score must carry over, otherwise the report is rebuilt anyway.
        fresh = adopted_all and not_older(path, subject_inputs(subject))
        if self.adopt(subject, 'fit_metrics', report.get('input_signature'), new, fresh,
                       'inputs changed after it was written or a silhouette score was not adopted'):
            report['input_signature'] = new
            report['silhouette_evaluation']['signature'] = selection
            for fit in report['fits'].values():
                if 'silhouette_selection_signature' in fit:
                    fit['silhouette_selection_signature'] = selection
            report['cache_artifacts'] = metric_artifacts(subject)
            report['fit_artifacts'] = fit_artifacts(subject)
            write_json(path, report)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output-root', type=Path, required=True, help='Processed dataset root (one folder per subject)')
    parser.add_argument('--subjects', nargs='+', help='Only these subject folder names')
    parser.add_argument('--apply', action='store_true', help='Rewrite signatures (default: report only)')
    args = parser.parse_args(argv)
    root = args.output_root.expanduser().resolve()
    subjects = ([root / name for name in args.subjects] if args.subjects else
                sorted(path for path in root.iterdir() if path.is_dir() and (path / 'calibration.json').is_file()))
    migration = Migration(args.apply)
    for subject in subjects:
        if not (subject / 'calibration.json').is_file():
            parser.error(f'Not a processed subject: {subject}')
        migration.subject(subject)
    print('\nSummary:')
    for (stage, outcome), count in sorted(migration.counts.items()):
        print(f'  {stage:32s} {outcome:24s} {count}')
    if not args.apply:
        print('\nDry run: nothing was written. Re-run with --apply to adopt.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
