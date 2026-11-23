"""Store MAMMA hand poses in the flat-hand SMPL-X convention.

MAMMA fits with ``flat_hand_mean=False``: its hand poses are offsets from
SMPL-X's mean (relaxed, curled) hand. Every consumer of ``smpl_params.npz``
builds SMPL-X with ``flat_hand_mean=True``, where zero is a flat hand. Read
that way, MAMMA's fingers lose their curl and bend backwards. Adding the
model's mean hand pose expresses the same hands in the flat convention.

The conversion runs after export rather than inside it: the export code is
part of the MAMMA fit signature, and changing it would refit every subject.
Converted files carry ``flat_hand_mean=True``, so the step is idempotent and
existing subjects are converted on their next run without refitting.
"""
from __future__ import annotations

import os
from pathlib import Path
import zipfile

MARKER = 'flat_hand_mean'


def hands_flat(path) -> bool:
    """Whether a parameter file declares flat-hand-convention hand poses."""
    import numpy as np
    try:
        with np.load(path, allow_pickle=False) as data:
            return MARKER in data.files and bool(data[MARKER])
    except (OSError, ValueError, EOFError, zipfile.BadZipFile):
        return False


def needs_hand_conversion(subject) -> bool:
    """A subject whose MAMMA export still uses MAMMA's own hand convention."""
    subject = Path(subject)
    path = subject / 'smpl_params.npz'
    return path.is_file() and (subject / 'mamma_output').is_dir() and not hands_flat(path)


def mamma_hands_mean(repo=None, models=None):
    """Mean hand poses (left, right) of the SMPL-X model MAMMA fits with.

    ``models`` is MAMMA's ``--smplx-models`` folder; by default it is resolved
    from the MAMMA checkout like the fit itself.
    """
    import numpy as np
    if models is None:
        from capture_processing.utils.mamma import _resolve_mamma_asset_path
        models = _resolve_mamma_asset_path(Path(repo), os.environ, 'MAMMA_SMPLX_LOCKHEAD_MODELS',
                                           'data/body_models/smplx_locked_head')
    # The mean hand is MANO's and identical in every gendered SMPL-X model.
    with np.load(Path(models) / 'smplx' / 'SMPLX_NEUTRAL.npz', allow_pickle=False) as model:
        return (np.asarray(model['hands_meanl'], dtype=np.float32).reshape(45),
                np.asarray(model['hands_meanr'], dtype=np.float32).reshape(45))


def convert_mamma_hands(path, hands_mean) -> bool:
    """Rewrite a MAMMA export in the flat-hand convention; False if already flat."""
    import numpy as np
    path = Path(path)
    if hands_flat(path):
        return False
    with np.load(path, allow_pickle=False) as data:
        arrays = {name: data[name] for name in data.files}
    left, right = hands_mean
    arrays['left_hand_pose'] = (arrays['left_hand_pose'] + left).astype(np.float32)
    arrays['right_hand_pose'] = (arrays['right_hand_pose'] + right).astype(np.float32)
    arrays[MARKER] = np.asarray(True)
    temporary = path.with_name(path.name + '.hands.tmp.npz')
    np.savez(temporary, **arrays)
    temporary.replace(path)
    return True


def normalize_subject_hands(subject, repo, tag='vci') -> bool:
    """Convert a subject's MAMMA ``smpl_params.npz`` in place.

    The confidence-filtered fit records the baseline export's file stamp in its
    signature. Converting only changes how the same fit is stored, so a fit that
    was current before the conversion is re-signed instead of being refitted.
    """
    import json
    from capture_processing.utils.mamma_additional_fit import MANIFEST, inputs
    subject = Path(subject)
    path = subject / 'smpl_params.npz'
    if not needs_hand_conversion(subject):
        return False
    manifest_path = subject / MANIFEST
    def signature():
        try:
            return inputs(subject, repo, tag)[-1]
        except (OSError, ValueError, KeyError):
            return None
    try:
        manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else None
    except (OSError, ValueError):
        manifest = None
    current = manifest is not None and manifest.get('signature') is not None and manifest.get('signature') == signature()
    convert_mamma_hands(path, mamma_hands_mean(repo))
    if current:
        manifest['signature'] = signature()
        temporary = manifest_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(manifest, indent=2) + '\n')
        temporary.replace(manifest_path)
    print(f'Converted MAMMA hand poses to the flat-hand convention: {path}', flush=True)
    return True
