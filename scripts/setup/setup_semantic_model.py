#!/usr/bin/env python3
"""Download the pinned official HF Sapiens2 snapshot and verify its content hashes."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
from capture_processing.utils.semantic_parsing import MODEL_ID, MODEL_REVISION, MODEL_FILES, MODEL_HASHES, atomic_json


def checksum(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def setup(destination, check=False):
    destination = Path(destination).expanduser().resolve()
    receipt = destination / 'vci_model.json'
    if check:
        record = json.loads(receipt.read_text())
        if (record.get('model_id') != MODEL_ID or record.get('revision') != MODEL_REVISION
                or record.get('label_schema') != 'sapiens2_dome29' or record.get('sha256') != MODEL_HASHES):
            raise ValueError('Sapiens2 model receipt does not match the pinned snapshot')
        for name in MODEL_FILES:
            if checksum(destination / name) != record['sha256'][name]:
                raise ValueError(f'Invalid Sapiens2 asset checksum: {name}')
        config = json.loads((destination / 'config.json').read_text())
        if config.get('architectures') != ['Sapiens2ForSemanticSegmentation'] or len(config.get('id2label', {})) != 29:
            raise ValueError('Unexpected segmentation model architecture/taxonomy')
        record['verified_files'] = {name: [(destination / name).stat().st_size,
            (destination / name).stat().st_mtime_ns] for name in MODEL_FILES}
        atomic_json(receipt, record)
        return
    from huggingface_hub import HfApi, hf_hub_download
    info = HfApi().model_info(MODEL_ID, revision=MODEL_REVISION, files_metadata=True)
    expected = {f.rfilename: f.lfs.sha256 for f in info.siblings if f.lfs}
    destination.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for name in MODEL_FILES:
        path = Path(hf_hub_download(MODEL_ID, name, revision=MODEL_REVISION, local_dir=destination))
        hashes[name] = checksum(path)
        if hashes[name] != MODEL_HASHES[name] or (name in expected and hashes[name] != expected[name]):
            raise ValueError(f'Download checksum mismatch: {name}')
    config = json.loads((destination / 'config.json').read_text())
    if config.get('architectures') != ['Sapiens2ForSemanticSegmentation'] or len(config.get('id2label', {})) != 29:
        raise ValueError('Unexpected segmentation model architecture/taxonomy')
    atomic_json(receipt, {'model_id': MODEL_ID, 'revision': MODEL_REVISION,
                         'label_schema': 'sapiens2_dome29', 'sha256': hashes})
    setup(destination, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', type=Path, default=PROJECT / 'models/sapiens2-seg-1b')
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    try:
        setup(args.model_dir, args.check)
    except Exception as exc:
        parser.exit(1, f'Sapiens2 model setup failed: {exc}\n')
    print(f'Verified Sapiens2 assets: {args.model_dir}')


if __name__ == '__main__':
    main()
