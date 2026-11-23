#!/usr/bin/env python3
"""Run the existing VCI batch converter against exactly one source subject."""
import argparse
from contextlib import contextmanager
import os
from pathlib import Path
import subprocess
import sys
import tempfile

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


@contextmanager
def gpu_reservation(subject):
    """Keep the assigned GPU visible in nvidia-smi during CPU preprocessing."""
    mib = int(os.environ.get('VCI_GPU_RESERVATION_MIB', '0'))
    if mib < 0:
        raise ValueError('VCI_GPU_RESERVATION_MIB must be nonnegative')
    if not mib:
        yield
        return
    if not os.environ.get('CUDA_VISIBLE_DEVICES'):
        raise ValueError('GPU reservation requires CUDA_VISIBLE_DEVICES')
    import torch
    # The distributor exposes the assigned physical GPU as logical cuda:0.
    reservation = torch.empty(mib * 1024 * 1024, dtype=torch.uint8, device='cuda:0')
    try:
        torch.cuda.synchronize()
        print(f'GPU reservation for {subject}: {mib} MiB on '
              f"CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']} "
              f'(PID {os.getpid()}); held until subject exits', flush=True)
        yield
    finally:
        del reservation
        torch.cuda.empty_cache()


def run_subject(input_root, subject, output_root, options):
    if not subject or Path(subject).name != subject or subject in ('.', '..'):
        raise ValueError('subject must be one directory name')
    source = Path(input_root).expanduser().resolve() / subject
    if not source.is_dir() or not any(p.is_dir() for p in source.glob('frame_*')):
        raise ValueError(f'VCI subject missing or contains no frame_* directories: {source}')
    if "--resume-validation" in options or any(option.startswith("--resume-validation=") for option in options):
        from capture_processing.cli import parse_args
        from capture_processing.core.vci_resume import quick_subject_complete
        from capture_processing.utils.smplx_mhr import options_from_args
        args = parse_args([str(input_root), str(output_root), *options])
        args.input_path = source
        args.smplx_mhr_options = options_from_args(args)
        if quick_subject_complete(args, Path(output_root).expanduser().resolve() / subject):
            print(f"Quick resume: {subject} already complete; skipped before GPU allocation.", flush=True)
            return 0
    # The batch converter discovers child directories. A temporary parent exposes
    # just this subject without copying data or changing the source directory.
    with tempfile.TemporaryDirectory(prefix='vci-subject-') as temporary:
        (Path(temporary) / subject).symlink_to(source, target_is_directory=True)
        command = [sys.executable, str(PROJECT_ROOT / 'process_dataset.py'),
                   temporary, str(Path(output_root).expanduser().resolve()), *options]
        with gpu_reservation(subject):
            return subprocess.run(command, cwd=PROJECT_ROOT).returncode


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input_root', type=Path)
    parser.add_argument('subject')
    parser.add_argument('output_root', type=Path)
    parser.add_argument('options', nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    options = args.options[1:] if args.options[:1] == ['--'] else args.options
    try:
        return run_subject(args.input_root, args.subject, args.output_root, options)
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    raise SystemExit(main())
