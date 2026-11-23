from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from capture_processing.utils.subject_lock import SUBJECT_BUSY_EXIT, SubjectBusyError, subject_lock


class SubjectLockTests(unittest.TestCase):
    def test_reentrant_but_excludes_threads_and_processes_and_releases_on_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with subject_lock(root):
                with subject_lock(root):
                    pass
                with ThreadPoolExecutor(1) as pool:
                    def competing():
                        with subject_lock(root):
                            pass
                    with self.assertRaisesRegex(RuntimeError, 'already being processed'):
                        pool.submit(competing).result()
                code = 'from capture_processing.utils.subject_lock import subject_lock; import sys\nwith subject_lock(sys.argv[1]): pass'
                result = subprocess.run([sys.executable, '-c', code, tmp], capture_output=True,text=True)
                self.assertNotEqual(result.returncode,0)
                self.assertIn('already being processed',result.stderr)
            with self.assertRaises(ValueError):
                with subject_lock(root):
                    raise ValueError('interrupted')
            with subject_lock(root):
                self.assertTrue((root/'.preprocessing.lock').exists())

    def test_busy_subject_exits_with_retry_later_status(self):
        project = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with subject_lock(root / 'subject'):
                with ThreadPoolExecutor(1) as pool:
                    def competing():
                        with subject_lock(root / 'subject'):
                            pass
                    with self.assertRaises(SubjectBusyError):
                        pool.submit(competing).result()
                self.assertIn('"host"', (root / 'subject' / '.preprocessing.lock').read_text())
                # The batch entry point maps a held lock to EX_TEMPFAIL for schedulers.
                (root / 'input' / 'subject' / 'frame_0').mkdir(parents=True)
                result = subprocess.run([sys.executable, str(project / 'process_dataset.py'), str(root / 'input'), str(root),
                                         '--skip-images', '--mhr-source', 'none', '--no-semantic-parsing',
                                         '--no-mamma-confidence-fit', '--skip-smplx-to-mhr'], capture_output=True, text=True, cwd=project)
                self.assertEqual(result.returncode, SUBJECT_BUSY_EXIT, result.stderr[-2000:])
