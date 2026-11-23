"""Reentrant per-thread ownership of a subject's mutating preprocessing stages."""
from contextlib import contextmanager
from functools import wraps
import json
import os
from pathlib import Path
import socket
import threading

_LOCAL = threading.local()
# Exit status for "another process holds this subject": a scheduler should
# retry later instead of counting the attempt as a processing failure.
SUBJECT_BUSY_EXIT = 75


class SubjectBusyError(RuntimeError):
    """Another process currently owns the subject's preprocessing lock."""


@contextmanager
def subject_lock(subject, operation='preprocessing'):
    import fcntl
    subject = Path(subject).expanduser().resolve()
    held = getattr(_LOCAL, 'held', set())
    if subject in held:
        yield
        return
    subject.mkdir(parents=True, exist_ok=True)
    path = subject / '.preprocessing.lock'
    # Never unlink the lock file: another process may already hold its inode.
    with path.open('a+') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            stream.seek(0)
            owner = stream.read().strip() or 'owner metadata unavailable'
            raise SubjectBusyError(f'Subject is already being processed: {subject}; {owner}') from exc
        owner = json.dumps({'host': socket.gethostname(), 'pid': os.getpid(), 'operation': operation})
        stream.seek(0)
        if stream.read() != owner:
            stream.seek(0)
            stream.truncate()
            stream.write(owner)
            stream.flush()
        _LOCAL.held = held | {subject}
        try:
            yield
        finally:
            _LOCAL.held = held
            fcntl.flock(stream, fcntl.LOCK_UN)


def subject_locked(function=None, *, argument=0):
    def decorate(fn):
        @wraps(fn)
        def locked(*args, **kwargs):
            import inspect
            bound = inspect.signature(fn).bind(*args, **kwargs)
            name = list(inspect.signature(fn).parameters)[argument]
            with subject_lock(bound.arguments[name], fn.__name__):
                return fn(*args, **kwargs)
        return locked
    return decorate(function) if function else decorate
