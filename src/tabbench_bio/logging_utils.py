"""Logging utilities for the benchmark pipeline."""

import logging
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path

from tabbench_bio.io_utils import archive_existing

LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s:%(funcName)s:%(lineno)d: %(message)s"

_RUN_FORMAT = logging.Formatter(fmt=LOG_FORMAT, datefmt="%Y-%m-%d %H:%M:%S")


@contextmanager
def run_file_logger(log_path: str):
    """Attach a per-run FileHandler for the duration of a context.

    All log records emitted by any logger in the process are written to
    *log_path* while the context is active.  The handler is always
    removed and closed on exit, even if an exception occurs.

    The handler is attached both to the root logger and directly to the
    ``tabbench_bio`` logger tree so that AutoGluon / Ray reconfigurations
    of the root logger (which happen during HPO) cannot silence our logs.

    Parameters
    ----------
    log_path : str
        Path to the log file.  The file is created (or overwritten) when
        the context is entered.
    """
    path = Path(log_path)
    archive_existing(path)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    handler = logging.FileHandler(temporary, mode="w", encoding="utf-8")
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(_RUN_FORMAT)

    # Attach to root and tabbench_bio directly so Ray/AG root reconfigurations
    # cannot drop our handler.
    _loggers = [logging.getLogger(), logging.getLogger("tabbench_bio")]
    for lg in _loggers:
        lg.addHandler(handler)
    try:
        yield str(temporary)
    finally:
        handler.flush()
        for lg in _loggers:
            lg.removeHandler(handler)
        handler.close()
        os.replace(temporary, path)
