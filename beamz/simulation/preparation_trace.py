"""Opt-in preparation phase timings and process memory, without device barriers."""

import json
import os
import sys
import time
from contextlib import contextmanager
from functools import wraps
from pathlib import Path


def _memory():
    try:
        import resource
    except ImportError:
        return {"host_peak_bytes": None, "host_rss_bytes": None}

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak *= 1 if sys.platform == "darwin" else 1024
    rss = None
    if sys.platform.startswith("linux"):
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                rss = int(line.split()[1]) * 1024
                break
    return {"host_peak_bytes": peak, "host_rss_bytes": rss}


@contextmanager
def preparation_phase(name):
    """Append start/end/error records to BEAMZ_TRACE_PREPARATION when enabled.

    Timings cover host work and enqueues, not asynchronous device execution.
    Peak RSS is process-wide; nested records are deliberately not additive.
    Memory fields are null on platforms without the Unix resource module.
    """
    path = os.getenv("BEAMZ_TRACE_PREPARATION")
    if not path:
        yield
        return
    started = time.perf_counter()

    def emit(event, **extra):
        with open(path, "a") as stream:
            stream.write(
                json.dumps(
                    dict(
                        phase=name,
                        event=event,
                        pid=os.getpid(),
                        time_ns=time.time_ns(),
                        **_memory(),
                        **extra,
                    )
                )
                + "\n"
            )

    emit("start")
    try:
        yield
    except BaseException as exc:
        emit(
            "error",
            elapsed_s=time.perf_counter() - started,
            error_type=type(exc).__name__,
        )
        raise
    else:
        emit("end", elapsed_s=time.perf_counter() - started)


def trace_preparation(name):
    def decorate(function):
        @wraps(function)
        def traced(*args, **kwargs):
            with preparation_phase(name):
                return function(*args, **kwargs)

        return traced

    return decorate
