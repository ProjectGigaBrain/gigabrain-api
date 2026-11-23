"""Connection to the memory and traceability service. Off unless asked for: see settings.py."""

from __future__ import annotations

import threading

from .hub import Hub, TurnHandle
from .runtime import HubRuntime
from .settings import HubConfigError, HubSettings

__all__ = ["Hub", "HubConfigError", "HubRuntime", "HubSettings", "TurnHandle", "connect", "flush_all"]

_runtimes: dict[HubSettings, HubRuntime] = {}
_lock = threading.Lock()


def connect(source: HubSettings | HubRuntime, project_key: str, provider) -> Hub | None:
    """The hub of one conversation. `source` is the settings (one runtime per process is shared) or, in
    tests, a runtime built with a fake transport. None if the service cannot be reached."""
    if isinstance(source, HubRuntime):
        runtime = source
    else:
        with _lock:
            runtime = _runtimes.get(source)
            if runtime is None:
                runtime = _runtimes[source] = HubRuntime(source)
    if not (runtime.settings.trace or runtime.settings.memory):
        return None
    return Hub.create(runtime, project_key, provider)


def flush_all(timeout: float = 10.0) -> None:
    with _lock:
        runtimes = list(_runtimes.values())
    for runtime in runtimes:
        runtime.flush(timeout)
