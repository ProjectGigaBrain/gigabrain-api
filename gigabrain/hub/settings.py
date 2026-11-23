"""Configuration of the connection to the memory and traceability service (the hub).

Everything comes from environment variables, and nothing is on by default: without
GIGABRAIN_HUB_URL and the --hub flag, GigaBrain runs exactly as it did before the hub existed.

    GIGABRAIN_HUB_URL       base URL of the service, e.g. http://localhost:8002 (required)
    GIGABRAIN_HUB_TOKEN     bearer token, if the service asks for one (optional)
    GIGABRAIN_HUB_TRACE     send every agent step to the service (default: on)
    GIGABRAIN_HUB_MEMORY    keep decisions in the service and give the agents what it remembers (default: on)
    GIGABRAIN_HUB_USER      who the conversations belong to; there is no login yet (default: po)
    GIGABRAIN_HUB_MIN_SCORE a memory excerpt scoring less than this is not shown to an agent (default: 0.1)
    GIGABRAIN_HUB_PERSON_MEMORY  the Twin remembers what the PO says, with Mem0 (default: on; needs GIGABRAIN_HUB_MEMORY)
    GIGABRAIN_HUB_PERSON_MIN_SCORE  a fact about the PO scoring less than this is not shown to the Twin (default: 0.2)
    GIGABRAIN_HUB_CONTEXT_WINDOW  context window of the model, if known (default: 0 = unknown)
"""

from __future__ import annotations

import os
from dataclasses import dataclass


class HubConfigError(ValueError):
    pass


def _flag(value: str | None, default: bool) -> bool:
    if value is None or not value.strip():
        return default
    return value.strip().lower() not in ("0", "false", "no", "off")


@dataclass(frozen=True)
class HubSettings:
    url: str
    token: str | None = None
    trace: bool = True
    memory: bool = True
    user: str = "po"
    min_score: float = 0.1
    person_memory: bool = True
    person_min_score: float = 0.2
    recall_limit: int = 5
    briefing_limit: int = 8
    context_window: int = 0
    timeout: float = 5.0
    max_chars: int = 200_000

    @classmethod
    def from_env(cls, environ=None) -> HubSettings:
        env = os.environ if environ is None else environ
        url = (env.get("GIGABRAIN_HUB_URL") or "").strip()
        if not url:
            raise HubConfigError("GIGABRAIN_HUB_URL is not set: it is the address of the memory and traceability service.")
        try:
            min_score = float(env.get("GIGABRAIN_HUB_MIN_SCORE") or 0.1)
            person_min_score = float(env.get("GIGABRAIN_HUB_PERSON_MIN_SCORE") or 0.2)
            window = int(env.get("GIGABRAIN_HUB_CONTEXT_WINDOW") or 0)
        except ValueError as exc:
            raise HubConfigError(f"invalid number in the GIGABRAIN_HUB_* variables: {exc}") from exc
        return cls(
            url=url,
            token=(env.get("GIGABRAIN_HUB_TOKEN") or "").strip() or None,
            trace=_flag(env.get("GIGABRAIN_HUB_TRACE"), True),
            memory=_flag(env.get("GIGABRAIN_HUB_MEMORY"), True),
            user=(env.get("GIGABRAIN_HUB_USER") or "po").strip() or "po",
            min_score=min_score,
            person_memory=_flag(env.get("GIGABRAIN_HUB_PERSON_MEMORY"), True),
            person_min_score=person_min_score,
            context_window=window,
        )
