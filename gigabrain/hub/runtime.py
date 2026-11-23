"""What the whole process shares: the client, the queue that sends in the background and what
was already registered in the service.

Everything an agent does is sent from a single worker thread, in the order it happened, so a
slow or absent service never makes a conversation wait. The one exception is a memory search,
which the agent is waiting for anyway (see hub.py).

If the service is down, jobs are retried a couple of times and then dropped, with a warning:
GigaBrain's own log (log.db and logs/*.jsonl) keeps everything, so nothing is lost for good.

There are two lanes. The fast one carries the trace. The slow one carries what writes to the memory:
a document is read as a whole by an LLM and Mem0 extracts facts with another, which takes seconds, and
the trace must not wait behind that. A slow job first waits for the fast lane to empty, because it
points to a step the fast lane is still sending.
"""

from __future__ import annotations

import atexit
import hashlib
import logging
import queue
import threading
import time
import uuid
from typing import Callable

from .client import HubClient, HubError

log = logging.getLogger("gigabrain.hub")

#: Ids chosen by GigaBrain, the same every time, so that sending the same thing twice changes nothing.
NAMESPACE = uuid.uuid5(uuid.NAMESPACE_URL, "https://github.com/ProjectGigaBrain/gigabrain-api/hub")

AGENT_ROLES = {
    "twin": "Digital Twin: talks to the Product Owner and proposes the requirements",
    "leader": "Leader of the Consultative Council: sends each question to the right expert",
    "expert": "Expert of one business topic: answers from its knowledge and recommends",
    "critic": "Critic: scores what the other agents write",
}
RETRIES = (0.3, 1.0)


def stable_id(*parts) -> uuid.UUID:
    return uuid.uuid5(NAMESPACE, ":".join(str(p) for p in parts))


class HubRuntime:
    def __init__(self, settings, transport=None):
        self.settings = settings
        self.client = HubClient(settings, transport)
        self.dropped = 0
        self._jobs = {"fast": queue.Queue(), "slow": queue.Queue()}
        self._pending = {"fast": 0, "slow": 0}
        self._cond = threading.Condition()
        self._lock = threading.Lock()
        self._workers: dict[str, threading.Thread] = {}
        self._ids: dict[tuple, uuid.UUID] = {}  # what the service answered when an agent or a prompt was registered
        self._down_until = 0.0
        self._projects: dict[str, uuid.UUID] = {}
        self._models: dict[tuple[str, str], int] = {}

    # ------------------------------------------------------------------ the queue

    def submit(self, label: str, job: Callable[[], None], slow: bool = False) -> None:
        lane = "slow" if slow else "fast"
        with self._cond:
            self._pending[lane] += 1
        self._start_worker(lane)
        self._jobs[lane].put((label, job))

    def _start_worker(self, lane: str) -> None:
        with self._lock:
            if lane not in self._workers:
                worker = threading.Thread(target=self._run, args=(lane,), name=f"gigabrain-hub-{lane}", daemon=True)
                self._workers[lane] = worker
                worker.start()
                if len(self._workers) == 1:
                    atexit.register(self.flush, 5.0)  # a short command line run still delivers what it queued

    def _run(self, lane: str) -> None:
        while True:
            label, job = self._jobs[lane].get()
            try:
                if lane == "slow":
                    self.flush(self.settings.timeout * 4, lane="fast")  # the step this job points to goes first
                self._attempt(label, job)
            except Exception as exc:  # never let the worker die
                self.dropped += 1
                if self.dropped <= 5 or self.dropped % 100 == 0:  # a service that is down must not fill the log
                    log.warning("hub: could not send %s (%s); %d dropped so far, GigaBrain's own log has them all",
                                label, exc, self.dropped)
            finally:
                with self._cond:
                    self._pending[lane] -= 1
                    self._cond.notify_all()

    @staticmethod
    def _attempt(label: str, job: Callable[[], None]) -> None:
        for delay in (*RETRIES, None):
            try:
                job()
                return
            except HubError as exc:
                if delay is None or not exc.retryable:
                    raise
                time.sleep(delay)

    def flush(self, timeout: float = 10.0, lane: str | None = None) -> bool:
        """Wait until everything queued (in one lane, or in both) has been sent. False if it took longer than `timeout`."""
        lanes = [lane] if lane else list(self._pending)
        deadline = time.monotonic() + timeout
        with self._cond:
            while any(self._pending[name] for name in lanes):
                left = deadline - time.monotonic()
                if left <= 0:
                    return False
                self._cond.wait(left)
        return True

    # ------------------------------------------------------------------ the catalog

    def ensure_project(self, key: str) -> uuid.UUID:
        """The project of the service that matches a GigaBrain project. Needs the answer now."""
        with self._lock:
            known = self._projects.get(key)
        if known:
            return known
        answer = self.client.post("/projetos", {"id": stable_id("project", key), "nome": f"gigabrain:{key}"})
        project_id = uuid.UUID(str(answer["id"]))
        with self._lock:
            self._projects[key] = project_id
        return project_id

    def ensure_model(self, provider: str, name: str) -> int:
        with self._lock:
            known = self._models.get((provider, name))
        if known is not None:
            return known
        answer = self.client.post("/modelos", {"provedor": provider, "nome": name, "modalidade": "texto",
                                               "janela_contexto": self.settings.context_window})
        with self._lock:
            self._models[(provider, name)] = int(answer["id"])
        return self._models[(provider, name)]

    # The id of an agent or a prompt is the one the SERVICE answers with, not the one proposed here: it may
    # already have the same agent (by name) or the same prompt (by text) under another id, and a step that
    # points to an id it does not have is refused. So each one is registered the first time it is used (once
    # for the whole process) and the answer is kept.

    def _usable(self) -> bool:
        return time.monotonic() >= self._down_until

    def _trip(self, what: str, exc: Exception) -> None:
        """The service did not answer a registration: stop asking for a while, so a conversation never waits on it."""
        self._down_until = time.monotonic() + 30
        log.warning("hub: could not register %s (%s); not trying again for 30 s", what, exc)

    def register_agent(self, role: str, model_id: int) -> uuid.UUID:
        """The id of an agent in the service (registering it, and its first version, the first time)."""
        key = ("agent", role)
        with self._lock:
            known = self._ids.get(key)
        if known:
            return known
        if self._usable():
            try:
                answer = self.client.post("/agentes", {"id": stable_id("agent", role), "nome": f"gigabrain.{role}",
                                                       "funcao": AGENT_ROLES[role]})
                agent_id = uuid.UUID(str(answer["id"]))
                self.client.post(f"/agentes/{agent_id}/versoes",
                                 {"id": stable_id("agent-version", role, 1), "versao": 1, "modelo_id": model_id})
            except (HubError, KeyError, ValueError) as exc:
                self._trip(f"the {role} agent", exc)
            else:
                with self._lock:
                    self._ids[key] = agent_id
                return agent_id
        return stable_id("agent", role)  # an execution is accepted with any id here; the link just will not be found

    def register_prompt(self, role: str, text: str) -> uuid.UUID | None:
        """The id of a prompt in the service, by what it says: the same text is the same version, whoever sent it.
        None if the service could not be asked (the step then goes without a prompt)."""
        key = f"gigabrain.{role}"
        token = ("prompt", key, hashlib.sha256(text.encode("utf-8")).hexdigest())
        with self._lock:
            known = self._ids.get(token)
        if known:
            return known
        if not self._usable():
            return None
        try:
            answer = self.client.post("/prompts", {"id": stable_id(*token), "chave": key, "tipo": "system", "template": text})
            prompt_id = uuid.UUID(str(answer["id"]))
        except (HubError, KeyError, ValueError) as exc:
            self._trip(f"the prompt {key}", exc)
            return None
        with self._lock:
            self._ids[token] = prompt_id
        return prompt_id
