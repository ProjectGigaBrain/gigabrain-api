"""HTTP client of the service. Synchronous, because the rest of GigaBrain is."""

from __future__ import annotations

import json


class HubError(RuntimeError):
    """The service could not be reached or refused the request. `status` is None when it was not reached."""

    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status

    @property
    def retryable(self) -> bool:
        return self.status is None or self.status >= 500


class HubClient:
    def __init__(self, settings, transport=None):
        import httpx  # here: a run without the hub must not need it

        self._httpx = httpx
        headers = {"Content-Type": "application/json"}
        if settings.token:
            headers["Authorization"] = f"Bearer {settings.token}"
        self._timeout = settings.timeout
        self._http = httpx.Client(base_url=settings.url.rstrip("/"), headers=headers, timeout=settings.timeout,
                                  transport=transport)

    def request(self, method: str, path: str, payload=None, timeout: float | None = None):
        # default=str: ids and dates inside the payloads are sent as text, never fail the request
        body = None if payload is None else json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
        try:
            response = self._http.request(method, path, content=body, timeout=timeout or self._timeout)
        except self._httpx.HTTPError as exc:
            raise HubError(f"{method} {path}: {exc}") from exc
        if response.status_code >= 400:
            raise HubError(f"{method} {path}: HTTP {response.status_code}: {response.text[:300]}", response.status_code)
        return response.json() if response.content else None

    def post(self, path: str, payload=None, timeout: float | None = None):
        return self.request("POST", path, payload, timeout)

    def patch(self, path: str, payload=None, timeout: float | None = None):
        return self.request("PATCH", path, payload, timeout)

    def close(self) -> None:
        self._http.close()
