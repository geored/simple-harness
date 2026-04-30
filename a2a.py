"""
a2a.py — Agent-to-Agent Communication Library
A threading-based, production-ready library for multi-agent coordination.
"""

from __future__ import annotations

import logging
import threading
import time
import urllib.request
import urllib.error
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum, auto
from queue import Queue, Full
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Custom Exceptions
# ---------------------------------------------------------------------------

class A2AError(Exception):
    """Base exception for all A2A library errors."""


class A2ACommsError(A2AError):
    """Raised when an agent-to-agent HTTP request fails."""


class A2ATimeoutError(A2ACommsError):
    """Raised when a request to a remote agent times out."""


class AgentNotAvailableError(A2AError):
    """Raised when the target agent is not found or its circuit is open."""


class QueueFullError(A2AError):
    """Raised when the task queue has reached its capacity limit."""


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

@dataclass
class _RegistryEntry:
    endpoint: str
    metadata: dict[str, Any]
    expires_at: float  # monotonic timestamp


class Registry:
    """
    Thread-safe agent registry with TTL-based expiry.

    All public methods acquire a single RLock so that compound operations
    (e.g. check-then-write inside `register`) are atomic.  RLock is chosen
    over Lock so that the same thread can safely call nested public methods
    without deadlocking.
    """

    def __init__(self, default_ttl: float = 300.0) -> None:
        self._default_ttl = default_ttl
        self._entries: dict[str, _RegistryEntry] = {}
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register(
        self,
        agent_id: str,
        endpoint: str,
        metadata: Optional[dict[str, Any]] = None,
        ttl: Optional[float] = None,
    ) -> None:
        """Register or refresh an agent entry."""
        expires_at = time.monotonic() + (ttl if ttl is not None else self._default_ttl)
        with self._lock:
            self._entries[agent_id] = _RegistryEntry(
                endpoint=endpoint,
                metadata=metadata or {},
                expires_at=expires_at,
            )
        logger.debug("Registered agent '%s' → %s", agent_id, endpoint)

    def lookup(self, agent_id: str) -> _RegistryEntry:
        """Return a live entry or raise AgentNotAvailableError."""
        with self._lock:
            entry = self._entries.get(agent_id)
            if entry is None or time.monotonic() > entry.expires_at:
                # Remove stale entry while we hold the lock.
                self._entries.pop(agent_id, None)
                raise AgentNotAvailableError(
                    f"Agent '{agent_id}' not found or TTL expired."
                )
            return entry

    def deregister(self, agent_id: str) -> None:
        """Remove an agent from the registry (idempotent)."""
        with self._lock:
            self._entries.pop(agent_id, None)
        logger.debug("Deregistered agent '%s'", agent_id)

    def evict_expired(self) -> int:
        """Purge all expired entries; returns count removed."""
        now = time.monotonic()
        with self._lock:
            expired = [k for k, v in self._entries.items() if now > v.expires_at]
            for k in expired:
                del self._entries[k]
        if expired:
            logger.debug("Evicted %d expired registry entries.", len(expired))
        return len(expired)

    def list_agents(self) -> list[str]:
        """Return IDs of all currently live agents."""
        now = time.monotonic()
        with self._lock:
            return [k for k, v in self._entries.items() if now <= v.expires_at]


# ---------------------------------------------------------------------------
# Circuit Breaker
# ---------------------------------------------------------------------------

class _CBState(Enum):
    CLOSED = auto()    # normal operation
    OPEN = auto()      # failing; reject requests immediately
    HALF_OPEN = auto() # probe to check recovery


@dataclass
class CircuitBreaker:
    """
    Per-agent circuit breaker (Closed → Open → Half-Open → Closed).

    State transitions are protected by an RLock.  The breaker trips to OPEN
    after `failure_threshold` consecutive failures and stays open for
    `recovery_timeout` seconds before allowing a single probe request.
    """

    failure_threshold: int = 3
    recovery_timeout: float = 30.0

    _state: _CBState = field(default=_CBState.CLOSED, init=False)
    _failures: int = field(default=0, init=False)
    _opened_at: float = field(default=0.0, init=False)
    _lock: threading.RLock = field(default_factory=threading.RLock, init=False)

    def allow_request(self) -> bool:
        """Return True if the caller may proceed with a request."""
        with self._lock:
            if self._state is _CBState.CLOSED:
                return True
            if self._state is _CBState.OPEN:
                if time.monotonic() - self._opened_at >= self.recovery_timeout:
                    self._state = _CBState.HALF_OPEN
                    logger.debug("CircuitBreaker → HALF_OPEN (probing)")
                    return True
                return False
            # HALF_OPEN: one probe at a time (already transitioned above)
            return True

    def record_success(self) -> None:
        with self._lock:
            self._failures = 0
            if self._state is not _CBState.CLOSED:
                logger.debug("CircuitBreaker → CLOSED")
            self._state = _CBState.CLOSED

    def record_failure(self) -> None:
        with self._lock:
            self._failures += 1
            if self._state is _CBState.HALF_OPEN or self._failures >= self.failure_threshold:
                self._state = _CBState.OPEN
                self._opened_at = time.monotonic()
                logger.warning(
                    "CircuitBreaker → OPEN after %d failure(s)", self._failures
                )

    @property
    def state(self) -> str:
        return self._state.name


# ---------------------------------------------------------------------------
# Bounded Task Queue
# ---------------------------------------------------------------------------

class BoundedTaskQueue:
    """
    A fixed-capacity queue that raises QueueFullError instead of blocking
    when at capacity, providing explicit backpressure to callers.
    """

    def __init__(self, maxsize: int = 256) -> None:
        self._queue: Queue[Callable[[], Any]] = Queue(maxsize=maxsize)

    def submit(self, task: Callable[[], Any]) -> None:
        """Enqueue *task* or raise QueueFullError immediately."""
        try:
            self._queue.put_nowait(task)
        except Full as exc:
            raise QueueFullError(
                f"Task queue is full ({self._queue.maxsize} slots)."
            ) from exc

    def get(self) -> Callable[[], Any]:
        return self._queue.get()

    def task_done(self) -> None:
        self._queue.task_done()

    def join(self) -> None:
        self._queue.join()

    @property
    def size(self) -> int:
        return self._queue.qsize()


# ---------------------------------------------------------------------------
# A2A HTTP Client
# ---------------------------------------------------------------------------

@dataclass
class RetryPolicy:
    max_attempts: int = 3
    backoff_base: float = 0.5   # seconds; delay doubles each retry
    backoff_max: float = 8.0


class A2AClient:
    """
    Sends JSON messages to remote agents over HTTP with timeout, retry,
    and circuit-breaker integration.
    """

    def __init__(
        self,
        registry: Registry,
        timeout: float = 5.0,
        retry_policy: Optional[RetryPolicy] = None,
    ) -> None:
        self._registry = registry
        self._timeout = timeout
        self._retry = retry_policy or RetryPolicy()
        # One circuit breaker per remote agent_id.
        self._breakers: dict[str, CircuitBreaker] = {}
        self._breakers_lock = threading.Lock()

    def _get_breaker(self, agent_id: str) -> CircuitBreaker:
        with self._breakers_lock:
            if agent_id not in self._breakers:
                self._breakers[agent_id] = CircuitBreaker()
            return self._breakers[agent_id]

    def send(
        self,
        agent_id: str,
        payload: dict[str, Any],
        path: str = "/message",
    ) -> dict[str, Any]:
        """
        Deliver *payload* to *agent_id*.  Returns the parsed JSON response.

        Raises:
            AgentNotAvailableError – agent unknown, TTL expired, or circuit open.
            A2ATimeoutError        – request timed out on every attempt.
            A2ACommsError          – non-timeout HTTP or network error.
        """
        entry = self._registry.lookup(agent_id)  # raises AgentNotAvailableError
        breaker = self._get_breaker(agent_id)

        if not breaker.allow_request():
            raise AgentNotAvailableError(
                f"Circuit breaker for '{agent_id}' is OPEN; requests suppressed."
            )

        url = entry.endpoint.rstrip("/") + path
        body = json.dumps(payload).encode()
        policy = self._retry
        last_exc: Exception = A2ACommsError("No attempts made.")

        for attempt in range(1, policy.max_attempts + 1):
            try:
                req = urllib.request.Request(
                    url,
                    data=body,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                    response_body = resp.read()

                result: dict[str, Any] = json.loads(response_body)
                breaker.record_success()
                return result

            except urllib.error.URLError as exc:
                # urllib wraps socket.timeout inside URLError.reason.
                is_timeout = isinstance(
                    getattr(exc, "reason", None), TimeoutError
                ) or "timed out" in str(exc).lower()

                if is_timeout:
                    last_exc = A2ATimeoutError(
                        f"Request to '{agent_id}' timed out (attempt {attempt})."
                    )
                else:
                    last_exc = A2ACommsError(
                        f"Request to '{agent_id}' failed (attempt {attempt}): {exc}"
                    )

                logger.warning("%s", last_exc)
                breaker.record_failure()

                if attempt < policy.max_attempts:
                    delay = min(
                        policy.backoff_base * (2 ** (attempt - 1)),
                        policy.backoff_max,
                    )
                    time.sleep(delay)

        raise last_exc


# ---------------------------------------------------------------------------
# Agent Worker Pool
# ---------------------------------------------------------------------------

class AgentWorkerPool:
    """
    Manages a pool of worker threads that drain a BoundedTaskQueue.

    Usage:
        pool = AgentWorkerPool(workers=4, queue_size=128)
        pool.start()
        pool.submit(my_callable)
        pool.shutdown(wait=True)
    """

    def __init__(self, workers: int = 4, queue_size: int = 256) -> None:
        self._queue = BoundedTaskQueue(maxsize=queue_size)
        self._executor = ThreadPoolExecutor(max_workers=workers)
        self._running = False
        self._drain_thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """Start the background drain thread."""
        if self._running:
            return
        self._running = True
        self._drain_thread = threading.Thread(
            target=self._drain, name="a2a-queue-drain", daemon=True
        )
        self._drain_thread.start()
        logger.info("AgentWorkerPool started.")

    def submit(self, task: Callable[[], Any]) -> None:
        """Enqueue a task or raise QueueFullError if at capacity."""
        if not self._running:
            raise A2AError("Pool has not been started.")
        self._queue.submit(task)

    def shutdown(self, wait: bool = True) -> None:
        """Signal drain loop to stop and optionally wait for completion."""
        self._running = False
        # Unblock the drain thread with a sentinel.
        try:
            self._queue.submit(lambda: None)
        except QueueFullError:
            pass
        if wait and self._drain_thread:
            self._drain_thread.join()
        self._executor.shutdown(wait=wait)
        logger.info("AgentWorkerPool shut down.")

    def _drain(self) -> None:
        """Continuously pull tasks from the queue and dispatch to the pool."""
        while self._running:
            task = self._queue.get()
            try:
                self._executor.submit(task)
            except Exception:
                logger.exception("Failed to submit task to executor.")
            finally:
                self._queue.task_done()


# ---------------------------------------------------------------------------
# High-Level Agent Node
# ---------------------------------------------------------------------------

class AgentNode:
    """
    Convenience façade that wires together Registry, A2AClient, and
    AgentWorkerPool into a single object representing one agent in the mesh.

    Example:
        node = AgentNode("planner", "http://localhost:8000", registry)
        node.start()
        node.send_async("executor", {"action": "run", "task": "summarise"})
        node.shutdown()
    """

    def __init__(
        self,
        agent_id: str,
        endpoint: str,
        registry: Registry,
        workers: int = 4,
        queue_size: int = 256,
        timeout: float = 5.0,
        ttl: float = 300.0,
        retry_policy: Optional[RetryPolicy] = None,
    ) -> None:
        self.agent_id = agent_id
        self._registry = registry
        self._client = A2AClient(registry, timeout=timeout, retry_policy=retry_policy)
        self._pool = AgentWorkerPool(workers=workers, queue_size=queue_size)
        registry.register(agent_id, endpoint, ttl=ttl)

    def start(self) -> None:
        self._pool.start()

    def shutdown(self, wait: bool = True) -> None:
        self._pool.shutdown(wait=wait)
        self._registry.deregister(self.agent_id)

    def send(
        self,
        target_id: str,
        payload: dict[str, Any],
        path: str = "/message",
    ) -> dict[str, Any]:
        """Synchronous send; blocks until response or exception."""
        return self._client.send(target_id, payload, path)

    def send_async(
        self,
        target_id: str,
        payload: dict[str, Any],
        path: str = "/message",
        on_result: Optional[Callable[[dict[str, Any]], None]] = None,
        on_error: Optional[Callable[[Exception], None]] = None,
    ) -> None:
        """
        Fire-and-forget send.  Optional callbacks receive the result or
        exception on the worker thread — keep them short and non-blocking.
        """
        def _task() -> None:
            try:
                result = self._client.send(target_id, payload, path)
                if on_result:
                    on_result(result)
            except A2AError as exc:
                logger.error("send_async to '%s' failed: %s", target_id, exc)
                if on_error:
                    on_error(exc)

        self._pool.submit(_task)

    def __repr__(self) -> str:  # pragma: no cover
        return f"AgentNode(id={self.agent_id!r})"
