"""In-process store for long-running jobs, addressed by task id.

Proxies and WAFs cut long-held connections, so callers submit work, get a task
id back immediately, poll a cheap status endpoint, then fetch the result once.
The result is dropped as it is read, and abandoned entries expire, so the store
does not grow.

Every method runs on the event loop (the endpoints are async), so the plain
dicts below need no locking. State lives in the worker process, so a caller has
to keep polling the instance that accepted the submission. That is the same contract docling's async API
has; behind a load balancer it needs a single replica or sticky sessions.
"""

import asyncio
import logging
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Literal

logger = logging.getLogger(__name__)

TaskStatus = Literal["pending", "running", "finished", "failed"]

#: How long a finished result is kept for collection before it is discarded.
DEFAULT_TTL_SECONDS = 3600.0


@dataclass
class Task:
    """One submitted job and, once it finishes, where its result waits."""

    id: str
    status: TaskStatus = "pending"
    #: Fraction done in [0, 1], or None while the job cannot report it.
    progress: float | None = None
    resource_id: str | None = None
    error: str | None = None
    updated_at: float = field(default_factory=time.monotonic)

    def touch(self) -> None:
        self.updated_at = time.monotonic()


class TaskStore:
    """Runs coroutines in the background and hands out their results once."""

    def __init__(self, ttl_seconds: float = DEFAULT_TTL_SECONDS) -> None:
        self._ttl_seconds = ttl_seconds
        self._tasks: dict[str, Task] = {}
        self._resources: dict[str, Any] = {}
        self._running: dict[str, asyncio.Task[None]] = {}

    def submit(self, work: Callable[[Task], Awaitable[Any]]) -> Task:
        """Start ``work`` in the background and return its pending task.

        Args:
            work: Coroutine function receiving its own task, so it can report
                progress while it runs. Its return value becomes the resource.

        Returns:
            The task, already registered and pollable.
        """
        self._expire()

        task = Task(id=uuid.uuid4().hex)
        self._tasks[task.id] = task
        self._running[task.id] = asyncio.create_task(self._run(task, work))
        return task

    async def _run(self, task: Task, work: Callable[[Task], Awaitable[Any]]) -> None:
        task.status = "running"
        task.touch()
        try:
            result = await work(task)
        except Exception as error:
            logger.exception("Task %s failed", task.id)
            task.status = "failed"
            task.error = str(error)
        else:
            resource_id = uuid.uuid4().hex
            self._resources[resource_id] = result
            task.resource_id = resource_id
            task.progress = 1.0
            task.status = "finished"
        finally:
            task.touch()
            _ = self._running.pop(task.id, None)

    def get(self, task_id: str) -> Task | None:
        self._expire()
        return self._tasks.get(task_id)

    def take_resource(self, resource_id: str) -> tuple[bool, Any]:
        """Hand out a result and forget it, so nothing is kept after collection.

        Returns:
            ``(True, result)`` when the resource existed, ``(False, None)``
            otherwise — the result itself may legitimately be ``None``.
        """
        self._expire()

        if resource_id not in self._resources:
            return False, None

        result = self._resources.pop(resource_id)
        for task in self._tasks.values():
            if task.resource_id == resource_id:
                del self._tasks[task.id]
                break

        return True, result

    def _expire(self) -> None:
        """Drop tasks nobody collected, and any result still attached to them."""
        cutoff = time.monotonic() - self._ttl_seconds
        stale = [task for task in self._tasks.values() if task.updated_at < cutoff and task.id not in self._running]

        for task in stale:
            if task.resource_id:
                _ = self._resources.pop(task.resource_id, None)
            del self._tasks[task.id]

        if stale:
            logger.debug("Expired %d abandoned tasks", len(stale))
