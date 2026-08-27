"""Response models for the poll-based task API."""

from pydantic import BaseModel, Field

from dcc_gliner_api.services.task_store import TaskStatus


class TaskAccepted(BaseModel):
    """Handed back the moment work is accepted, before it has run."""

    task_id: str = Field(description="Poll `GET /task/{task_id}` with this")


class TaskState(BaseModel):
    """Where a submitted job stands, cheap enough to poll in a loop."""

    task_id: str
    status: TaskStatus
    progress: float | None = Field(
        default=None,
        description="Fraction of the work done, in [0, 1]; null while unknown",
    )
    resource_id: str | None = Field(
        default=None,
        description="Set once finished: fetch `GET /resource/{resource_id}` exactly once",
    )
    error: str | None = Field(default=None, description="Set when the status is failed")
