"""The tool output contract (ARCHITECTURE §4). Validation refuses look-ahead at construction time."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .time import LookAheadError, utc


class ToolResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    tool: str
    version: str
    asset: str
    as_of: datetime
    data_max_ts: datetime
    value: dict[str, Any]
    uncertainty: dict[str, Any] = Field(default_factory=dict)
    quality: dict[str, Any] = Field(default_factory=dict)
    summary: str = ""

    @model_validator(mode="after")
    def _no_lookahead(self) -> ToolResult:
        if utc(self.data_max_ts) > utc(self.as_of):
            raise LookAheadError(f"{self.tool}: data_max_ts {self.data_max_ts} > as_of {self.as_of}")
        return self
