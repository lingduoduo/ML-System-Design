from __future__ import annotations

import time
from typing import Any, Dict

from .schema import AgentState


class TraceLogger:
    def log(self, state: AgentState, stage: str, payload: Dict[str, Any]) -> None:
        state.trace.append(
            {
                "timestamp": time.time(),
                "stage": stage,
                "payload": payload,
            }
        )
