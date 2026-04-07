from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass
class Monitor:
    history: List[Dict] = field(default_factory=list)

    def log_request(self, query: str, retrieved: List[Tuple[float, Dict]], response: str) -> Dict:
        event = {
            "query": query,
            "retrieved_chunks": [metadata["text"] for _, metadata in retrieved],
            "response_preview": response[:200],
        }
        self.history.append(event)
        return event
