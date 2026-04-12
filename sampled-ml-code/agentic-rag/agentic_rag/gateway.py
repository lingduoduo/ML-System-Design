from __future__ import annotations

from collections import defaultdict
from typing import Dict

from .schema import UserRequest


class Gateway:
    def __init__(self, rate_limit_per_user: int = 100):
        self.rate_limit_per_user = rate_limit_per_user
        self.user_request_counts: Dict[str, int] = defaultdict(int)

    def check(self, request: UserRequest) -> None:
        request_count = self.user_request_counts[request.user_id] + 1
        self.user_request_counts[request.user_id] = request_count

        if request_count > self.rate_limit_per_user:
            raise RuntimeError("Rate limit exceeded")

        if not request.message.strip():
            raise ValueError("Empty message")

    def dedupe_key(self, request: UserRequest) -> str:
        return f"{request.user_id}:{request.channel}:{request.message.strip().lower()}"
