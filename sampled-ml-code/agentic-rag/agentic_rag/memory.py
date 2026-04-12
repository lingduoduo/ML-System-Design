from __future__ import annotations

import asyncio
import threading
from collections import defaultdict, deque
from typing import Any, Dict, List


class MemoryStore:
    def __init__(self):
        self.short_term: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10))
        self.long_term: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self._lock = threading.RLock()

    def load_short_term(self, user_id: str) -> List[Dict[str, str]]:
        with self._lock:
            return list(self.short_term[user_id])

    def load_long_term(self, user_id: str) -> Dict[str, Any]:
        with self._lock:
            return dict(self.long_term[user_id])

    def save_turn(self, user_id: str, user_message: str, assistant_message: str) -> None:
        with self._lock:
            self.short_term[user_id].append({"role": "user", "content": user_message})
            self.short_term[user_id].append({"role": "assistant", "content": assistant_message})
            # deque automatically maintains maxlen

    def update_long_term(self, user_id: str, updates: Dict[str, Any]) -> None:
        with self._lock:
            self.long_term[user_id].update(updates)

    async def load_short_term_async(self, user_id: str) -> List[Dict[str, str]]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.load_short_term, user_id)

    async def load_long_term_async(self, user_id: str) -> Dict[str, Any]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.load_long_term, user_id)

    async def save_turn_async(self, user_id: str, user_message: str, assistant_message: str) -> None:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self.save_turn, user_id, user_message, assistant_message)

    async def update_long_term_async(self, user_id: str, updates: Dict[str, Any]) -> None:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self.update_long_term, user_id, updates)
