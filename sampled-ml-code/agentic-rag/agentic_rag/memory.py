from __future__ import annotations

import asyncio
import hashlib
import threading
from collections import defaultdict, deque
from datetime import datetime
from typing import Any, Dict, List

import numpy as np

try:
    import faiss

    FAISS_AVAILABLE = True
except ImportError:
    faiss = None
    FAISS_AVAILABLE = False


EMBEDDING_DIM = 128


class MemoryVector:
    """Vectorized memory entry used by hierarchical memory storage."""

    def __init__(
        self,
        content: str,
        role: str = "system",
        embedding: np.ndarray | None = None,
        importance: float = 0.5,
        metadata: Dict[str, Any] | None = None,
    ):
        self.content = content
        self.role = role
        self.importance = importance
        self.metadata = metadata or {}
        self.timestamp = datetime.now()
        self.access_count = 0
        self.last_accessed_at = self.timestamp.timestamp()
        self.embedding = embedding if embedding is not None else self._generate_embedding()

    def update_access(self) -> None:
        self.access_count += 1
        self.last_accessed_at = datetime.now().timestamp()

    def _generate_embedding(self) -> np.ndarray:
        """Generate a deterministic embedding-style vector from the content."""
        lowered = self.content.lower()
        seed_bytes = hashlib.blake2b(lowered.encode("utf-8"), digest_size=8).digest()
        seed = int.from_bytes(seed_bytes, "little") % (2**32)
        rng = np.random.RandomState(seed)
        vector = rng.randn(EMBEDDING_DIM).astype("float32")

        keywords = ["exclusivity", "clause", "breach", "important", "contract"]
        bucket = max(1, EMBEDDING_DIM // len(keywords))
        for idx, keyword in enumerate(keywords):
            if keyword in lowered:
                start = idx * bucket
                end = min(EMBEDDING_DIM, start + bucket)
                vector[start:end] += 0.5

        norm = np.linalg.norm(vector)
        return vector / norm if norm > 0 else vector


class MemoryRetriever:
    """FAISS-based memory retriever."""

    def __init__(self, embedding_dim: int = EMBEDDING_DIM):
        self.embedding_dim = embedding_dim

    def retrieve(self, query: str, storage, top_k: int = 5) -> List[MemoryVector]:
        """Retrieve relevant memories."""
        query_vector = self._encode_query(query)
        return storage.search(query_vector, top_k)

    def _encode_query(self, query: str) -> np.ndarray:
        """Encode the query."""
        lowered = query.lower()
        seed_bytes = hashlib.blake2b(lowered.encode("utf-8"), digest_size=8).digest()
        seed = int.from_bytes(seed_bytes, "little") % (2**32)
        rng = np.random.RandomState(seed)
        vector = rng.randn(self.embedding_dim).astype("float32")

        keywords = ["exclusivity", "clause", "breach", "important", "contract"]
        bucket = max(1, self.embedding_dim // len(keywords))
        for idx, keyword in enumerate(keywords):
            if keyword in lowered:
                start = idx * bucket
                end = min(self.embedding_dim, start + bucket)
                vector[start:end] += 0.5

        norm = np.linalg.norm(vector)
        return vector / norm if norm > 0 else vector


class MemoryStorage:
    """Hierarchical short-term and long-term memory with optional FAISS search."""

    def __init__(self, short_limit: int = 15, long_limit: int = 100, embedding_dim: int = EMBEDDING_DIM):
        self.short_memories: List[MemoryVector] = []
        self.long_memories: List[MemoryVector] = []
        self.short_limit = short_limit
        self.long_limit = long_limit
        self.embedding_dim = embedding_dim

        self.short_index = self._build_index()
        self.long_index = self._build_index()

    def add_memory(self, memory: MemoryVector) -> None:
        if memory.importance >= 0.6:
            self._add_long_term(memory)
        else:
            self._add_short_term(memory)

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> List[MemoryVector]:
        if top_k <= 0:
            return []

        normalized_query = self._normalize(query_vector)
        results: List[tuple[MemoryVector, float]] = []
        results.extend(self._search_partition(normalized_query, self.short_memories, self.short_index, top_k))
        results.extend(self._search_partition(normalized_query, self.long_memories, self.long_index, top_k))

        deduped: Dict[int, tuple[MemoryVector, float]] = {}
        for memory, score in results:
            key = id(memory)
            previous = deduped.get(key)
            if previous is None or score > previous[1]:
                deduped[key] = (memory, score)

        ranked = sorted(deduped.values(), key=lambda item: item[1], reverse=True)
        memories = [memory for memory, _ in ranked[:top_k]]
        for memory in memories:
            memory.update_access()
        return memories

    def get_all_memories(self) -> List[MemoryVector]:
        return list(self.short_memories) + list(self.long_memories)

    def get_status(self) -> Dict[str, int]:
        return {
            "short_term": len(self.short_memories),
            "long_term": len(self.long_memories),
            "total": len(self.short_memories) + len(self.long_memories),
        }

    def _build_index(self):
        if FAISS_AVAILABLE:
            return faiss.IndexFlatIP(self.embedding_dim)
        return None

    def _add_short_term(self, memory: MemoryVector) -> None:
        """Add short-term memory and rebuild if the partition exceeds its limit."""
        self.short_memories.append(memory)
        if self.short_index is not None:
            self.short_index.add(memory.embedding.reshape(1, -1))

        if len(self.short_memories) > self.short_limit:
            important = [mem for mem in self.short_memories if mem.importance > 0.5]
            for mem in important[: len(important) // 2]:
                self._add_long_term(mem)

            self.short_memories = self.short_memories[-max(1, self.short_limit // 2) :]
            self._rebuild_short_index()

    def _add_long_term(self, memory: MemoryVector) -> None:
        """Add long-term memory if an equivalent entry is not already stored."""
        if any(existing.content == memory.content and existing.role == memory.role for existing in self.long_memories):
            return

        self.long_memories.append(memory)
        if self.long_index is not None:
            self.long_index.add(memory.embedding.reshape(1, -1))

        if len(self.long_memories) > self.long_limit:
            self.long_memories = self.long_memories[-self.long_limit :]
            self._rebuild_long_index()

    def _rebuild_short_index(self) -> None:
        """Rebuild the short-term FAISS index from current memories."""
        self.short_index = self._build_index()
        if self.short_index is not None and self.short_memories:
            self.short_index.add(np.vstack([memory.embedding for memory in self.short_memories]))

    def _rebuild_long_index(self) -> None:
        """Rebuild the long-term FAISS index from current memories."""
        self.long_index = self._build_index()
        if self.long_index is not None and self.long_memories:
            self.long_index.add(np.vstack([memory.embedding for memory in self.long_memories]))

    def _search_partition(self, query_vector: np.ndarray, memories: List[MemoryVector], index, top_k: int) -> List[tuple[MemoryVector, float]]:
        if not memories:
            return []

        limit = min(top_k, len(memories))
        if index is not None:
            scores, indices = index.search(query_vector.reshape(1, -1), limit)
            return [
                (memories[idx], float(score))
                for score, idx in zip(scores[0], indices[0])
                if idx >= 0
            ]

        return sorted(
            ((memory, float(np.dot(query_vector, memory.embedding))) for memory in memories),
            key=lambda item: item[1],
            reverse=True,
        )[:limit]


class MemoryStore:
    def __init__(self, short_term_limit: int = 10, vector_short_limit: int = 15, vector_long_limit: int = 100):
        self.short_term: Dict[str, deque] = defaultdict(lambda: deque(maxlen=short_term_limit))
        self.long_term: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self.vector_memory: Dict[str, MemoryStorage] = defaultdict(
            lambda: MemoryStorage(short_limit=vector_short_limit, long_limit=vector_long_limit)
        )
        self.retriever = MemoryRetriever(embedding_dim=EMBEDDING_DIM)
        self._lock = threading.RLock()

    def load_short_term(self, user_id: str) -> List[Dict[str, str]]:
        with self._lock:
            return list(self.short_term[user_id])

    def load_long_term(self, user_id: str) -> Dict[str, Any]:
        with self._lock:
            memory = dict(self.long_term[user_id])
            memory.setdefault("memory_status", self.vector_memory[user_id].get_status())
            return memory

    def save_turn(self, user_id: str, user_message: str, assistant_message: str) -> None:
        with self._lock:
            self.short_term[user_id].append({"role": "user", "content": user_message})
            self.short_term[user_id].append({"role": "assistant", "content": assistant_message})

            self.vector_memory[user_id].add_memory(self._build_memory_vector(user_message, "user"))
            self.vector_memory[user_id].add_memory(self._build_memory_vector(assistant_message, "assistant"))

    def update_long_term(self, user_id: str, updates: Dict[str, Any]) -> None:
        with self._lock:
            self.long_term[user_id].update(updates)

    def search_memories(self, user_id: str, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        with self._lock:
            matches = self.retriever.retrieve(query, self.vector_memory[user_id], top_k=top_k)
            return [
                {
                    "role": memory.role,
                    "content": memory.content,
                    "importance": memory.importance,
                    "access_count": memory.access_count,
                    **memory.metadata,
                }
                for memory in matches
            ]

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

    async def search_memories_async(self, user_id: str, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.search_memories, user_id, query, top_k)

    def _build_memory_vector(self, content: str, role: str) -> MemoryVector:
        importance = self._estimate_importance(content, role)
        return MemoryVector(
            content=content,
            role=role,
            embedding=self._embed_text(content),
            importance=importance,
            metadata={"length": len(content)},
        )

    @staticmethod
    def _estimate_importance(content: str, role: str) -> float:
        base = 0.45 if role == "assistant" else 0.35
        if len(content) > 160:
            base += 0.15
        if any(token in content.lower() for token in ("refund", "order", "policy", "ticket", "urgent")):
            base += 0.2
        return min(base, 1.0)

    @staticmethod
    def _embed_text(text: str, dim: int = EMBEDDING_DIM) -> np.ndarray:
        embedding = np.zeros(dim, dtype=np.float32)
        for token in text.lower().split():
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(digest, "little") % dim
            embedding[index] += 1.0

        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding /= norm
        return embedding


class MemoryTransformer:
    """FAISS-optimized memory transformer built on the local memory stack."""

    def __init__(self, short_limit: int = 15, long_limit: int = 100):
        self.storage = MemoryStorage(short_limit=short_limit, long_limit=long_limit)
        self.retriever = MemoryRetriever(embedding_dim=EMBEDDING_DIM)

    def process(self, content: str) -> Dict[str, Any]:
        """Compute importance, store the memory, and return current storage status."""
        memory = MemoryVector(
            content=content,
            role="system",
            embedding=MemoryStore._embed_text(content),
            importance=self._calculate_importance(content),
            metadata={"length": len(content)},
        )
        self.storage.add_memory(memory)
        return {
            "memory_status": self.storage.get_status(),
            "content": content,
        }

    def query(self, query: str, top_k: int = 5) -> List[MemoryVector]:
        """Retrieve relevant memories from hierarchical storage."""
        return self.retriever.retrieve(query, self.storage, top_k)

    @staticmethod
    def _calculate_importance(content: str) -> float:
        """Estimate whether content should remain short-term or be promoted."""
        lowered = content.lower()
        score = 0.3
        keywords = ["exclusivity", "liquidated damages", "important", "termination", "obligation"]
        for keyword in keywords:
            if keyword in lowered:
                score += 0.15
        if len(content) > 30:
            score += 0.1
        return min(1.0, score)
