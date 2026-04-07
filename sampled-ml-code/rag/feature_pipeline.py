from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Tuple

import numpy as np


def chunk_text(text: str, chunk_size: int = 50) -> List[str]:
    words = text.split()
    return [" ".join(words[index:index + chunk_size]) for index in range(0, len(words), chunk_size)]


@dataclass
class SimpleEmbedder:
    dimension: int = 8
    _cache: Dict[str, np.ndarray] = field(default_factory=dict)

    def embed(self, text: str) -> np.ndarray:
        cached = self._cache.get(text)
        if cached is not None:
            return cached

        rng = np.random.default_rng(abs(hash(text)) % (2**32))
        embedding = rng.random(self.dimension)
        self._cache[text] = embedding
        return embedding


@dataclass
class VectorDB:
    rows: List[Dict] = field(default_factory=list)
    _matrix: np.ndarray | None = None

    def add(self, vector: np.ndarray, metadata: Dict) -> None:
        self.rows.append({"vector": vector, "metadata": metadata})
        self._matrix = None

    def _ensure_matrix(self) -> np.ndarray:
        if self._matrix is None:
            if not self.rows:
                self._matrix = np.empty((0, 0))
            else:
                self._matrix = np.vstack([row["vector"] for row in self.rows])
        return self._matrix

    def search(self, query_vector: np.ndarray, top_k: int = 3) -> List[Tuple[float, Dict]]:
        if not self.rows:
            return []

        matrix = self._ensure_matrix()
        scores = matrix @ query_vector
        limit = min(top_k, len(self.rows))

        if limit == len(self.rows):
            top_indices = np.argsort(scores)[::-1]
        else:
            partition = np.argpartition(scores, -limit)[-limit:]
            top_indices = partition[np.argsort(scores[partition])[::-1]]

        return [(float(scores[index]), self.rows[index]["metadata"]) for index in top_indices]


def build_feature_store(
    docs: Iterable[Dict],
    embedder: SimpleEmbedder | None = None,
    chunk_size: int = 10,
) -> tuple[SimpleEmbedder, VectorDB, List[Dict]]:
    embedder = embedder or SimpleEmbedder()
    vector_db = VectorDB()
    instruct_dataset: List[Dict] = []

    for doc in docs:
        for chunk_id, chunk in enumerate(chunk_text(doc["content"], chunk_size=chunk_size)):
            metadata = {
                "doc_id": doc["doc_id"],
                "title": doc["title"],
                "chunk_id": chunk_id,
                "text": chunk,
            }
            vector_db.add(embedder.embed(chunk), metadata)
            instruct_dataset.append(
                {
                    "instruction": f"Summarize this chunk from {doc['title']}",
                    "input": chunk,
                    "output": f"Summary placeholder for: {chunk}",
                }
            )

    return embedder, vector_db, instruct_dataset
