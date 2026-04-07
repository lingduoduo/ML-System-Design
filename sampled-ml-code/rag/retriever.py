from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from feature_pipeline import SimpleEmbedder, VectorDB


@dataclass
class RetrievalClient:
    embedder: SimpleEmbedder
    vector_db: VectorDB

    def retrieve(self, query: str, top_k: int = 3) -> List[Tuple[float, Dict]]:
        query_vector = self.embedder.embed(query)
        return self.vector_db.search(query_vector, top_k=top_k)
