from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple


def build_prompt(query: str, retrieved_chunks: List[Tuple[float, Dict]]) -> str:
    context = "\n".join(metadata["text"] for _, metadata in retrieved_chunks)
    return (
        "You are a helpful assistant.\n\n"
        f"Question:\n{query}\n\n"
        f"Context:\n{context}\n\n"
        "Answer:\n"
    )


@dataclass
class LLMTwin:
    model: Any

    def answer(self, query: str, retrieved_chunks: List[Tuple[float, Dict]]) -> str:
        prompt = build_prompt(query, retrieved_chunks)
        if hasattr(self.model, "generate"):
            return self.model.generate(prompt)
        if hasattr(self.model, "invoke"):
            return self.model.invoke(prompt)
        raise TypeError("LLMTwin model must implement either `generate` or `invoke`.")
