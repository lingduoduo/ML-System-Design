from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from training_pipeline import FineTunedLLM


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
    model: FineTunedLLM

    def answer(self, query: str, retrieved_chunks: List[Tuple[float, Dict]]) -> str:
        return self.model.generate(build_prompt(query, retrieved_chunks))
