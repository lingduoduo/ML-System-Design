from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, Iterable, List
import uuid


@dataclass
class NoSQLDB:
    docs: Dict[str, Dict] = field(default_factory=dict)

    def insert_many(self, records: Iterable[Dict]) -> None:
        for record in records:
            self.docs[record["doc_id"]] = record

    def find_all(self) -> List[Dict]:
        return list(self.docs.values())


def fetch_medium_articles() -> List[Dict]:
    return [
        {
            "source": "medium",
            "title": "LLM Ops Basics",
            "content": "Intro to LLMOps and production systems.",
        },
        {
            "source": "medium",
            "title": "RAG Design",
            "content": "How retrieval augmented generation works.",
        },
    ]


def fetch_github_readmes() -> List[Dict]:
    return [
        {
            "source": "github",
            "title": "awesome-llm",
            "content": "Repository about LLM tools and frameworks.",
        },
        {
            "source": "github",
            "title": "rag-starter",
            "content": "Example RAG pipeline with embeddings and vector DB.",
        },
    ]


def clean_text(text: str) -> str:
    return " ".join(text.strip().split())


def etl_pipeline(raw_docs: Iterable[Dict]) -> List[Dict]:
    created_at = datetime.now(timezone.utc).isoformat()
    processed = []
    for doc in raw_docs:
        processed.append(
            {
                "doc_id": str(uuid.uuid4()),
                "source": doc["source"],
                "title": clean_text(doc["title"]),
                "content": clean_text(doc["content"]),
                "created_at": created_at,
            }
        )
    return processed


def build_document_store() -> NoSQLDB:
    db = NoSQLDB()
    raw_docs = [*fetch_medium_articles(), *fetch_github_readmes()]
    db.insert_many(etl_pipeline(raw_docs))
    return db
