from __future__ import annotations

import asyncio
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, List, Sequence

from .config import (
    CACHE_SIZE,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DOCUMENT_FILES,
    DOCUMENT_LABELS,
    LANGCHAIN_AVAILABLE,
    MAP_PROMPT,
    REDUCE_PROMPT,
    embeddings,
    llm,
)
from .schema import RetrievedDocument

try:
    from langchain_community.document_loaders import TextLoader
    from langchain_community.vectorstores import FAISS
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    TextLoader = None
    FAISS = None
    RecursiveCharacterTextSplitter = None

# Pre-compile regex patterns for performance
PARAGRAPH_SPLIT = re.compile(r"\n\s*\n")
WORD_TOKENIZE = re.compile(r"\b[a-z0-9]+\b")
ORDER_ID_PATTERN = re.compile(r"\bORD-\d+\b", re.IGNORECASE)


def chunk_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> List[str]:
    cleaned = text.strip()
    if not cleaned:
        return []

    paragraphs = [paragraph.strip() for paragraph in PARAGRAPH_SPLIT.split(cleaned) if paragraph.strip()]
    if not paragraphs:
        return [cleaned]

    chunks: List[str] = []
    current_parts: List[str] = []
    current_length = 0

    for paragraph in paragraphs:
        paragraph_length = len(paragraph)
        separator_length = 2 if current_parts else 0

        if current_parts and current_length + separator_length + paragraph_length > chunk_size:
            chunks.append("\n\n".join(current_parts))

            overlap_parts: List[str] = []
            overlap_length = 0
            for existing in reversed(current_parts):
                projected = len(existing) + (2 if overlap_parts else 0)
                if overlap_length + projected > chunk_overlap:
                    break
                overlap_parts.insert(0, existing)
                overlap_length += projected

            current_parts = overlap_parts
            current_length = sum(len(part) for part in current_parts) + max(0, len(current_parts) - 1) * 2

        if paragraph_length > chunk_size:
            if current_parts:
                chunks.append("\n\n".join(current_parts))
                current_parts = []
                current_length = 0

            # Optimized chunking for long paragraphs
            for start in range(0, paragraph_length, chunk_size - chunk_overlap):
                end = min(start + chunk_size, paragraph_length)
                piece = paragraph[start:end].strip()
                if piece:
                    chunks.append(piece)
                if end >= paragraph_length:
                    break
            continue

        current_parts.append(paragraph)
        current_length += separator_length + paragraph_length

    if current_parts:
        chunks.append("\n\n".join(current_parts))

    return chunks


def tokenize(text: str) -> set[str]:
    return {token for token in WORD_TOKENIZE.findall(text.lower()) if len(token) > 1}


@lru_cache(maxsize=CACHE_SIZE)
def query_tokens_for(text: str) -> frozenset[str]:
    return frozenset(tokenize(text))


def lexical_score_from_tokens(query_tokens: Sequence[str] | set[str] | frozenset[str], text_tokens: set[str]) -> float:
    if not query_tokens:
        return 0.0

    overlap = len(set(query_tokens) & text_tokens)
    return overlap / len(query_tokens)


def lexical_score(query: str, text: str) -> float:
    query_tokens = tokenize(query)
    if not query_tokens:
        return 0.0

    text_tokens = tokenize(text)
    return lexical_score_from_tokens(query_tokens, text_tokens)


@lru_cache(maxsize=None)
def load_document_text(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def fallback_summary(text: str, max_lines: int = 8) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return "\n".join(lines[:max_lines]) if lines else "No content available."


def load_and_split_txt(
    file_path: Path,
    doc_name: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> List[Any]:
    if not LANGCHAIN_AVAILABLE or TextLoader is None or RecursiveCharacterTextSplitter is None:
        return []

    loader = TextLoader(str(file_path), encoding="utf-8")
    docs = loader.load()
    for doc in docs:
        doc.metadata = {**(doc.metadata or {}), "source": str(file_path), "doc_name": doc_name}

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_documents(docs)


def summarize_document(corpus_name: str, title: str) -> str:
    file_path = DOCUMENT_FILES.get(corpus_name)
    if not file_path or not file_path.exists():
        return f"No {title} available for summarization."

    if LANGCHAIN_AVAILABLE and llm and MAP_PROMPT and REDUCE_PROMPT:
        try:
            chunks = load_and_split_txt(file_path, doc_name=title)
            chunk_summaries = [
                llm.invoke(MAP_PROMPT.format(doc_name=title, chunk=chunk.page_content)).content
                for chunk in chunks
            ]
            combined = "\n".join(f"- {summary}" for summary in chunk_summaries)
            final_summary = llm.invoke(
                REDUCE_PROMPT.format(doc_name=title, summaries=combined)
            ).content
            return f"{title.title()} Summary:\n{final_summary}"
        except Exception as exc:
            return f"Summarization failed: {exc}"

    try:
        content = load_document_text(str(file_path))
        return f"{title.title()} Summary:\n{fallback_summary(content)}"
    except Exception as exc:
        return f"Failed to read {title}: {exc}"


class RetrievalModel:
    def __init__(
        self,
        corpus_name: str,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    ):
        self.corpus_name = corpus_name
        self.vector_store = None
        self.documents: List[RetrievedDocument] = []
        self.document_tokens: List[set[str]] = []
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.raw_text = ""
        self._load_documents()

    def _load_documents(self) -> None:
        file_path = DOCUMENT_FILES.get(self.corpus_name)
        if not file_path or not file_path.exists():
            print(f"[Retrieval] No file found for {self.corpus_name}: {file_path}")
            return

        try:
            self.raw_text = load_document_text(str(file_path))
        except Exception as exc:
            print(f"[Retrieval] Failed to read {self.corpus_name}: {exc}")
            return

        if LANGCHAIN_AVAILABLE and embeddings and FAISS is not None:
            try:
                self.documents = load_and_split_txt(
                    file_path,
                    doc_name=DOCUMENT_LABELS.get(self.corpus_name, self.corpus_name),
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.chunk_overlap,
                )
                self.vector_store = FAISS.from_documents(self.documents, embeddings)
                print(f"[Retrieval] Loaded {len(self.documents)} chunks for {self.corpus_name}")
            except Exception as exc:
                print(f"[Retrieval] Failed to load {self.corpus_name}: {exc}")
        else:
            self.documents = [
                RetrievedDocument(
                    doc_id=f"{self.corpus_name}-{index}",
                    source=self.corpus_name,
                    text=chunk,
                    score=0.0,
                    metadata={"source": str(file_path)},
                )
                for index, chunk in enumerate(chunk_text(self.raw_text, self.chunk_size, self.chunk_overlap), start=1)
            ]
            self.document_tokens = [tokenize(document.text) for document in self.documents]
            print(f"[Retrieval] Loaded fallback chunks for {self.corpus_name}: {len(self.documents)}")

    def retrieve(self, query: str, top_k: int = 3) -> List[RetrievedDocument]:
        if self.vector_store and LANGCHAIN_AVAILABLE:
            try:
                docs = self.vector_store.similarity_search_with_score(query, k=top_k)
                return [
                    RetrievedDocument(
                        doc_id=f"{self.corpus_name}-{index}",
                        source=self.corpus_name,
                        text=doc.page_content,
                        score=1.0 / (1.0 + score),
                        metadata=doc.metadata,
                    )
                    for index, (doc, score) in enumerate(docs, start=1)
                ]
            except Exception as exc:
                print(f"[Retrieval] Vector search failed: {exc}")

        if not self.documents:
            return []

        query_tokens = query_tokens_for(query)
        scored_docs = [
            RetrievedDocument(
                doc_id=document.doc_id,
                source=document.source,
                text=document.text,
                score=lexical_score_from_tokens(query_tokens, doc_tokens),
                metadata=document.metadata,
            )
            for document, doc_tokens in zip(self.documents, self.document_tokens)
        ]
        ranked_docs = sorted(scored_docs, key=lambda doc: doc.score, reverse=True)
        return [doc for doc in ranked_docs[:top_k] if doc.score > 0] or ranked_docs[:1]

    def batch_retrieve(self, queries: List[str], top_k: int = 3) -> List[List[RetrievedDocument]]:
        """Batch processing for multiple queries with optimized performance."""
        if not queries:
            return []

        if self.vector_store and LANGCHAIN_AVAILABLE:
            try:
                # Batch vector search
                results = []
                for query in queries:
                    docs_with_scores = self.vector_store.similarity_search_with_score(query, k=top_k)
                    docs = [
                        RetrievedDocument(
                            doc_id=f"{self.corpus_name}-{index}",
                            source=self.corpus_name,
                            text=doc.page_content,
                            score=1.0 / (1.0 + score),
                            metadata=doc.metadata,
                        )
                        for index, (doc, score) in enumerate(docs_with_scores, start=1)
                    ]
                    results.append(docs)
                return results
            except Exception as exc:
                print(f"[Retrieval] Batch vector search failed: {exc}")

        if not self.documents:
            return [[] for _ in queries]

        results: List[List[RetrievedDocument]] = []
        for query in queries:
            query_tokens = query_tokens_for(query)
            scored_docs = [
                RetrievedDocument(
                    doc_id=document.doc_id,
                    source=document.source,
                    text=document.text,
                    score=lexical_score_from_tokens(query_tokens, doc_tokens),
                    metadata=document.metadata,
                )
                for document, doc_tokens in zip(self.documents, self.document_tokens)
            ]
            ranked_docs = sorted(scored_docs, key=lambda doc: doc.score, reverse=True)
            results.append([doc for doc in ranked_docs[:top_k] if doc.score > 0] or ranked_docs[:1])
        return results

    async def retrieve_async(self, query: str, top_k: int = 3) -> List[RetrievedDocument]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.retrieve, query, top_k)

    async def batch_retrieve_async(self, queries: List[str], top_k: int = 3) -> List[List[RetrievedDocument]]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.batch_retrieve, queries, top_k)


class Reranker:
    def rerank(self, query: str, docs: List[RetrievedDocument]) -> List[RetrievedDocument]:
        return sorted(docs, key=lambda d: d.score, reverse=True)

    def batch_rerank(self, queries: List[str], docs_list: List[List[RetrievedDocument]]) -> List[List[RetrievedDocument]]:
        return [self.rerank(query, docs) for query, docs in zip(queries, docs_list)]
