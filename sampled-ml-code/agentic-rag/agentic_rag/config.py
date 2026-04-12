from __future__ import annotations

import os
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

try:
    from langchain_core.prompts import PromptTemplate
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings

    LANGCHAIN_AVAILABLE = True
except ImportError:
    PromptTemplate = None
    ChatOpenAI = None
    OpenAIEmbeddings = None
    LANGCHAIN_AVAILABLE = False

# Performance settings
CACHE_SIZE = 1000
RESPONSE_CACHE_SIZE = 256

# Advanced optimization settings
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    np = None
    NUMPY_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = torch.cuda.is_available()
except ImportError:
    TORCH_AVAILABLE = False

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

DEFAULT_CHUNK_SIZE = 512
DEFAULT_CHUNK_OVERLAP = 80
DEFAULT_TOP_K = 3

DOCUMENT_FILES = {
    "user_docs": DATA_DIR / "user_docs.txt",
    "policy_docs": DATA_DIR / "policy_docs.txt",
}
DOCUMENT_LABELS = {
    "user_docs": "user documentation",
    "policy_docs": "refund policy",
}

openai_api_key = os.getenv("OPENAI_API_KEY")
API_BASE = os.getenv("OPENAI_BASE_URL", "").strip() or None


def sanity_check_openai_compatible(candidate_llm: Any, candidate_embeddings: Any) -> None:
    llm_result = candidate_llm.invoke("Reply with exactly: OK")
    print("[LangChain] LLM OK:", getattr(llm_result, "content", llm_result))

    embedding_vector = candidate_embeddings.embed_query("hello")
    print(f"[LangChain] Embeddings OK. dim = {len(embedding_vector)}")


llm = None
embeddings = None
if LANGCHAIN_AVAILABLE and openai_api_key:
    llm_kwargs = {
        "api_key": openai_api_key,
        "model": "gpt-3.5-turbo",
        "temperature": 0,
    }
    emb_kwargs = {
        "api_key": openai_api_key,
        "model": "text-embedding-3-small",
    }
    if API_BASE:
        llm_kwargs["base_url"] = API_BASE
        emb_kwargs["base_url"] = API_BASE

    llm = ChatOpenAI(**llm_kwargs)
    embeddings = OpenAIEmbeddings(**emb_kwargs)

    try:
        sanity_check_openai_compatible(llm, embeddings)
    except Exception as exc:
        print(f"[LangChain] Setup failed: {exc}")
        llm = None
        embeddings = None

if LANGCHAIN_AVAILABLE and PromptTemplate is not None:
    MAP_PROMPT = PromptTemplate.from_template(
        "You are summarizing a support document chunk.\n"
        "Document: {doc_name}\n"
        "Chunk:\n{chunk}\n\n"
        "Write a concise, factual summary focused on user-relevant details:"
    )
    REDUCE_PROMPT = PromptTemplate.from_template(
        "You are writing a final summary from chunk summaries.\n"
        "Document: {doc_name}\n"
        "Chunk summaries:\n{summaries}\n\n"
        "Write a coherent high-level summary grounded only in the chunk summaries:"
    )
else:
    MAP_PROMPT = None
    REDUCE_PROMPT = None

# Numba availability - set to False to avoid import issues on some systems
NUMBA_AVAILABLE = False
