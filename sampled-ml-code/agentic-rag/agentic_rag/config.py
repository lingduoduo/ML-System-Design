from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()


from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Performance settings
CACHE_SIZE = 1000
RESPONSE_CACHE_SIZE = 256


def _env_flag(name: str, default: bool = False) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}

# Advanced optimization settings
import numpy as np
NUMPY_AVAILABLE = True


import torch
TORCH_INSTALLED = True
TORCH_AVAILABLE = torch.cuda.is_available()


ENABLE_GUMBEL_TOOL_SELECTION = _env_flag("ENABLE_GUMBEL_TOOL_SELECTION", default=False)
GUMBEL_TOOL_TEMPERATURE = float(os.getenv("GUMBEL_TOOL_TEMPERATURE", "0.5"))
GUMBEL_TOOL_HARD = _env_flag("GUMBEL_TOOL_HARD", default=True)
TOOL_SELECTOR_MODEL_PATH = os.getenv("TOOL_SELECTOR_MODEL_PATH", "").strip() or None

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

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
API_BASE = os.getenv("OPENAI_BASE_URL", "").strip() or None


def sanity_check_openai_compatible(candidate_llm: Any, candidate_embeddings: Any) -> None:
    llm_result = candidate_llm.invoke("Reply with exactly: OK")
    print("[LangChain] LLM OK:", getattr(llm_result, "content", llm_result))

    embedding_vector = candidate_embeddings.embed_query("hello")
    print(f"[LangChain] Embeddings OK. dim = {len(embedding_vector)}")


@lru_cache(maxsize=1)
def get_llm() -> Any:
    if not OPENAI_API_KEY:
        return None

    llm_kwargs = {
        "api_key": OPENAI_API_KEY,
        "model": "gpt-3.5-turbo",
        "temperature": 0,
    }
    if API_BASE:
        llm_kwargs["base_url"] = API_BASE
    return ChatOpenAI(**llm_kwargs)


@lru_cache(maxsize=1)
def get_embeddings() -> Any:
    if not OPENAI_API_KEY:
        return None

    emb_kwargs = {
        "api_key": OPENAI_API_KEY,
        "model": "text-embedding-3-small",
    }
    if API_BASE:
        emb_kwargs["base_url"] = API_BASE
    return OpenAIEmbeddings(**emb_kwargs)


@lru_cache(maxsize=1)
def get_openai_clients() -> tuple[Any, Any]:
    llm = get_llm()
    embeddings = get_embeddings()
    if llm is None or embeddings is None:
        return None, None

    try:
        sanity_check_openai_compatible(llm, embeddings)
    except Exception as exc:
        print(f"[LangChain] Setup failed: {exc}")
        return None, None
    return llm, embeddings

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

# Numba availability - set to False to avoid import issues on some systems
NUMBA_AVAILABLE = False
