from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any

from Inference_pipeline import LLMTwin
from data_collection_pipeline import NoSQLDB, build_document_store
from deploy import Deployer
from feature_pipeline import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DOCUMENT_PATH,
    FILE_PATTERN,
    TOP_K,
    LangChainFeatureStore,
    SimpleEmbedder,
    VectorDB,
    build_embeddings,
    build_feature_store,
    build_langchain_feature_store,
    build_local_hf_llm,
)
from monitoring import Monitor
from retriever import LangChainRetrievalClient, RetrievalClient
from training_pipeline import (
    ExperimentTracker,
    FineTunedLLM,
    ModelRegistry,
    train_and_register_model,
)


@dataclass
class RAGSystem:
    db: NoSQLDB
    embedder: SimpleEmbedder | Any
    vector_db: VectorDB | Any
    instruct_dataset: list[dict]
    retrieval_client: RetrievalClient | LangChainRetrievalClient
    llm_twin: LLMTwin
    monitor: Monitor
    tracker: ExperimentTracker
    registry: ModelRegistry
    deployment_info: dict
    langchain_feature_store: LangChainFeatureStore | None = None


def build_rag_system() -> RAGSystem:
    db = build_document_store()
    monitor = Monitor()
    embedder, vector_db, instruct_dataset = build_feature_store(db.find_all())
    tracker, registry, accepted_model, _ = train_and_register_model(instruct_dataset)

    try:
        logging.info("Loading documents from %s (pattern: %s)...", DOCUMENT_PATH, FILE_PATTERN)
        langchain_feature_store = build_langchain_feature_store(
            document_path=DOCUMENT_PATH,
            file_pattern=FILE_PATTERN,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            top_k=TOP_K,
        )
        logging.info("Loaded %s documents", len(langchain_feature_store.documents))
        logging.info("Created %s chunks", len(langchain_feature_store.chunks))
        embeddings = build_embeddings()
        logging.info("Building FAISS vector store...")
        vectorstore = langchain_feature_store.vector_store
        retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})
        llm = build_local_hf_llm()

        retrieval_client: RetrievalClient | LangChainRetrievalClient = LangChainRetrievalClient(
            retriever=retriever
        )
        llm_twin = LLMTwin(model=llm)
        deployment_info = Deployer().deploy(FineTunedLLM(model_name="local-hf-llm"))

        return RAGSystem(
            db=db,
            embedder=embeddings,
            vector_db=vectorstore,
            instruct_dataset=instruct_dataset,
            retrieval_client=retrieval_client,
            llm_twin=llm_twin,
            monitor=monitor,
            tracker=tracker,
            registry=registry,
            deployment_info=deployment_info,
            langchain_feature_store=langchain_feature_store,
        )
    except (ImportError, FileNotFoundError, ValueError) as exc:
        logging.warning("Falling back to toy in-memory RAG pipeline: %s", exc)

    if accepted_model is None:
        accepted_model = FineTunedLLM(model_name="fallback-llm")

    retrieval_client = RetrievalClient(embedder=embedder, vector_db=vector_db)
    llm_twin = LLMTwin(model=accepted_model)
    deployment_info = Deployer().deploy(accepted_model)

    return RAGSystem(
        db=db,
        embedder=embedder,
        vector_db=vector_db,
        instruct_dataset=instruct_dataset,
        retrieval_client=retrieval_client,
        llm_twin=llm_twin,
        monitor=monitor,
        tracker=tracker,
        registry=registry,
        deployment_info=deployment_info,
        langchain_feature_store=None,
    )
