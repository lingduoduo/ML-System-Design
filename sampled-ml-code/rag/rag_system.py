from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any

from Inference_pipeline import LLMTwin, GraphContext, build_agent_graph, build_rag_chain
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
    build_feature_store,
    build_langchain_feature_store,
    build_local_hf_llm,
    create_rewrite_chain,
)
from monitoring import EvaluationTracker
from retriever import (
    LangChainRetrievalClient,
    MultiPathRetriever,
    RetrievalClient,
    build_llm_reranker,
)
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
    evaluator: EvaluationTracker
    tracker: ExperimentTracker
    registry: ModelRegistry
    deployment_info: dict
    langchain_feature_store: LangChainFeatureStore | None = None
    rewrite_chain: Any | None = None
    rag_chain: Any | None = None
    multi_step_agent: Any | None = None


@dataclass
class RichRAGRuntime:
    embedder: Any
    vectorstore: Any
    retrieval_client: LangChainRetrievalClient
    llm_twin: LLMTwin
    rewrite_chain: Any
    rag_chain: Any
    multi_step_agent: Any
    langchain_feature_store: LangChainFeatureStore


@dataclass
class RichRAGBuilder:
    document_path: str = DOCUMENT_PATH
    file_pattern: str = FILE_PATTERN
    chunk_size: int = CHUNK_SIZE
    chunk_overlap: int = CHUNK_OVERLAP
    top_k: int = TOP_K

    def build(self) -> RichRAGRuntime:
        from langchain_community.retrievers import BM25Retriever

        logging.info("Loading documents from %s (pattern: %s)...", self.document_path, self.file_pattern)
        langchain_feature_store = build_langchain_feature_store(
            document_path=self.document_path,
            file_pattern=self.file_pattern,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            top_k=self.top_k,
        )
        logging.info("Loaded %s documents", len(langchain_feature_store.documents))
        logging.info("Created %s chunks", len(langchain_feature_store.chunks))
        logging.info("Building FAISS vector store...")

        llm = build_local_hf_llm()
        bm25_retriever = BM25Retriever.from_documents(langchain_feature_store.chunks)
        bm25_retriever.k = self.top_k
        hybrid_retriever = MultiPathRetriever(
            bm25_retriever=bm25_retriever,
            vectorstore=langchain_feature_store.vector_store,
            vector_top_k=self.top_k,
            bm25_top_k=self.top_k,
            reranker=build_llm_reranker(llm),
            rerank_top_k=self.top_k,
        )

        retrieval_client = LangChainRetrievalClient(retriever=hybrid_retriever)
        llm_twin = LLMTwin(model=llm)
        rewrite_chain = create_rewrite_chain(llm)
        rag_chain = build_rag_chain(hybrid_retriever, llm)
        multi_step_agent = build_agent_graph(GraphContext(llm=llm, rag_chain=rag_chain))

        return RichRAGRuntime(
            embedder=langchain_feature_store.embeddings,
            vectorstore=langchain_feature_store.vector_store,
            retrieval_client=retrieval_client,
            llm_twin=llm_twin,
            rewrite_chain=rewrite_chain,
            rag_chain=rag_chain,
            multi_step_agent=multi_step_agent,
            langchain_feature_store=langchain_feature_store,
        )


def build_rag_system() -> RAGSystem:
    db = build_document_store()
    evaluator = EvaluationTracker()
    embedder, vector_db, instruct_dataset = build_feature_store(db.find_all())
    tracker, registry, accepted_model, _ = train_and_register_model(instruct_dataset)

    try:
        runtime = RichRAGBuilder().build()
        deployment_info = Deployer().deploy(FineTunedLLM(model_name="local-hf-llm"))

        return RAGSystem(
            db=db,
            embedder=runtime.embedder,
            vector_db=runtime.vectorstore,
            instruct_dataset=instruct_dataset,
            retrieval_client=runtime.retrieval_client,
            llm_twin=runtime.llm_twin,
            evaluator=evaluator,
            tracker=tracker,
            registry=registry,
            deployment_info=deployment_info,
            langchain_feature_store=runtime.langchain_feature_store,
            rewrite_chain=runtime.rewrite_chain,
            rag_chain=runtime.rag_chain,
            multi_step_agent=runtime.multi_step_agent,
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
        evaluator=evaluator,
        tracker=tracker,
        registry=registry,
        deployment_info=deployment_info,
        langchain_feature_store=None,
        rewrite_chain=None,
        rag_chain=None,
        multi_step_agent=None,
    )
