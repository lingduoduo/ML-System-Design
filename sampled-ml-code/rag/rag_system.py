from __future__ import annotations

from dataclasses import dataclass
import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import logging
from typing import Any, List, Optional

from feature_pipeline import _import_langchain_dependencies
from Inference_pipeline import (
    CrossEncoderReranker,
    LLMTwin,
    GraphContext,
    RAGQueryEngine,
    build_agent_graph,
    build_rag_chain,
)
from data_collection_pipeline import NoSQLDB, build_document_store
from deploy import Deployer
from feature_pipeline import (
    BGE_RERANKER_MODEL,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DOCUMENT_PATH,
    FILE_PATTERN,
    TOP_K,
    RERANK_TOP_N,
    LangChainFeatureStore,
    SimpleEmbedder,
    VectorDB,
    build_feature_store,
    build_langchain_feature_store,
    build_local_hf_llm,
    create_rewrite_chain,
)
from evaluation import EvaluationTracker
from retriever import (
    LangChainRetrievalClient,
    MultiPathRetriever,
    RetrievalClient,
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
    runtime_mode: str
    langchain_feature_store: LangChainFeatureStore | None = None
    rewrite_chain: Any | None = None
    rag_chain: Any | None = None
    multi_step_agent: Any | None = None
    query_engine: RAGQueryEngine | None = None


@dataclass
class RichRAGRuntime:
    embedder: Any
    vectorstore: Any
    retrieval_client: LangChainRetrievalClient
    llm_twin: LLMTwin
    rewrite_chain: Any
    rag_chain: Any
    multi_step_agent: Any
    query_engine: RAGQueryEngine
    langchain_feature_store: LangChainFeatureStore


@dataclass
class CRAGConfig:
    top_k: int = TOP_K
    trigger_web_if_any_no: bool = True
    tavily_max_results: int = 5


@lru_cache(maxsize=1)
def _create_relevance_prompt() -> Any:
    deps = _import_langchain_dependencies()
    return deps["PromptTemplate"].from_template(
        "You are a helpful assistant that determines whether a document chunk is relevant to answering a user's question.\n\n"
        "Question:\n{question}\n\n"
        "Chunk:\n{chunk}\n\n"
        "Respond with exactly one word: yes or no."
    )


@lru_cache(maxsize=1)
def _create_rewrite_query_prompt() -> Any:
    deps = _import_langchain_dependencies()
    return deps["PromptTemplate"].from_template(
        "Rewrite the user question into a concise, search-ready query for a web search.\n\n"
        "Original question:\n{question}\n\n"
        "Search-ready query:"
    )


@lru_cache(maxsize=1)
def _create_answer_prompt() -> Any:
    deps = _import_langchain_dependencies()
    return deps["PromptTemplate"].from_template(
        "Answer the question using only the provided context. If the context does not contain enough information, say you cannot answer.\n\n"
        "Question:\n{question}\n\n"
        "Context:\n{context}\n\n"
        "Answer:"
    )


def _get_search_retriever(vectorstore: Any, top_k: int) -> Any:
    if hasattr(vectorstore, "as_retriever"):
        return vectorstore.as_retriever(search_kwargs={"k": top_k})
    return vectorstore


@lru_cache(maxsize=1)
def _get_str_output_parser() -> Any:
    return _import_langchain_dependencies()["StrOutputParser"]()


def _run_coroutine_sync(coroutine: Any) -> Any:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coroutine)

    with ThreadPoolExecutor(max_workers=1) as executor:
        return executor.submit(lambda: asyncio.run(coroutine)).result()


def _get_document_text(doc: Any) -> str:
    return (getattr(doc, "page_content", "") or "").strip()


def _build_text_chain(prompt: Any, llm: Any) -> Any:
    return prompt | llm | _get_str_output_parser()


def _normalize_relevance_label(result: str) -> str:
    normalized = result.strip().lower()
    return "yes" if normalized.startswith("y") else "no"


def _retrieve_local_documents(question: str, vectorstore: Any, top_k: int) -> list[Any]:
    retriever = _get_search_retriever(vectorstore, top_k)
    try:
        return list(retriever.invoke(question) or [])
    except Exception:
        return []


def local_docs_sufficiency(
    question: str,
    vectorstore: Any,
    llm_grader: Any,
    cfg: CRAGConfig,
) -> tuple[bool, list[Any], list[str]]:
    retrieved_docs = _retrieve_local_documents(question, vectorstore, cfg.top_k)
    if not retrieved_docs:
        return False, [], []

    chunks = [_get_document_text(doc) for doc in retrieved_docs]
    labels = _run_coroutine_sync(grade_relevance(llm_grader, question, chunks))
    return any(label == "yes" for label in labels), retrieved_docs, labels


def crag_query_sync(
    question: str,
    vectorstore: Any,
    cfg: CRAGConfig,
    llm_answer: Any,
    llm_grader: Any,
    llm_rewrite: Any,
    tavily_tool: Optional[Any] = None,
    retrieved_docs: Optional[List[Any]] = None,
    relevance_labels: Optional[List[str]] = None,
) -> dict:
    return _run_coroutine_sync(
        crag_query(
            question=question,
            vectorstore=vectorstore,
            cfg=cfg,
            llm_answer=llm_answer,
            llm_grader=llm_grader,
            llm_rewrite=llm_rewrite,
            tavily_tool=tavily_tool,
            retrieved_docs=retrieved_docs,
            relevance_labels=relevance_labels,
        )
    )


async def grade_relevance(
    llm: Any,
    question: str,
    chunks: List[str],
) -> List[str]:
    """
    Returns list of 'yes'/'no' per chunk.
    """
    chain = _build_text_chain(_create_relevance_prompt(), llm)
    tasks = [
        chain.ainvoke({"question": question, "chunk": chunk})
        for chunk in chunks
    ]
    raw_results = await asyncio.gather(*tasks)
    return [_normalize_relevance_label(str(result)) for result in raw_results]


async def rewrite_query(llm: Any, question: str) -> str:
    chain = _build_text_chain(_create_rewrite_query_prompt(), llm)
    return (await chain.ainvoke({"question": question})).strip()


def should_trigger_web(relevance_labels: List[str], trigger_if_any_no: bool) -> bool:
    if not relevance_labels:
        return True
    if trigger_if_any_no:
        return "no" in relevance_labels
    return relevance_labels.count("no") > relevance_labels.count("yes")


def fuse_context(local_chunks: List[str], web_snippets: List[str]) -> str:
    parts: List[str] = []
    if local_chunks:
        parts.append("LOCAL CONTEXT:\n" + "\n\n".join(local_chunks))
    if web_snippets:
        parts.append("WEB SEARCH CONTEXT:\n" + "\n\n".join(web_snippets))
    return "\n\n".join(parts).strip()


async def answer_question(llm: Any, question: str, context: str) -> str:
    chain = _build_text_chain(_create_answer_prompt(), llm)
    return (await chain.ainvoke({"question": question, "context": context})).strip()


async def crag_query(
    question: str,
    vectorstore: Any,
    cfg: CRAGConfig,
    llm_answer: Any,
    llm_grader: Any,
    llm_rewrite: Any,
    tavily_tool: Optional[Any] = None,
    retrieved_docs: Optional[List[Any]] = None,
    relevance_labels: Optional[List[str]] = None,
) -> dict:
    """
    Full CRAG run: local retrieve -> grade -> optional web correction -> fuse -> answer
    Returns a structured dict for inspection.
    """
    # Step 1: local retrieve
    local_docs = list(retrieved_docs) if retrieved_docs is not None else _retrieve_local_documents(question, vectorstore, cfg.top_k)
    retrieved_chunks = [_get_document_text(doc) for doc in local_docs]

    # Step 2: relevance grading
    relevance = list(relevance_labels) if relevance_labels is not None else await grade_relevance(llm_grader, question, retrieved_chunks)

    # Step 3: keep relevant local text
    relevant_local = [chunk for chunk, lab in zip(retrieved_chunks, relevance) if lab == "yes"]

    # Step 4: correct with query rewrite + web search if needed
    web_snippets: List[str] = []
    transformed_query: Optional[str] = None

    if tavily_tool and should_trigger_web(relevance, cfg.trigger_web_if_any_no):
        transformed_query = await rewrite_query(llm_rewrite, question)
        try:
            web_results = tavily_tool.invoke({"query": transformed_query, "max_results": cfg.tavily_max_results})
            for item in web_results:
                content = item.get("content") or item.get("snippet") or ""
                title = item.get("title") or ""
                if title and content:
                    web_snippets.append(f"{title}\n{content}")
                elif content:
                    web_snippets.append(content)
        except Exception as e:
            web_snippets = [f"[Web search failed: {e}]" ]

    # Step 5: fuse + answer
    fused = fuse_context(relevant_local, web_snippets)
    if not fused:
        final_answer = "Sorry, I could not find relevant information to answer your question."
    else:
        final_answer = await answer_question(llm_answer, question, fused)

    return {
        "question": question,
        "retrieved_count": len(local_docs),
        "relevance_labels": relevance,
        "kept_local_chunks": len(relevant_local),
        "relevant_local": relevant_local,
        "web_snippets": web_snippets,
        "transformed_query": transformed_query,
        "used_web": bool(web_snippets),
        "final_answer": final_answer,
    }


@dataclass
class RichRAGBuilder:
    document_path: str = DOCUMENT_PATH
    file_pattern: str = FILE_PATTERN
    chunk_size: int = CHUNK_SIZE
    chunk_overlap: int = CHUNK_OVERLAP
    top_k: int = TOP_K
    rerank_top_n: int = RERANK_TOP_N
    reranker_model_name: str = BGE_RERANKER_MODEL

    def _build_optional_cross_encoder_reranker(self) -> Any | None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError:
            logging.info("CrossEncoder reranker unavailable; proceeding without query-engine reranking.")
            return None

        logging.info("Loading BGE reranker: %s", self.reranker_model_name)
        model = CrossEncoder(self.reranker_model_name)
        logging.info("BGE reranker ready with top_n=%s", self.rerank_top_n)
        return CrossEncoderReranker(model=model, default_top_n=self.rerank_top_n)

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
        )

        retrieval_client = LangChainRetrievalClient(retriever=hybrid_retriever)
        llm_twin = LLMTwin(model=llm)
        rewrite_chain = create_rewrite_chain(llm)
        rag_chain = build_rag_chain(hybrid_retriever, llm)
        multi_step_agent = build_agent_graph(GraphContext(llm=llm, rag_chain=rag_chain))
        query_engine = RAGQueryEngine(
            retriever=hybrid_retriever,
            llm=llm,
            reranker=self._build_optional_cross_encoder_reranker(),
            default_retrieve_top_k=self.top_k,
        )

        return RichRAGRuntime(
            embedder=langchain_feature_store.embeddings,
            vectorstore=langchain_feature_store.vector_store,
            retrieval_client=retrieval_client,
            llm_twin=llm_twin,
            rewrite_chain=rewrite_chain,
            rag_chain=rag_chain,
            multi_step_agent=multi_step_agent,
            query_engine=query_engine,
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
            runtime_mode="rich",
            langchain_feature_store=runtime.langchain_feature_store,
            rewrite_chain=runtime.rewrite_chain,
            rag_chain=runtime.rag_chain,
            multi_step_agent=runtime.multi_step_agent,
            query_engine=runtime.query_engine,
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
        runtime_mode="fallback",
        langchain_feature_store=None,
        rewrite_chain=None,
        rag_chain=None,
        multi_step_agent=None,
        query_engine=None,
    )
