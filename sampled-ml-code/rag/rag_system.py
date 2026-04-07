from __future__ import annotations

from dataclasses import dataclass

from Inference_pipeline import LLMTwin
from data_collection_pipeline import NoSQLDB, build_document_store
from deploy import Deployer
from feature_pipeline import SimpleEmbedder, VectorDB, build_feature_store
from monitoring import Monitor
from retriever import RetrievalClient
from training_pipeline import (
    ExperimentTracker,
    FineTunedLLM,
    ModelRegistry,
    train_and_register_model,
)


@dataclass
class RAGSystem:
    db: NoSQLDB
    embedder: SimpleEmbedder
    vector_db: VectorDB
    instruct_dataset: list[dict]
    retrieval_client: RetrievalClient
    llm_twin: LLMTwin
    monitor: Monitor
    tracker: ExperimentTracker
    registry: ModelRegistry
    deployment_info: dict


def build_rag_system() -> RAGSystem:
    db = build_document_store()
    embedder, vector_db, instruct_dataset = build_feature_store(db.find_all())
    tracker, registry, accepted_model, _ = train_and_register_model(instruct_dataset)

    if accepted_model is None:
        accepted_model = FineTunedLLM(model_name="fallback-llm")

    retrieval_client = RetrievalClient(embedder=embedder, vector_db=vector_db)
    llm_twin = LLMTwin(model=accepted_model)
    monitor = Monitor()
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
    )
