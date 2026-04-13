from __future__ import annotations

from .approval import HumanApprovalService
from .config import (
    ENABLE_GUMBEL_TOOL_SELECTION,
    GUMBEL_TOOL_HARD,
    GUMBEL_TOOL_TEMPERATURE,
    TOOL_SELECTOR_MODEL_PATH,
)
from .gateway import Gateway
from .memory import MemoryStore
from .observability import TraceLogger
from .planner import ReturnPlannerAgent
from .qa import QAAgent
from .reflection import ReflectionModel
from .retrieval import Reranker, RetrievalModel
from .router import LabelingModel, RouterAgent
from .tool_selection import ToolSelectionModel, load_model
from .tools import build_tool_registry
from .workflow import AgentWorkflow


def build_workflow() -> AgentWorkflow:
    gateway = Gateway(rate_limit_per_user=100)
    memory_store = MemoryStore()
    router_agent = RouterAgent(labeling_model=LabelingModel())
    trained_selector = load_model(TOOL_SELECTOR_MODEL_PATH) if TOOL_SELECTOR_MODEL_PATH else None
    planner_agent = ReturnPlannerAgent(
        tool_selector=ToolSelectionModel(
            enable_gumbel=ENABLE_GUMBEL_TOOL_SELECTION,
            temperature=GUMBEL_TOOL_TEMPERATURE,
            hard=GUMBEL_TOOL_HARD,
            trained_selector=trained_selector,
        )
    )
    qa_agent = QAAgent()
    reflection_model = ReflectionModel()
    human_approval = HumanApprovalService()
    user_doc_retriever = RetrievalModel(corpus_name="user_docs")
    policy_retriever = RetrievalModel(corpus_name="policy_docs")
    reranker = Reranker()
    tool_registry = build_tool_registry()
    logger = TraceLogger()

    return AgentWorkflow(
        gateway=gateway,
        memory_store=memory_store,
        router_agent=router_agent,
        planner_agent=planner_agent,
        qa_agent=qa_agent,
        reflection_model=reflection_model,
        human_approval=human_approval,
        user_doc_retriever=user_doc_retriever,
        policy_retriever=policy_retriever,
        reranker=reranker,
        tool_registry=tool_registry,
        logger=logger,
    )
