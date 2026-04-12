from __future__ import annotations

import asyncio
import time
import threading
import uuid
from collections import OrderedDict
from typing import Dict, List, Optional, Set

from .approval import HumanApprovalService
from .config import DEFAULT_TOP_K, RESPONSE_CACHE_SIZE, TORCH_AVAILABLE
from .gateway import Gateway
from .memory import MemoryStore
from .monitoring import MonitoringDashboard
from .observability import TraceLogger
from .planner import ReturnPlannerAgent
from .qa import QAAgent
from .reflection import ReflectionModel
from .retrieval import Reranker, RetrievalModel
from .router import RouterAgent
from .schema import AgentState, RetrievedDocument, RouteDecision, TaskNode, TaskStatus, TaskType, UserRequest
from .tools import ToolRegistry

class AgentWorkflow:
    def __init__(
        self,
        gateway: Gateway,
        memory_store: MemoryStore,
        router_agent: RouterAgent,
        planner_agent: ReturnPlannerAgent,
        qa_agent: QAAgent,
        reflection_model: ReflectionModel,
        human_approval: HumanApprovalService,
        user_doc_retriever: RetrievalModel,
        policy_retriever: RetrievalModel,
        reranker: Reranker,
        tool_registry: ToolRegistry,
        logger: TraceLogger,
    ):
        self.gateway = gateway
        self.memory_store = memory_store
        self.router_agent = router_agent
        self.planner_agent = planner_agent
        self.qa_agent = qa_agent
        self.reflection_model = reflection_model
        self.human_approval = human_approval
        self.user_doc_retriever = user_doc_retriever
        self.policy_retriever = policy_retriever
        self.reranker = reranker
        self.tool_registry = tool_registry
        self.logger = logger
        self.response_cache: OrderedDict[str, str] = OrderedDict()
        self._cache_lock = threading.RLock()
        self._stats_lock = threading.RLock()
        self.monitoring_dashboard = MonitoringDashboard()
        self.performance_stats = {
            "total_requests": 0,
            "cache_hits": 0,
            "avg_response_time": 0.0,
            "total_response_time": 0.0,
            "gpu_accelerated": TORCH_AVAILABLE,
        }

    def run(self, request: UserRequest) -> AgentState:
        start_time = time.perf_counter()
        self._increment_total_requests()

        state = AgentState(request_id=str(uuid.uuid4()), user_request=request)
        self.gateway.check(request)
        self.logger.log(state, "gateway", {"status": "passed"})

        if self._try_fill_from_cache(state):
            self._increment_cache_hits()
            self.memory_store.save_turn(
                user_id=request.user_id,
                user_message=request.message,
                assistant_message=state.final_response,
            )
            self._update_response_time(start_time)
            return state

        state.short_term_memory = self.memory_store.load_short_term(request.user_id)
        state.long_term_memory = self.memory_store.load_long_term(request.user_id)
        self.logger.log(
            state,
            "memory_loaded",
            {
                "short_term_count": len(state.short_term_memory),
                "long_term_keys": list(state.long_term_memory.keys()),
            },
        )

        route_decision = self.router_agent.route(request.message)
        self._log_route(state, route_decision)
        self._retrieve_context(state, route_decision)
        self._plan_and_execute(state, route_decision)
        self._draft_reflect_finalize(state)
        self.memory_store.save_turn(
            user_id=request.user_id,
            user_message=request.message,
            assistant_message=state.final_response,
        )
        self._update_response_time(start_time)
        return state

    async def run_async(self, request: UserRequest) -> AgentState:
        start_time = time.perf_counter()
        self._increment_total_requests()

        state = AgentState(request_id=str(uuid.uuid4()), user_request=request)
        self.gateway.check(request)
        self.logger.log(state, "gateway", {"status": "passed"})

        if self._try_fill_from_cache(state):
            await self.memory_store.save_turn_async(
                user_id=request.user_id,
                user_message=request.message,
                assistant_message=state.final_response,
            )
            self._increment_cache_hits()
            self._update_response_time(start_time)
            return state

        memory_tasks = await asyncio.gather(
            self.memory_store.load_short_term_async(request.user_id),
            self.memory_store.load_long_term_async(request.user_id),
        )
        state.short_term_memory = memory_tasks[0]
        state.long_term_memory = memory_tasks[1]

        self.logger.log(
            state,
            "memory_loaded",
            {
                "short_term_count": len(state.short_term_memory),
                "long_term_keys": list(state.long_term_memory.keys()),
            },
        )

        route_decision = self.router_agent.route(request.message)
        self._log_route(state, route_decision)
        await self._retrieve_context_async(state, route_decision)
        self._plan_and_execute(state, route_decision)
        self._draft_reflect_finalize(state)
        await self.memory_store.save_turn_async(
            user_id=request.user_id,
            user_message=request.message,
            assistant_message=state.final_response,
        )
        self._update_response_time(start_time)
        return state

    def _retrieve_documents(
        self,
        message: str,
        route: Optional[str],
        target_routes: Set[str],
        retriever: RetrievalModel,
        top_k: int = DEFAULT_TOP_K,
    ) -> List[RetrievedDocument]:
        if route not in target_routes:
            return []

        documents = retriever.retrieve(message, top_k=top_k)
        return self.reranker.rerank(message, documents)

    async def _retrieve_documents_async(
        self,
        query: str,
        route: Optional[str],
        target_routes: Set[str],
        retriever: RetrievalModel,
    ) -> List[RetrievedDocument]:
        if route in target_routes:
            loop = asyncio.get_running_loop()
            documents = await loop.run_in_executor(None, retriever.retrieve, query, DEFAULT_TOP_K)
            return self.reranker.rerank(query, documents)
        return []

    def get_performance_stats(self) -> Dict[str, float]:
        with self._stats_lock:
            return dict(self.performance_stats)

    def batch_run(self, requests: List[UserRequest]) -> List[AgentState]:
        if not requests:
            return []
        return [self.run(request) for request in requests]

    async def batch_run_async(self, requests: List[UserRequest]) -> List[AgentState]:
        if not requests:
            return []
        tasks = [self.run_async(request) for request in requests]
        return await asyncio.gather(*tasks)

    def should_retry_task(self, task: TaskNode) -> bool:
        """Check if task should be retried."""
        if task.status != TaskStatus.FAILED:
            return False
        if task.retry_count >= task.max_retries:
            return False
        # Don't retry non-critical tasks multiple times
        if not task.is_critical and task.retry_count > 0:
            return False
        return True

    def check_early_exit(self, completed_tasks: Dict[str, bool], failed_tasks: Dict[str, bool]) -> bool:
        """Check if execution should be terminated early."""
        # Exit if any critical task has failed
        # This would need to be connected to task criticality tracking
        if len(failed_tasks) > 3:
            return True
        return False

    def get_dashboard_summary(self) -> Dict[str, any]:
        """Get comprehensive monitoring dashboard summary."""
        return self.monitoring_dashboard.get_dashboard_summary()

    def get_monitoring_metrics(self) -> Dict[str, float]:
        """Get quick performance summary from monitoring."""
        return self.monitoring_dashboard.get_performance_summary()

    def _try_fill_from_cache(self, state: AgentState) -> bool:
        dedupe_key = self.gateway.dedupe_key(state.user_request)
        with self._cache_lock:
            cached_response = self.response_cache.get(dedupe_key)
            if cached_response is None:
                return False

            self.response_cache.move_to_end(dedupe_key)
        state.final_response = cached_response
        self.logger.log(state, "cache_hit", {"dedupe_key": dedupe_key})
        self.logger.log(state, "completed", {"request_id": state.request_id, "cached": True})
        return True

    def _log_route(self, state: AgentState, route_decision: RouteDecision) -> None:
        state.route = route_decision.intent
        self.logger.log(
            state,
            "routing",
            {
                "route": route_decision.intent,
                "confidence": route_decision.confidence,
                "should_plan": route_decision.should_plan,
                "should_use_tools": route_decision.should_use_tools,
                "target_agent": route_decision.target_agent,
            },
        )

    def _retrieve_context(self, state: AgentState, route_decision: RouteDecision) -> None:
        if not route_decision.should_retrieve:
            return

        state.retrieved_user_docs = self._retrieve_documents(
            state.user_request.message,
            route_decision.intent,
            {"general_qa", "tool_workflow"},
            self.user_doc_retriever,
        )
        state.retrieved_policy_docs = self._retrieve_documents(
            state.user_request.message,
            route_decision.intent,
            {"policy_qa", "tool_workflow"},
            self.policy_retriever,
        )
        self._log_retrieval(state)

    async def _retrieve_context_async(self, state: AgentState, route_decision: RouteDecision) -> None:
        if not route_decision.should_retrieve:
            return

        user_docs, policy_docs = await asyncio.gather(
            self._retrieve_documents_async(
                state.user_request.message,
                route_decision.intent,
                {"general_qa", "tool_workflow"},
                self.user_doc_retriever,
            ),
            self._retrieve_documents_async(
                state.user_request.message,
                route_decision.intent,
                {"policy_qa", "tool_workflow"},
                self.policy_retriever,
            ),
        )
        state.retrieved_user_docs = user_docs
        state.retrieved_policy_docs = policy_docs
        self._log_retrieval(state)

    def _log_retrieval(self, state: AgentState) -> None:
        self.logger.log(
            state,
            "retrieval",
            {
                "user_docs": [doc.doc_id for doc in state.retrieved_user_docs],
                "policy_docs": [doc.doc_id for doc in state.retrieved_policy_docs],
            },
        )

    def _plan_and_execute(self, state: AgentState, route_decision: RouteDecision) -> None:
        if not route_decision.should_plan:
            self.logger.log(
                state,
                "planning_skipped",
                {"reason": "router_selected_direct_qa", "route": route_decision.intent},
            )
            return

        state.plan = self.planner_agent.create_plan(state)
        self.logger.log(
            state,
            "planning",
            {
                "workflow": state.plan.workflow,
                "steps": [step.description for step in state.plan.steps],
                "requires_human_approval": state.plan.requires_human_approval,
            },
        )

        if route_decision.should_use_tools:
            self._execute_tool_steps(state)

    def _execute_tool_steps(self, state: AgentState) -> None:
        if not state.plan:
            return

        for step in state.plan.steps:
            if not step.needs_tool or not step.tool_call:
                continue

            result = self.tool_registry.execute(step.tool_call)
            state.tool_results.append(result)
            self.logger.log(
                state,
                "tool_execution",
                {
                    "tool_name": step.tool_call.tool_name,
                    "success": result.success,
                    "error": result.error,
                },
            )

    def _draft_reflect_finalize(self, state: AgentState) -> None:
        state.draft_response = self.qa_agent.generate_response(state)
        self.logger.log(state, "draft_response", {"length": len(state.draft_response)})

        reflection_result = self.reflection_model.review(state)
        self.logger.log(state, "reflection", {"result": reflection_result})
        if reflection_result.startswith("REFINE:"):
            state.draft_response += f"\n\n[Reflection Notes] {reflection_result}"

        requires_human_approval = state.plan.requires_human_approval if state.plan else False
        if requires_human_approval:
            approved = self.human_approval.request_approval(state)
            self.logger.log(state, "human_approval", {"approved": approved})
            if not approved:
                state.final_response = "This request requires human review before completion."
                return

        state.final_response = state.draft_response
        self._cache_response(state.user_request, state.final_response)
        self.logger.log(state, "completed", {"request_id": state.request_id, "cached": False})

    def _update_response_time(self, start_time: float) -> None:
        response_time = time.perf_counter() - start_time
        with self._stats_lock:
            self.performance_stats["total_response_time"] += response_time
            self.performance_stats["avg_response_time"] = (
                self.performance_stats["total_response_time"] / self.performance_stats["total_requests"]
            )

    def _cache_response(self, request: UserRequest, response: str) -> None:
        dedupe_key = self.gateway.dedupe_key(request)
        with self._cache_lock:
            self.response_cache[dedupe_key] = response
            self.response_cache.move_to_end(dedupe_key)
            if len(self.response_cache) > RESPONSE_CACHE_SIZE:
                self.response_cache.popitem(last=False)

    def _increment_total_requests(self) -> None:
        with self._stats_lock:
            self.performance_stats["total_requests"] += 1

    def _increment_cache_hits(self) -> None:
        with self._stats_lock:
            self.performance_stats["cache_hits"] += 1
