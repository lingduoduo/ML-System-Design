from __future__ import annotations

import asyncio
import time
import threading
import uuid
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Set

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
from .schema import (
    AgentState,
    PerformanceMetrics,
    RetrievedDocument,
    RouteDecision,
    TaskNode,
    TaskStatus,
    ToolExecutionResult,
    ToolResult,
    ToolType,
    UserRequest,
)
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
        state = self._initialize_state(request)

        if self._try_fill_from_cache(state):
            self._finalize_sync_request(state, start_time, cache_hit=True)
            return state

        self._load_memory_sync(state)

        route_decision = self.router_agent.route(request.message)
        self._log_route(state, route_decision)
        self._retrieve_context(state, route_decision)
        self._plan_and_execute(state, route_decision)
        self._draft_reflect_finalize(state)
        self._finalize_sync_request(state, start_time)
        return state

    async def run_async(self, request: UserRequest) -> AgentState:
        start_time = time.perf_counter()
        state = self._initialize_state(request)

        if self._try_fill_from_cache(state):
            await self._finalize_async_request(state, start_time, cache_hit=True)
            return state

        await self._load_memory_async(state)

        route_decision = self.router_agent.route(request.message)
        self._log_route(state, route_decision)
        await self._retrieve_context_async(state, route_decision)
        self._plan_and_execute(state, route_decision)
        self._draft_reflect_finalize(state)
        await self._finalize_async_request(state, start_time)
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

    def should_retry_task(self, task: TaskNode, result: ToolExecutionResult) -> bool:
        if result.status != TaskStatus.FAILED:
            return False
        if result.retry_count >= task.max_retries:
            return False
        if not task.is_critical and result.retry_count > 1:
            return False
        return True

    def check_early_exit(self, state: Dict[str, Any]) -> bool:
        task_dag = state.get("task_dag", [])
        critical_failed = any(
            task.is_critical and task.status == TaskStatus.FAILED
            for task in task_dag
        )
        if critical_failed:
            return True
        if state.get("error_count", 0) > 3:
            return True
        return False

    def execute_task(self, task: TaskNode, state: Dict[str, Any]) -> ToolExecutionResult:
        processed_params = self._process_task_params(task.params, state)

        agent = self.tool_registry.agents.get(task.tool_name)
        if not agent:
            return ToolExecutionResult(
                tool_name=task.tool_name,
                status=TaskStatus.FAILED,
                result=None,
                performance=PerformanceMetrics(0, 0, 0, 0),
                error_code="AGENT_NOT_FOUND",
                error_message=f"Tool agent {task.tool_name} not found",
            )

        task.status = TaskStatus.RUNNING
        result = agent.execute(processed_params)
        task.result = result
        task.status = result.status

        self.monitoring_dashboard.update_task_status(task.task_id, result)
        return result

    def dynamic_dispatch(self, state: Dict[str, Any]) -> Optional[TaskNode]:
        task_dag = state.get("task_dag", [])
        completed_tasks = state.get("completed_tasks", {})
        best_task: Optional[TaskNode] = None
        best_key: Optional[tuple[int, float, str]] = None
        for task in task_dag:
            if task.status not in {TaskStatus.PENDING, TaskStatus.RETRYING}:
                continue
            if not all(dep_id in completed_tasks for dep_id in task.dependencies):
                continue

            candidate_key = (task.priority, -task.timeout, task.task_id)
            if best_key is None or candidate_key < best_key:
                best_task = task
                best_key = candidate_key

        return best_task

    def get_dashboard_summary(self) -> Dict[str, Any]:
        """Get comprehensive monitoring dashboard summary."""
        return self.monitoring_dashboard.get_dashboard_summary()

    def get_monitoring_metrics(self) -> Dict[str, float]:
        """Get quick performance summary from monitoring."""
        return self.monitoring_dashboard.get_performance_summary()

    def _process_task_params(self, params: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
        processed: Dict[str, Any] = {}
        for key, value in params.items():
            if isinstance(value, str) and value.startswith("{{") and value.endswith("}}"):
                state_key = value[2:-2]
                processed[key] = state.get(state_key, "")
            else:
                processed[key] = value
        return processed

    def _initialize_state(self, request: UserRequest) -> AgentState:
        state = AgentState(request_id=str(uuid.uuid4()), user_request=request)
        self.gateway.check(request)
        self.logger.log(state, "gateway", {"status": "passed"})
        return state

    def _log_memory_loaded(self, state: AgentState) -> None:
        self.logger.log(
            state,
            "memory_loaded",
            {
                "short_term_count": len(state.short_term_memory),
                "long_term_keys": list(state.long_term_memory.keys()),
            },
        )

    def _load_memory_sync(self, state: AgentState) -> None:
        request = state.user_request
        state.short_term_memory = self.memory_store.load_short_term(request.user_id)
        state.long_term_memory = self.memory_store.load_long_term(request.user_id)
        self._log_memory_loaded(state)

    async def _load_memory_async(self, state: AgentState) -> None:
        request = state.user_request
        state.short_term_memory, state.long_term_memory = await asyncio.gather(
            self.memory_store.load_short_term_async(request.user_id),
            self.memory_store.load_long_term_async(request.user_id),
        )
        self._log_memory_loaded(state)

    def _finalize_sync_request(self, state: AgentState, start_time: float, *, cache_hit: bool = False) -> None:
        request = state.user_request
        self.memory_store.save_turn(
            user_id=request.user_id,
            user_message=request.message,
            assistant_message=state.final_response,
        )
        self._update_metrics_batch(
            cache_hit=cache_hit,
            response_time=time.perf_counter() - start_time,
        )

    async def _finalize_async_request(self, state: AgentState, start_time: float, *, cache_hit: bool = False) -> None:
        request = state.user_request
        await self.memory_store.save_turn_async(
            user_id=request.user_id,
            user_message=request.message,
            assistant_message=state.final_response,
        )
        self._update_metrics_batch(
            cache_hit=cache_hit,
            response_time=time.perf_counter() - start_time,
        )

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

        task_dag = self._build_task_dag(state)
        if not task_dag:
            return

        execution_state: Dict[str, Any] = {
            "task_dag": task_dag,
            "completed_tasks": {},
            "failed_tasks": {},
            "error_count": 0,
        }

        while True:
            if self.check_early_exit(execution_state):
                self.logger.log(
                    state,
                    "tool_execution_stopped",
                    {
                        "reason": "early_exit",
                        "error_count": execution_state["error_count"],
                    },
                )
                break

            next_task = self.dynamic_dispatch(execution_state)
            if next_task is None:
                break

            task_result = self.execute_task(next_task, execution_state)
            result = self._to_tool_result(task_result)
            state.tool_results.append(result)

            if task_result.status == TaskStatus.COMPLETED:
                execution_state["completed_tasks"][next_task.task_id] = task_result
                execution_state[next_task.task_id] = task_result.result
                execution_state[next_task.tool_name] = task_result.result
            else:
                execution_state["error_count"] += 1
                if self.should_retry_task(next_task, task_result):
                    task_result.retry_count += 1
                    next_task.result = task_result
                    next_task.status = TaskStatus.RETRYING
                    self.logger.log(
                        state,
                        "tool_retry_scheduled",
                        {
                            "task_id": next_task.task_id,
                            "tool_name": next_task.tool_name,
                            "retry_count": task_result.retry_count,
                        },
                    )
                    continue

                execution_state["failed_tasks"][next_task.task_id] = task_result
                if not next_task.is_critical:
                    next_task.status = TaskStatus.SKIPPED

            self.logger.log(
                state,
                "tool_execution",
                {
                    "task_id": next_task.task_id,
                    "tool_name": next_task.tool_name,
                    "success": result.success,
                    "error": result.error,
                    "execution_time": task_result.performance.execution_time,
                },
            )

    def _build_task_dag(self, state: AgentState) -> List[TaskNode]:
        if not state.plan:
            return []

        task_dag: List[TaskNode] = []
        previous_task_id: Optional[str] = None
        for priority, step in enumerate(state.plan.steps, start=1):
            if not step.needs_tool or not step.tool_call:
                continue

            task_id = f"{state.request_id}:{step.step_id}"
            task_dag.append(
                TaskNode(
                    task_id=task_id,
                    tool_name=step.tool_call.tool_name,
                    tool_type=self._resolve_tool_type(step.tool_call.tool_name),
                    params=dict(step.tool_call.arguments),
                    dependencies=[previous_task_id] if previous_task_id else [],
                    priority=priority,
                    max_retries=2,
                    timeout=30.0,
                    is_critical=priority == 1,
                )
            )
            previous_task_id = task_id

        return task_dag

    def _resolve_tool_type(self, tool_name: str) -> ToolType:
        agent = self.tool_registry.agents.get(tool_name)
        if agent is not None:
            return agent.tool_type
        return ToolType.GENERATION

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

    def _cache_response(self, request: UserRequest, response: str) -> None:
        """Cache response with O(1) operations."""
        dedupe_key = self.gateway.dedupe_key(request)
        with self._cache_lock:
            # Remove old entry if exists to update LRU order
            if dedupe_key in self.response_cache:
                del self.response_cache[dedupe_key]
            
            # Add new entry
            self.response_cache[dedupe_key] = response
            
            # Evict oldest if over limit
            while len(self.response_cache) > RESPONSE_CACHE_SIZE:
                self.response_cache.popitem(last=False)

    def _update_metrics_batch(self, cache_hit: bool = False, response_time: float = 0.0) -> None:
        """Batch metric updates to reduce lock contention."""
        with self._stats_lock:
            self.performance_stats["total_requests"] += 1
            if cache_hit:
                self.performance_stats["cache_hits"] += 1
            if response_time > 0:
                self.performance_stats["total_response_time"] += response_time
                self.performance_stats["avg_response_time"] = (
                    self.performance_stats["total_response_time"] / self.performance_stats["total_requests"]
                )

    @staticmethod
    def _to_tool_result(task_result: ToolExecutionResult) -> ToolResult:
        return ToolResult(
            tool_name=task_result.tool_name,
            success=task_result.status == TaskStatus.COMPLETED,
            output=task_result.result,
            error=task_result.error_message,
        )
