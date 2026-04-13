from __future__ import annotations

import time
from collections import deque
from typing import Any, Callable, Dict, List

from .retrieval import summarize_document
from .schema import PerformanceMetrics, TaskStatus, ToolCall, ToolExecutionResult, ToolResult, ToolType


class BaseToolAgent:
    """Base class for tool agents with performance tracking."""

    __slots__ = (
        "name",
        "tool_type",
        "execution_history",
        "call_count",
        "success_count",
        "total_execution_time",
        "_metrics_dirty",
        "_cached_metrics",
    )

    def __init__(self, name: str, tool_type: ToolType, max_history: int = 100):
        self.name = name
        self.tool_type = tool_type
        self.execution_history: deque[ToolExecutionResult] = deque(maxlen=max_history)
        self.call_count = 0
        self.success_count = 0
        self.total_execution_time = 0.0
        self._metrics_dirty = True
        self._cached_metrics = {
            "call_count": 0,
            "success_count": 0,
            "success_rate": 0.0,
            "avg_execution_time": 0.0,
        }

    def execute(self, params: Dict[str, Any]) -> ToolExecutionResult:
        """Execute tool with performance metrics."""
        start_time = time.perf_counter()
        self.call_count += 1

        try:
            result = self._execute_core(params)
            execution_time = time.perf_counter() - start_time
            self.total_execution_time += execution_time
            self.success_count += 1
            self._metrics_dirty = True

            performance = PerformanceMetrics(
                execution_time=execution_time,
                cost_estimate=self._estimate_cost(params),
                memory_usage=self._get_memory_usage(),
                success_rate=self._calculate_success_rate(),
            )

            out = ToolExecutionResult(
                tool_name=self.name,
                status=TaskStatus.COMPLETED,
                result=result,
                performance=performance,
                optimization_suggestions=self._generate_optimization_suggestions(performance),
            )
            self.execution_history.append(out)
            return out

        except Exception as e:
            execution_time = time.perf_counter() - start_time
            self.total_execution_time += execution_time
            self._metrics_dirty = True
            out = ToolExecutionResult(
                tool_name=self.name,
                status=TaskStatus.FAILED,
                result=None,
                performance=PerformanceMetrics(execution_time, 0, 0, 0),
                error_code="EXECUTION_ERROR",
                error_message=str(e),
            )
            self.execution_history.append(out)
            return out

    def _execute_core(self, params: Dict[str, Any]) -> Any:
        """Execute core tool logic. Override in subclasses."""
        raise NotImplementedError

    def _estimate_cost(self, params: Dict[str, Any]) -> float:
        """Estimate execution cost. Override for custom logic."""
        return 0.01

    def _get_memory_usage(self) -> float:
        """Estimate memory usage in MB. Override for custom logic."""
        return 10.0

    def _calculate_success_rate(self) -> float:
        """Calculate success rate from cumulative execution counts."""
        return self.success_count / max(1, self.call_count)

    def _generate_optimization_suggestions(self, performance: PerformanceMetrics) -> List[str]:
        """Generate optimization suggestions based on metrics."""
        suggestions = []
        if performance.execution_time > 5.0:
            suggestions.append(
                f"High latency ({performance.execution_time:.2f}s): Consider caching or optimization"
            )
        if performance.memory_usage > 100.0:
            suggestions.append(
                f"High memory usage ({performance.memory_usage:.1f}MB): Consider batch processing"
            )
        return suggestions

    def _update_cached_metrics(self) -> None:
        self._cached_metrics["call_count"] = self.call_count
        self._cached_metrics["success_count"] = self.success_count
        self._cached_metrics["success_rate"] = self._calculate_success_rate()
        self._cached_metrics["avg_execution_time"] = (
            self.total_execution_time / max(1, self.call_count)
        )
        self._metrics_dirty = False

    def get_metrics(self) -> Dict[str, float]:
        """Get agent performance metrics."""
        if self._metrics_dirty:
            self._update_cached_metrics()
        return dict(self._cached_metrics)


class FunctionToolAgent(BaseToolAgent):
    """Adapter that wraps a plain callable in a monitored tool agent."""
    __slots__ = ('_func', '_cost_estimate', '_memory_usage')

    def __init__(
        self,
        name: str,
        tool_type: ToolType,
        func: Callable[..., Any],
        *,
        cost_estimate: float = 0.01,
        memory_usage: float = 10.0,
        max_history: int = 100,
    ):
        super().__init__(name, tool_type, max_history=max_history)
        self._func = func
        self._cost_estimate = cost_estimate
        self._memory_usage = memory_usage

    def _execute_core(self, params: Dict[str, Any]) -> Any:
        return self._func(**params)

    def _estimate_cost(self, params: Dict[str, Any]) -> float:
        return self._cost_estimate

    def _get_memory_usage(self) -> float:
        return self._memory_usage


class ToolRegistry:
    def __init__(self):
        self.tools: Dict[str, Callable[..., Any]] = {}
        self.agents: Dict[str, BaseToolAgent] = {}
        self._agent_cache: Dict[str, BaseToolAgent | None] = {}

    def register(
        self,
        name: str,
        func: Callable[..., Any],
        *,
        agent: BaseToolAgent | None = None,
    ) -> None:
        self.tools[name] = func
        self._agent_cache.clear()  # Invalidate cache on registration
        if agent is not None:
            self.register_agent(agent)

    def register_agent(self, agent: BaseToolAgent) -> None:
        """Register a tool agent."""
        self.agents[agent.name] = agent
        self._agent_cache[agent.name] = agent

    def execute(self, tool_call: ToolCall) -> ToolResult:
        if tool_call.tool_name not in self.tools:
            return ToolResult(
                tool_name=tool_call.tool_name,
                success=False,
                output=None,
                error=f"Unknown tool: {tool_call.tool_name}",
            )

        try:
            result = self.tools[tool_call.tool_name](**tool_call.arguments)
            return ToolResult(tool_name=tool_call.tool_name, success=True, output=result)
        except Exception as exc:
            return ToolResult(
                tool_name=tool_call.tool_name,
                success=False,
                output=None,
                error=str(exc),
            )

    def get_agent_metrics(self, tool_name: str) -> Dict[str, float]:
        """Get metrics for a tool agent."""
        agent = self.agents.get(tool_name)
        if agent:
            return agent.get_metrics()
        return {}

    def execute_with_metrics(self, tool_call: ToolCall) -> ToolExecutionResult:
        """Execute tool and return result with metrics."""
        # Check cache first
        if tool_call.tool_name not in self._agent_cache:
            self._agent_cache[tool_call.tool_name] = self.agents.get(tool_call.tool_name)
        
        agent = self._agent_cache.get(tool_call.tool_name)
        
        if agent is not None:
            return agent.execute(tool_call.arguments)

        # Fall back to basic execution without metrics
        result = self.execute(tool_call)
        return ToolExecutionResult(
            tool_name=result.tool_name,
            status=TaskStatus.COMPLETED if result.success else TaskStatus.FAILED,
            result=result.output,
            performance=PerformanceMetrics(
                execution_time=0.0,
                cost_estimate=0.0,
                memory_usage=0.0,
                success_rate=1.0 if result.success else 0.0,
            ),
            error_message=result.error,
        )


def search_orders(order_id: str) -> Dict[str, Any]:
    return {"order_id": order_id, "status": "shipped", "eta": "2026-04-15"}


def create_ticket(issue: str, severity: str = "medium") -> Dict[str, Any]:
    return {"ticket_id": "TICK-123", "issue": issue, "severity": severity}


def summarize_user_docs(_: str = "") -> str:
    return summarize_document("user_docs", "user documentation")


def summarize_policy_docs(_: str = "") -> str:
    return summarize_document("policy_docs", "refund policy")


def build_tool_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(
        "search_orders",
        search_orders,
        agent=FunctionToolAgent(
            "search_orders",
            ToolType.DATA_RETRIEVAL,
            search_orders,
            cost_estimate=0.002,
            memory_usage=5.0,
        ),
    )
    registry.register(
        "create_ticket",
        create_ticket,
        agent=FunctionToolAgent(
            "create_ticket",
            ToolType.GENERATION,
            create_ticket,
            cost_estimate=0.005,
            memory_usage=6.0,
        ),
    )
    registry.register(
        "summarize_user_docs",
        summarize_user_docs,
        agent=FunctionToolAgent(
            "summarize_user_docs",
            ToolType.ANALYSIS,
            summarize_user_docs,
            cost_estimate=0.02,
            memory_usage=12.0,
        ),
    )
    registry.register(
        "summarize_policy_docs",
        summarize_policy_docs,
        agent=FunctionToolAgent(
            "summarize_policy_docs",
            ToolType.ANALYSIS,
            summarize_policy_docs,
            cost_estimate=0.02,
            memory_usage=12.0,
        ),
    )
    return registry
