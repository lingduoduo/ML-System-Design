from __future__ import annotations

import time
from typing import Any, Callable, Dict, List

from .retrieval import summarize_document
from .schema import PerformanceMetrics, TaskStatus, ToolCall, ToolExecutionResult, ToolResult


class BaseToolAgent:
    """Base class for tool agents with performance tracking."""

    def __init__(self, name: str):
        self.name = name
        self.execution_history: List[ToolExecutionResult] = []
        self.call_count = 0
        self.success_count = 0

    def execute(self, params: Dict[str, Any]) -> ToolExecutionResult:
        """Execute tool with performance metrics."""
        start_time = time.time()
        self.call_count += 1

        try:
            result = self._execute_core(params)
            execution_time = time.time() - start_time
            self.success_count += 1

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
            execution_time = time.time() - start_time
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
        """Calculate success rate from history."""
        if not self.execution_history:
            return 1.0
        successful = sum(
            1 for r in self.execution_history
            if r.status == TaskStatus.COMPLETED
        )
        return successful / len(self.execution_history)

    def _generate_optimization_suggestions(self, performance: PerformanceMetrics) -> List[str]:
        """Generate optimization suggestions based on metrics."""
        suggestions = []
        if performance.execution_time > 5.0:
            suggestions.append(f"High latency ({performance.execution_time:.2f}s): Consider caching or optimization")
        if performance.memory_usage > 100.0:
            suggestions.append(f"High memory usage ({performance.memory_usage:.1f}MB): Consider batch processing")
        return suggestions

    def get_metrics(self) -> Dict[str, float]:
        """Get agent performance metrics."""
        return {
            "call_count": self.call_count,
            "success_count": self.success_count,
            "success_rate": self.success_count / max(1, self.call_count),
            "avg_execution_time": sum(
                r.performance.execution_time for r in self.execution_history
            ) / max(1, len(self.execution_history)),
        }


class ToolRegistry:
    def __init__(self):
        self.tools: Dict[str, Callable[..., Any]] = {}
        self.agents: Dict[str, BaseToolAgent] = {}

    def register(self, name: str, func: Callable[..., Any]) -> None:
        self.tools[name] = func

    def register_agent(self, agent: BaseToolAgent) -> None:
        """Register a tool agent."""
        self.agents[agent.name] = agent

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
    registry.register("search_orders", search_orders)
    registry.register("create_ticket", create_ticket)
    registry.register("summarize_user_docs", summarize_user_docs)
    registry.register("summarize_policy_docs", summarize_policy_docs)
    return registry
