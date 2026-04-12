"""Performance monitoring and dashboarding for task execution.

Provides real-time metrics tracking, performance analytics, and execution dashboards
for the agentic RAG system.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List

from .schema import PerformanceMetrics, TaskStatus, ToolExecutionResult


@dataclass
class SystemMetrics:
    """Aggregated system-level metrics."""
    total_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    skipped_tasks: int = 0
    average_execution_time: float = 0.0
    total_cost: float = 0.0
    total_memory_usage: float = 0.0
    success_rate: float = 1.0


class MonitoringDashboard:
    """Tracks execution metrics and provides performance analytics."""

    def __init__(self, max_history: int = 1000):
        self.task_metrics: Dict[str, ToolExecutionResult] = {}
        self.system_metrics = SystemMetrics()
        self.max_history = max_history
        self.execution_history: List[SystemMetrics] = []

    def update_task_status(self, task_id: str, result: ToolExecutionResult) -> None:
        """Update metrics for a task execution."""
        self.task_metrics[task_id] = result
        self._aggregate_metrics()

    def _aggregate_metrics(self) -> None:
        """Recalculate aggregated system metrics."""
        completed = sum(
            1 for r in self.task_metrics.values()
            if r.status == TaskStatus.COMPLETED
        )
        failed = sum(
            1 for r in self.task_metrics.values()
            if r.status == TaskStatus.FAILED
        )
        skipped = sum(
            1 for r in self.task_metrics.values()
            if r.status == TaskStatus.SKIPPED
        )

        total_time = sum(r.performance.execution_time for r in self.task_metrics.values())
        avg_time = total_time / max(1, len(self.task_metrics))
        total_cost = sum(r.performance.cost_estimate for r in self.task_metrics.values())
        total_memory = sum(r.performance.memory_usage for r in self.task_metrics.values())

        total = len(self.task_metrics)
        success_rate = completed / max(1, total)

        self.system_metrics = SystemMetrics(
            total_tasks=total,
            completed_tasks=completed,
            failed_tasks=failed,
            skipped_tasks=skipped,
            average_execution_time=avg_time,
            total_cost=total_cost,
            total_memory_usage=total_memory,
            success_rate=success_rate,
        )

    def get_task_metrics(self, task_id: str) -> ToolExecutionResult | None:
        """Get metrics for a specific task."""
        return self.task_metrics.get(task_id)

    def get_dashboard_summary(self) -> Dict[str, Any]:
        """Get comprehensive dashboard summary."""
        return {
            "system_metrics": asdict(self.system_metrics),
            "task_count": len(self.task_metrics),
            "task_details": {
                tid: {
                    "tool_name": result.tool_name,
                    "status": result.status.value,
                    "execution_time": result.performance.execution_time,
                    "cost_estimate": result.performance.cost_estimate,
                    "memory_usage": result.performance.memory_usage,
                    "retry_count": result.retry_count,
                    "error": result.error_message,
                }
                for tid, result in self.task_metrics.items()
            },
            "success_rate": self.system_metrics.success_rate,
        }

    def get_performance_summary(self) -> Dict[str, float]:
        """Get quick performance summary."""
        return {
            "avg_execution_time": self.system_metrics.average_execution_time,
            "total_cost": self.system_metrics.total_cost,
            "success_rate": self.system_metrics.success_rate,
            "total_tasks": self.system_metrics.total_tasks,
        }

    def reset(self) -> None:
        """Reset all metrics."""
        self.task_metrics.clear()
        self.system_metrics = SystemMetrics()
