"""Performance monitoring and dashboarding for task execution.

Provides real-time metrics tracking, performance analytics, and execution dashboards
for the agentic RAG system.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
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
        self.task_metrics: OrderedDict[str, ToolExecutionResult] = OrderedDict()
        self.system_metrics = SystemMetrics()
        self.max_history = max_history
        self._lock = threading.RLock()
        # Cache for incremental updates
        self._total_time: float = 0.0
        self._total_cost: float = 0.0
        self._total_memory: float = 0.0
        self._task_counts = {"completed": 0, "failed": 0, "skipped": 0}

    def update_task_status(self, task_id: str, result: ToolExecutionResult) -> None:
        """Update metrics for a task execution using incremental updates."""
        with self._lock:
            old_result = self.task_metrics.get(task_id)

            if old_result is not None:
                self._remove_result(old_result)
                self.task_metrics.move_to_end(task_id)

            self.task_metrics[task_id] = result
            self._add_result(result)

            while len(self.task_metrics) > self.max_history:
                _, evicted_result = self.task_metrics.popitem(last=False)
                self._remove_result(evicted_result)

            self._update_system_metrics()
    
    def _update_status_counts(self, result: ToolExecutionResult, decrement: bool = False) -> None:
        """Update status counts incrementally."""
        delta = -1 if decrement else 1
        if result.status == TaskStatus.COMPLETED:
            self._task_counts["completed"] += delta
        elif result.status == TaskStatus.FAILED:
            self._task_counts["failed"] += delta
        elif result.status == TaskStatus.SKIPPED:
            self._task_counts["skipped"] += delta

    def _add_result(self, result: ToolExecutionResult) -> None:
        self._total_time += result.performance.execution_time
        self._total_cost += result.performance.cost_estimate
        self._total_memory += result.performance.memory_usage
        self._update_status_counts(result, decrement=False)

    def _remove_result(self, result: ToolExecutionResult) -> None:
        self._total_time -= result.performance.execution_time
        self._total_cost -= result.performance.cost_estimate
        self._total_memory -= result.performance.memory_usage
        self._update_status_counts(result, decrement=True)
    
    def _update_system_metrics(self) -> None:
        """Update system metrics using cached values."""
        total = len(self.task_metrics)
        completed = self._task_counts["completed"]
        failed = self._task_counts["failed"]
        skipped = self._task_counts["skipped"]
        
        avg_time = self._total_time / max(1, total)
        success_rate = completed / max(1, total)
        
        self.system_metrics = SystemMetrics(
            total_tasks=total,
            completed_tasks=completed,
            failed_tasks=failed,
            skipped_tasks=skipped,
            average_execution_time=avg_time,
            total_cost=self._total_cost,
            total_memory_usage=self._total_memory,
            success_rate=success_rate,
        )

    def get_task_metrics(self, task_id: str) -> ToolExecutionResult | None:
        """Get metrics for a specific task."""
        with self._lock:
            return self.task_metrics.get(task_id)

    def get_dashboard_summary(self) -> Dict[str, Any]:
        """Get comprehensive dashboard summary."""
        with self._lock:
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
                }
                ,
                "success_rate": self.system_metrics.success_rate,
            }

    def get_performance_summary(self) -> Dict[str, float]:
        """Get quick performance summary."""
        with self._lock:
            return {
                "avg_execution_time": self.system_metrics.average_execution_time,
                "total_cost": self.system_metrics.total_cost,
                "success_rate": self.system_metrics.success_rate,
                "total_tasks": self.system_metrics.total_tasks,
            }

    def reset(self) -> None:
        """Reset all metrics."""
        with self._lock:
            self.task_metrics.clear()
            self.system_metrics = SystemMetrics()
            self._total_time = 0.0
            self._total_cost = 0.0
            self._total_memory = 0.0
            self._task_counts = {"completed": 0, "failed": 0, "skipped": 0}
