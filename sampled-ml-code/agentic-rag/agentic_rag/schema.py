from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional
from enum import Enum


@dataclass(slots=True)
class UserRequest:
    user_id: str
    channel: str
    message: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RetrievedDocument:
    doc_id: str
    source: str
    text: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ToolCall:
    tool_name: str
    arguments: Dict[str, Any]


@dataclass(slots=True)
class ToolResult:
    tool_name: str
    success: bool
    output: Any
    error: Optional[str] = None


@dataclass(slots=True)
class PlanStep:
    step_id: str
    description: str
    needs_tool: bool = False
    tool_call: Optional[ToolCall] = None


@dataclass(slots=True)
class Plan:
    workflow: str
    steps: List[PlanStep]
    requires_human_approval: bool = False


@dataclass(slots=True)
class RouteDecision:
    intent: str
    confidence: float
    should_retrieve: bool = True
    should_plan: bool = False
    should_use_tools: bool = False
    target_agent: str = "qa"


@dataclass(slots=True)
class AgentState:
    request_id: str
    user_request: UserRequest
    route: Optional[str] = None
    short_term_memory: List[Dict[str, str]] = field(default_factory=list)
    long_term_memory: Dict[str, Any] = field(default_factory=dict)
    retrieved_user_docs: List[RetrievedDocument] = field(default_factory=list)
    retrieved_policy_docs: List[RetrievedDocument] = field(default_factory=list)
    plan: Optional[Plan] = None
    tool_results: List[ToolResult] = field(default_factory=list)
    draft_response: Optional[str] = None
    final_response: Optional[str] = None
    trace: List[Dict[str, Any]] = field(default_factory=list)


class TaskStatus(Enum):
    """Task execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    RETRYING = "retrying"


class TaskType(Enum):
    """Task execution type."""
    RETRIEVAL = "retrieval"
    ROUTING = "routing"
    PLANNING = "planning"
    TOOL_EXECUTION = "tool_execution"
    RESPONSE_GENERATION = "response_generation"


@dataclass(slots=True)
class PerformanceMetrics:
    """Task performance metrics."""
    execution_time: float
    cost_estimate: float = 0.0
    memory_usage: float = 0.0
    success_rate: float = 1.0


@dataclass(slots=True)
class TaskNode:
    """Execution task with dependencies."""
    task_id: str
    task_type: TaskType
    description: str
    params: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    priority: int = 1
    max_retries: int = 2
    timeout: float = 30.0
    is_critical: bool = False
    status: TaskStatus = TaskStatus.PENDING
    retry_count: int = 0
    result: Optional[Any] = None


@dataclass(slots=True)
class ToolExecutionResult:
    """Tool execution result with metrics."""
    tool_name: str
    status: TaskStatus
    result: Any
    performance: PerformanceMetrics
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    optimization_suggestions: List[str] = field(default_factory=list)
    retry_count: int = 0
