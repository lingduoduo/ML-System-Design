from __future__ import annotations

from typing import Any, Callable, Dict

from .retrieval import summarize_document
from .schema import ToolCall, ToolResult


class ToolRegistry:
    def __init__(self):
        self.tools: Dict[str, Callable[..., Any]] = {}

    def register(self, name: str, func: Callable[..., Any]) -> None:
        self.tools[name] = func

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
