from __future__ import annotations

from typing import Optional

from .retrieval import ORDER_ID_PATTERN
from .schema import ToolCall


class ToolSelectionModel:
    def choose_tool(self, message: str) -> Optional[ToolCall]:
        msg = message.lower()

        if "order" in msg and "status" in msg:
            order_id = self._extract_order_id(message)
            return ToolCall(tool_name="search_orders", arguments={"order_id": order_id})

        if "ticket" in msg or "issue" in msg:
            return ToolCall(
                tool_name="create_ticket",
                arguments={"issue": message, "severity": "medium"},
            )

        if "summarize" in msg or "summary" in msg:
            if "user" in msg or "docs" in msg:
                return ToolCall(tool_name="summarize_user_docs", arguments={})
            if "policy" in msg:
                return ToolCall(tool_name="summarize_policy_docs", arguments={})

        return None

    def _extract_order_id(self, message: str) -> str:
        match = ORDER_ID_PATTERN.search(message)
        if match:
            return match.group(0).upper()
        return "ORD-001"
