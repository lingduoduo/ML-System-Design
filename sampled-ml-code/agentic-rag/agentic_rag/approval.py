from __future__ import annotations

from .schema import AgentState


class HumanApprovalService:
    def request_approval(self, state: AgentState) -> bool:
        return not state.plan.requires_human_approval
