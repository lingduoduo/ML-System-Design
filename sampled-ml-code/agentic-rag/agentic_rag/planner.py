from __future__ import annotations

from typing import List

from .schema import AgentState, Plan, PlanStep
from .tool_selection import ToolSelectionModel


class ReturnPlannerAgent:
    def __init__(self, tool_selector: ToolSelectionModel):
        self.tool_selector = tool_selector

    def create_plan(self, state: AgentState) -> Plan:
        route = state.route or "general_qa"
        steps: List[PlanStep] = [
            PlanStep(
                step_id="step-1",
                description="Review user request, memory, and retrieved context",
            )
        ]

        tool_call = self.tool_selector.choose_tool(state.user_request.message)
        if tool_call:
            steps.append(
                PlanStep(
                    step_id="step-2",
                    description=f"Execute tool {tool_call.tool_name}",
                    needs_tool=True,
                    tool_call=tool_call,
                )
            )

        requires_human = "refund over $500" in state.user_request.message.lower()
        steps.append(
            PlanStep(
                step_id="step-3",
                description="Draft final answer using context, tool results, and policy",
            )
        )

        return Plan(
            workflow=route,
            steps=steps,
            requires_human_approval=requires_human,
        )
