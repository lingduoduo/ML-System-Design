from __future__ import annotations

from .schema import RouteDecision


class LabelingModel:
    def classify(self, message: str) -> RouteDecision:
        msg = message.lower()
        has_policy_question = "refund" in msg or "policy" in msg
        has_order_workflow = "order" in msg or "status" in msg
        has_summary_request = "summarize" in msg or "summary" in msg
        is_multi_step = any(token in msg for token in [" and ", " then ", " also ", " after "]) and (
            has_policy_question or has_order_workflow
        )

        if has_policy_question and has_order_workflow:
            return RouteDecision(
                intent="tool_workflow",
                confidence=0.95,
                should_retrieve=True,
                should_plan=True,
                should_use_tools=True,
                target_agent="planner",
            )
        if has_summary_request:
            return RouteDecision(
                intent="general_qa",
                confidence=0.88,
                should_retrieve=True,
                should_plan=True,
                should_use_tools=True,
                target_agent="planner",
            )
        if has_policy_question:
            return RouteDecision(
                intent="policy_qa",
                confidence=0.93,
                should_retrieve=True,
                should_plan=is_multi_step,
                should_use_tools=False,
                target_agent="qa" if not is_multi_step else "planner",
            )
        if has_order_workflow:
            return RouteDecision(
                intent="tool_workflow",
                confidence=0.91,
                should_retrieve=True,
                should_plan=True,
                should_use_tools=True,
                target_agent="planner",
            )
        return RouteDecision(
            intent="general_qa",
            confidence=0.80,
            should_retrieve=True,
            should_plan=False,
            should_use_tools=False,
            target_agent="qa",
        )


class RouterAgent:
    def __init__(self, labeling_model: LabelingModel):
        self.labeling_model = labeling_model

    def route(self, message: str) -> RouteDecision:
        return self.labeling_model.classify(message)
