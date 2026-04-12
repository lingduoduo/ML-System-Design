from __future__ import annotations

from .schema import AgentState


class ReflectionModel:
    def review(self, state: AgentState) -> str:
        draft = state.draft_response or ""

        issues = []
        if not draft:
            issues.append("Draft response is empty.")
        if state.route == "policy_qa" and not state.retrieved_policy_docs:
            issues.append("No policy documents were retrieved for policy response.")
        if "I am not sure" in draft:
            issues.append("Response is too uncertain; add retrieved evidence or ask follow-up.")

        if issues:
            return "REFINE: " + " | ".join(issues)
        return "PASS"
