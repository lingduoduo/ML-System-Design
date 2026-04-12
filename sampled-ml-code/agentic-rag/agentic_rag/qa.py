from __future__ import annotations

from typing import List

from .schema import AgentState, RetrievedDocument, ToolResult


class QAAgent:
    def generate_response(self, state: AgentState) -> str:
        user_docs = self._format_docs(state.retrieved_user_docs)
        policy_docs = self._format_docs(state.retrieved_policy_docs)
        tool_summary = self._format_tool_results(state.tool_results)
        memory_summary = str(state.long_term_memory) if state.long_term_memory else "None"

        # Use list comprehension for better performance
        response_parts = [
            f"User request: {state.user_request.message}",
            "",
            f"Route: {state.route}",
            "",
            "Memory:",
            memory_summary,
            "",
            "Retrieved User Docs:",
            user_docs,
            "",
            "Retrieved Policy Docs:",
            policy_docs,
            "",
            "Tool Results:",
            tool_summary,
            "",
            "Final Answer:",
            "Based on the available context, here is the response to the user.",
        ]
        return "\n".join(response_parts)

    @staticmethod
    def _format_docs(documents: List[RetrievedDocument]) -> str:
        if not documents:
            return "None"
        return "\n".join(doc.text for doc in documents)

    @staticmethod
    def _format_tool_results(results: List[ToolResult]) -> str:
        if not results:
            return "None"
        return "\n".join(
            f"Tool={result.tool_name}, success={result.success}, output={result.output}, error={result.error}"
            for result in results
        )
