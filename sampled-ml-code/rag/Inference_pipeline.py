from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, TypedDict


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def build_prompt(query: str, retrieved_chunks: List[Tuple[float, Dict]]) -> str:
    context = "\n".join(metadata["text"] for _, metadata in retrieved_chunks)
    return (
        "You are a helpful assistant.\n\n"
        f"Question:\n{query}\n\n"
        f"Context:\n{context}\n\n"
        "Answer:\n"
    )


@dataclass
class LLMTwin:
    model: Any

    def answer(self, query: str, retrieved_chunks: List[Tuple[float, Dict]]) -> str:
        prompt = build_prompt(query, retrieved_chunks)
        if hasattr(self.model, "generate"):
            return self.model.generate(prompt)
        if hasattr(self.model, "invoke"):
            return self.model.invoke(prompt)
        raise TypeError("LLMTwin model must implement either `generate` or `invoke`.")


def _import_agent_dependencies() -> Dict[str, Any]:
    try:
        from langchain_core.messages import AIMessage, HumanMessage
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.runnables import RunnableLambda, RunnablePassthrough
        from langgraph.graph import END, StateGraph
    except ImportError as exc:
        raise ImportError(
            "Inference multi-step features require `langchain-core` and `langgraph`."
        ) from exc

    return {
        "AIMessage": AIMessage,
        "ChatPromptTemplate": ChatPromptTemplate,
        "END": END,
        "HumanMessage": HumanMessage,
        "RunnableLambda": RunnableLambda,
        "RunnablePassthrough": RunnablePassthrough,
        "StateGraph": StateGraph,
        "StrOutputParser": StrOutputParser,
    }


def _format_docs(docs: List[Any]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


def build_rag_chain(retriever: Any, llm: Any) -> Any:
    deps = _import_agent_dependencies()
    prompt = deps["ChatPromptTemplate"].from_messages(
        [
            (
                "system",
                "You answer questions using the provided context. "
                "If the context is insufficient, say so clearly.",
            ),
            (
                "human",
                "Question:\n{question}\n\nContext:\n{context}\n\nAnswer:",
            ),
        ]
    )
    return (
        {
            "context": retriever | deps["RunnableLambda"](_format_docs),
            "question": deps["RunnablePassthrough"](),
        }
        | prompt
        | llm
        | deps["StrOutputParser"]()
    )


def search_child_friendly_attractions(rag_chain: Any, query: str) -> str:
    logging.info("[Tool] Searching child-friendly attractions: %s", query)
    try:
        tool_query = (
            "Find child-friendly attractions relevant to the user's request. "
            "Return a short list with brief reasons.\n\n"
            f"User request: {query}"
        )
        result = rag_chain.invoke(tool_query)
        result = result.strip() if isinstance(result, str) else str(result).strip()
        return result or "No relevant child-friendly attractions found."
    except Exception as exc:
        logging.error("[Tool] Error in child-friendly attractions search: %s", exc)
        return "An error occurred while searching for child-friendly attractions."


def search_nearby_restaurants(rag_chain: Any, query: str) -> str:
    logging.info("[Tool] Searching nearby restaurants: %s", query)
    try:
        tool_query = (
            "Find recommended restaurants near the mentioned attractions/areas. "
            "Return a short list with brief reasons.\n\n"
            f"User request: {query}"
        )
        result = rag_chain.invoke(tool_query)
        result = result.strip() if isinstance(result, str) else str(result).strip()
        return result or "No nearby recommended restaurants found."
    except Exception as exc:
        logging.error("[Tool] Error in nearby restaurants search: %s", exc)
        return "An error occurred while searching for nearby restaurants."


class AgentState(TypedDict, total=False):
    messages: List[Any]
    scratchpad: str
    active_query: str
    last_action: Optional[Dict[str, Any]]
    final_answer: Optional[str]


def rewrite_ambiguous_query_for_tools(llm: Any, query: str) -> str:
    deps = _import_agent_dependencies()
    prompt = deps["ChatPromptTemplate"].from_messages(
        [
            (
                "system",
                "You rewrite ambiguous travel queries for downstream tool and function calls.\n"
                "Rules:\n"
                "- Keep the rewrite concise and easy for a tool call to use.\n"
                "- Preserve the user's intent.\n"
                "- Do not add extra detail that the user did not provide.\n"
                "- Resolve vague wording into a cleaner search-ready query when possible.\n"
                "- Return only the rewritten query.",
            ),
            ("human", "Original query:\n{query}"),
        ]
    )
    chain = prompt | llm | deps["StrOutputParser"]()
    rewritten_query = chain.invoke({"query": query}).strip()
    return rewritten_query or query


def create_planner_prompt() -> Any:
    deps = _import_agent_dependencies()
    return deps["ChatPromptTemplate"].from_messages(
        [
            (
                "system",
                "You are a travel assistant agent.\n"
                "You can use tools to gather information, then write a final answer.\n\n"
                "Available tools:\n"
                "1) RewriteAmbiguousQuery(query: str)\n"
                "   - Use first when the user query is ambiguous, underspecified, or too conversational for tool calls.\n"
                "   - Rewrite it into a short, search-ready query for later tool use.\n"
                "   - Do not add new details.\n"
                "2) SearchChildFriendlyAttractions(query: str)\n"
                "   - Use to find child-friendly attractions.\n"
                "3) SearchNearbyRestaurants(query: str)\n"
                "   - Use to find restaurants near attractions/areas.\n\n"
                "Rules:\n"
                "- Decide the next step.\n"
                "- First check whether the current query is ambiguous.\n"
                "- If it is ambiguous, call RewriteAmbiguousQuery before other tools.\n"
                "- The rewrite must stay concise and must not become more detailed than the original request.\n"
                "- Prefer short, tool-friendly queries that are easy for MCP-style function calls to consume.\n"
                "- Output ONLY a JSON object (no markdown, no extra text).\n"
                "- If you need a tool, output:\n"
                '  {"action": "tool", "tool_name": "RewriteAmbiguousQuery", "tool_input": "..."}\n'
                "  OR\n"
                '  {"action": "tool", "tool_name": "SearchChildFriendlyAttractions", "tool_input": "..."}\n'
                "  OR\n"
                '  {"action": "tool", "tool_name": "SearchNearbyRestaurants", "tool_input": "..."}\n'
                "- If you are ready to answer, output:\n"
                '  {"action": "final", "answer": "..."}\n'
                "- Use the latest rewritten query if one exists.",
            ),
            (
                "human",
                "User request:\n{user_query}\n\nCurrent working query:\n{active_query}\n\nScratchpad so far:\n{scratchpad}\n",
            ),
        ]
    )


def _safe_parse_json(text: str) -> Dict[str, Any]:
    text = text.strip()
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        text = match.group(0).strip()
    return json.loads(text)


@dataclass
class GraphContext:
    llm: Any
    rag_chain: Any


def planner_node(state: AgentState, ctx: GraphContext) -> AgentState:
    deps = _import_agent_dependencies()
    user_query = ""
    for message in state.get("messages", []):
        if isinstance(message, deps["HumanMessage"]):
            user_query = message.content
            break

    active_query = state.get("active_query", user_query).strip() or user_query
    scratchpad = state.get("scratchpad", "").strip() or "(empty)"
    chain = create_planner_prompt() | ctx.llm | deps["StrOutputParser"]()
    raw = chain.invoke(
        {
            "user_query": user_query,
            "active_query": active_query,
            "scratchpad": scratchpad,
        }
    )
    logging.info("[Planner raw]\n%s", raw)

    try:
        action = _safe_parse_json(raw)
    except Exception as exc:
        logging.error("[Planner] Failed to parse JSON. Error: %s", exc)
        action = {
            "action": "final",
            "answer": "Sorry, I couldn't decide the next step due to a formatting issue.",
        }

    state["last_action"] = action
    state.setdefault("messages", []).append(deps["AIMessage"](content=raw))
    return state


def tool_node(state: AgentState, ctx: GraphContext) -> AgentState:
    deps = _import_agent_dependencies()
    action = state.get("last_action") or {}
    tool_name = action.get("tool_name")
    tool_input = action.get("tool_input", "")
    active_query = state.get("active_query", tool_input).strip()

    if not tool_name:
        return state

    if not tool_input:
        tool_input = active_query

    if tool_name == "RewriteAmbiguousQuery":
        observation = rewrite_ambiguous_query_for_tools(ctx.llm, tool_input)
        state["active_query"] = observation
    elif tool_name == "SearchChildFriendlyAttractions":
        observation = search_child_friendly_attractions(ctx.rag_chain, tool_input)
    elif tool_name == "SearchNearbyRestaurants":
        observation = search_nearby_restaurants(ctx.rag_chain, tool_input)
    else:
        observation = f"Unknown tool: {tool_name}"

    scratchpad = state.get("scratchpad", "")
    scratchpad += (
        f"\n\n[Action] {tool_name}\n"
        f"[Input] {tool_input}\n"
        f"[Observation]\n{observation}\n"
    )
    state["scratchpad"] = scratchpad.strip()
    state.setdefault("messages", []).append(
        deps["AIMessage"](content=f"TOOL_OBSERVATION({tool_name}):\n{observation}")
    )
    return state


def finalizer_node(state: AgentState, ctx: GraphContext) -> AgentState:
    del ctx
    action = state.get("last_action") or {}
    state["final_answer"] = action.get("answer", "")
    return state


def route_after_planner(state: AgentState) -> str:
    action = state.get("last_action") or {}
    if action.get("action") == "tool":
        return "tool"
    return "final"


def build_agent_graph(ctx: GraphContext) -> Any:
    deps = _import_agent_dependencies()
    graph = deps["StateGraph"](AgentState)
    graph.add_node("planner", lambda state: planner_node(state, ctx))
    graph.add_node("tool", lambda state: tool_node(state, ctx))
    graph.add_node("final", lambda state: finalizer_node(state, ctx))
    graph.set_entry_point("planner")
    graph.add_conditional_edges("planner", route_after_planner, {"tool": "tool", "final": "final"})
    graph.add_edge("tool", "planner")
    graph.add_edge("final", deps["END"])
    return graph.compile()


def run_multi_step_search(agent_app: Any, query: str, max_steps: int = 6) -> str:
    deps = _import_agent_dependencies()
    initial_state: AgentState = {
        "messages": [deps["HumanMessage"](content=query)],
        "scratchpad": "",
        "active_query": query,
    }
    output = agent_app.invoke(initial_state, config={"recursion_limit": max_steps})
    answer = output.get("final_answer") or ""
    return answer.strip()
