from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
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


@dataclass
class RetrievalToolset:
    rag_chain: Any

    def _invoke_tool_query(self, query: str, instruction: str, empty_message: str, error_message: str) -> str:
        try:
            result = self.rag_chain.invoke(f"{instruction}\n\nUser request: {query}")
            result = result.strip() if isinstance(result, str) else str(result).strip()
            return result or empty_message
        except Exception as exc:
            logging.error("%s: %s", error_message, exc)
            return error_message

    def search_child_friendly_attractions(self, query: str) -> str:
        logging.info("[Tool] Searching child-friendly attractions: %s", query)
        return self._invoke_tool_query(
            query=query,
            instruction=(
                "Find child-friendly attractions relevant to the user's request. "
                "Return a short list with brief reasons."
            ),
            empty_message="No relevant child-friendly attractions found.",
            error_message="An error occurred while searching for child-friendly attractions.",
        )

    def search_nearby_restaurants(self, query: str) -> str:
        logging.info("[Tool] Searching nearby restaurants: %s", query)
        return self._invoke_tool_query(
            query=query,
            instruction=(
                "Find recommended restaurants near the mentioned attractions/areas. "
                "Return a short list with brief reasons."
            ),
            empty_message="No nearby recommended restaurants found.",
            error_message="An error occurred while searching for nearby restaurants.",
        )


@dataclass
class QueryUnderstandingEngine:
    llm: Any

    def _invoke_text_model(self, prompt: Any, variables: Dict[str, Any]) -> str:
        deps = _import_agent_dependencies()
        chain = prompt | self.llm | deps["StrOutputParser"]()
        result = chain.invoke(variables)
        return result.strip() if isinstance(result, str) else str(result).strip()

    def generate_hyde_context(self, query: str) -> str:
        deps = _import_agent_dependencies()
        prompt = deps["ChatPromptTemplate"].from_messages(
            [
                (
                    "system",
                    "You generate a short hypothetical note for retrieval support.\n"
                    "Rules:\n"
                    "- Infer the likely intent behind an ambiguous request.\n"
                    "- Keep it short, around 2 to 4 sentences.\n"
                    "- Stay generic and plausible.\n"
                    "- Do not invent exact names, addresses, prices, or schedules.\n"
                    "- Focus on the type of place, companion type, and nearby needs if implied.\n"
                    "- Return only the hypothetical note.",
                ),
                ("human", "Original query:\n{query}"),
            ]
        )
        return self._invoke_text_model(prompt, {"query": query})

    def decompose_query(self, query: str) -> str:
        deps = _import_agent_dependencies()
        prompt = deps["ChatPromptTemplate"].from_messages(
            [
                (
                    "system",
                    "You decompose user queries into short subquestions for retrieval.\n"
                    "Rules:\n"
                    "- Break the request into 2 or 3 focused retrieval questions.\n"
                    "- Keep each subquestion concise.\n"
                    "- Preserve the user's intent.\n"
                    "- Do not add unsupported details.\n"
                    "- Return only the subquestions, one per line.",
                ),
                ("human", "Original query:\n{query}"),
            ]
        )
        return self._invoke_text_model(prompt, {"query": query})

    def rewrite_ambiguous_query(self, query: str) -> str:
        deps = _import_agent_dependencies()
        hyde_context = self.generate_hyde_context(query)
        prompt = deps["ChatPromptTemplate"].from_messages(
            [
                (
                    "system",
                    "You rewrite ambiguous user queries for downstream tool and function calls.\n"
                    "Rules:\n"
                    "- Keep the rewrite concise and easy for a tool call to use.\n"
                    "- Preserve the user's intent.\n"
                    "- Do not add extra detail that the user did not provide.\n"
                    "- Use the hypothetical travel note only to disambiguate, not to add specifics.\n"
                    "- Prefer a short search-ready query that is easy for MCP-style tool calls to consume.\n"
                    "- Return only the rewritten query.",
                ),
                (
                    "human",
                    "Original query:\n{query}\n\nHypothetical travel note:\n{hyde_context}",
                ),
            ]
        )
        rewritten_query = self._invoke_text_model(
            prompt,
            {"query": query, "hyde_context": hyde_context},
        )
        return rewritten_query or query


def search_child_friendly_attractions(rag_chain: Any, query: str) -> str:
    return RetrievalToolset(rag_chain=rag_chain).search_child_friendly_attractions(query)


def search_nearby_restaurants(rag_chain: Any, query: str) -> str:
    return RetrievalToolset(rag_chain=rag_chain).search_nearby_restaurants(query)


def generate_hyde_context_for_query(llm: Any, query: str) -> str:
    return QueryUnderstandingEngine(llm=llm).generate_hyde_context(query)


def decompose_query_into_subquestions(llm: Any, query: str) -> str:
    return QueryUnderstandingEngine(llm=llm).decompose_query(query)


def rewrite_ambiguous_query_for_tools(llm: Any, query: str) -> str:
    return QueryUnderstandingEngine(llm=llm).rewrite_ambiguous_query(query)


def create_planner_prompt() -> Any:
    deps = _import_agent_dependencies()
    return deps["ChatPromptTemplate"].from_messages(
        [
            (
                "system",
                "You are a retrieval planning agent.\n"
                "You can use tools to gather information, then write a final answer.\n\n"
                "Available tools:\n"
                "1) RewriteAmbiguousQuery(query: str)\n"
                "   - Use first when the user query is ambiguous, underspecified, or too conversational for tool calls.\n"
                "   - Rewrite it into a short, search-ready query for later tool use.\n"
                "   - This tool uses a HyDE-style hypothetical travel note internally before producing the final rewrite.\n"
                "   - Do not add new details.\n"
                "2) DecomposeQueryIntoSubquestions(query: str)\n"
                "   - Use when the request contains multiple needs or should be handled as several smaller retrieval questions.\n"
                "   - Return short subquestions the planner can use in later tool calls.\n"
                "3) SearchChildFriendlyAttractions(query: str)\n"
                "   - Use to find child-friendly attractions.\n"
                "4) SearchNearbyRestaurants(query: str)\n"
                "   - Use to find restaurants near attractions/areas.\n\n"
                "Rules:\n"
                "- Decide the next step.\n"
                "- First check whether the current query is ambiguous.\n"
                "- Use three query-understanding strategies when helpful: concise query rewrite, multi-question decomposition, and HyDE-style disambiguation.\n"
                "- If the query is ambiguous, call RewriteAmbiguousQuery before other retrieval tools.\n"
                "- If the request bundles multiple needs, call DecomposeQueryIntoSubquestions before retrieval tools.\n"
                "- The rewrite must stay concise and must not become more detailed than the original request.\n"
                "- Prefer short, tool-friendly queries that are easy for MCP-style function calls to consume.\n"
                "- Output ONLY a JSON object (no markdown, no extra text).\n"
                "- If you need a tool, output:\n"
                '  {"action": "tool", "tool_name": "RewriteAmbiguousQuery", "tool_input": "..."}\n'
                "  OR\n"
                '  {"action": "tool", "tool_name": "DecomposeQueryIntoSubquestions", "tool_input": "..."}\n'
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
    query_understanding: QueryUnderstandingEngine = field(init=False)
    tools: RetrievalToolset = field(init=False)

    def __post_init__(self) -> None:
        self.query_understanding = QueryUnderstandingEngine(llm=self.llm)
        self.tools = RetrievalToolset(rag_chain=self.rag_chain)


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
        observation = ctx.query_understanding.rewrite_ambiguous_query(tool_input)
        state["active_query"] = observation
    elif tool_name == "DecomposeQueryIntoSubquestions":
        observation = ctx.query_understanding.decompose_query(tool_input)
    elif tool_name == "SearchChildFriendlyAttractions":
        observation = ctx.tools.search_child_friendly_attractions(tool_input)
    elif tool_name == "SearchNearbyRestaurants":
        observation = ctx.tools.search_nearby_restaurants(tool_input)
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
