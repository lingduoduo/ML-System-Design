from __future__ import annotations

from functools import lru_cache
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, TypedDict


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


@dataclass(frozen=True)
class PromptExample:
    input_text: str
    output_text: str


@dataclass(frozen=True)
class MemoryTurn:
    user_query: str
    answer: str


@dataclass
class PromptOptimizationConfig:
    system_role: str = "You are a precise retrieval-grounded assistant."
    response_format: str = "json"
    max_memory_turns: int = 3
    use_few_shot: bool = True
    include_metadata: bool = True
    include_external_data: bool = True
    few_shot_examples: List[PromptExample] = field(
        default_factory=lambda: [
            PromptExample(
                input_text="Question: What is retrieval-augmented generation?\nContext: RAG combines retrieval with generation.",
                output_text=json.dumps(
                    {
                        "answer": "Retrieval-augmented generation combines retrieving relevant information with text generation.",
                        "grounded": True,
                        "confidence": "high",
                    },
                    ensure_ascii=True,
                ),
            ),
            PromptExample(
                input_text="Question: What is the refund policy?\nContext: The provided context does not mention refunds.",
                output_text=json.dumps(
                    {
                        "answer": "I don't know based on the provided context.",
                        "grounded": False,
                        "confidence": "low",
                    },
                    ensure_ascii=True,
                ),
            ),
        ]
    )


@dataclass
class PromptMemory:
    max_turns: int = 3
    sessions: Dict[str, List[MemoryTurn]] = field(default_factory=dict)

    def get(self, session_id: str | None = None) -> List[MemoryTurn]:
        if not session_id:
            return []
        return list(self.sessions.get(session_id, []))[-self.max_turns :]

    def append(self, user_query: str, answer: str, session_id: str | None = None) -> None:
        if not session_id:
            return
        turns = self.sessions.setdefault(session_id, [])
        turns.append(MemoryTurn(user_query=user_query, answer=answer))
        if len(turns) > self.max_turns:
            self.sessions[session_id] = turns[-self.max_turns :]


def _xml_escape(value: Any) -> str:
    text = "" if value is None else str(value)
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )


def _to_xml_block(tag: str, value: Any, indent: str = "") -> str:
    if isinstance(value, dict):
        inner = "\n".join(_to_xml_block(key, item, indent + "  ") for key, item in value.items())
        return f"{indent}<{tag}>\n{inner}\n{indent}</{tag}>"
    if isinstance(value, list):
        inner = "\n".join(_to_xml_block("item", item, indent + "  ") for item in value)
        return f"{indent}<{tag}>\n{inner}\n{indent}</{tag}>"
    return f"{indent}<{tag}>{_xml_escape(value)}</{tag}>"


def _normalize_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    normalized: Dict[str, Any] = {}
    for key, value in metadata.items():
        if value is None or key == "text":
            continue
        if isinstance(value, (str, int, float, bool)):
            normalized[key] = value
        else:
            normalized[key] = str(value)
    return normalized


def _format_memory_xml(memory_turns: List[MemoryTurn]) -> str:
    if not memory_turns:
        return "<memory />"
    items = [
        {"turn_id": index, "user_query": turn.user_query, "assistant_answer": turn.answer}
        for index, turn in enumerate(memory_turns, start=1)
    ]
    return _to_xml_block("memory", items)


def _format_few_shot_examples(examples: List[PromptExample], response_format: str) -> str:
    if not examples:
        return "<few_shot_examples />"
    rendered: List[str] = []
    for index, example in enumerate(examples, start=1):
        rendered.append(
            _to_xml_block(
                "example",
                {
                    "id": index,
                    "input": example.input_text,
                    "ideal_output": example.output_text,
                    "output_format": response_format,
                },
            )
        )
    return "<few_shot_examples>\n" + "\n".join(rendered) + "\n</few_shot_examples>"


def _build_structured_response_instruction(response_format: str) -> str:
    if response_format.lower() == "xml":
        return (
            "Return XML with this structure only:\n"
            "<response><answer>...</answer><grounded>true|false</grounded>"
            "<confidence>high|medium|low</confidence><citations><item>Doc ids or metadata</item></citations></response>"
        )
    return (
        'Return JSON only with keys: "answer", "grounded", "confidence", "citations". '
        '"citations" must be a list of doc ids or short metadata references.'
    )


def _build_prompt_body(
    query: str,
    context: str,
    *,
    prompt_config: PromptOptimizationConfig,
    metadata: Dict[str, Any] | None = None,
    external_data: Dict[str, Any] | None = None,
    memory_turns: List[MemoryTurn] | None = None,
) -> str:
    memory_xml = _format_memory_xml(memory_turns or [])
    metadata_xml = _to_xml_block("request_metadata", metadata or {})
    external_xml = _to_xml_block("external_data", external_data or {})
    examples_xml = (
        _format_few_shot_examples(prompt_config.few_shot_examples, prompt_config.response_format)
        if prompt_config.use_few_shot
        else "<few_shot_examples />"
    )
    return (
        f"{prompt_config.system_role}\n"
        "Reason through the task internally before answering, but do not reveal hidden reasoning.\n"
        "Ground every claim in the supplied context or metadata.\n"
        "If the context is insufficient, say so clearly.\n"
        f"{_build_structured_response_instruction(prompt_config.response_format)}\n\n"
        "<task>\n"
        "Use retrieved evidence, optional external data, and short-term memory to answer the user query.\n"
        "</task>\n"
        f"{examples_xml}\n"
        f"{memory_xml}\n"
        f"{metadata_xml}\n"
        f"{external_xml}\n"
        f"{_to_xml_block('query', query)}\n"
        f"{_to_xml_block('retrieved_context', context)}\n"
        "<quality_checks>\n"
        "  <item>Prefer grounded answers over fluent guesses.</item>\n"
        "  <item>Use citations when possible.</item>\n"
        "  <item>Keep the final answer concise and directly useful.</item>\n"
        "</quality_checks>\n"
    )


def build_prompt(
    query: str,
    retrieved_chunks: List[Tuple[float, Dict]],
    *,
    prompt_config: PromptOptimizationConfig | None = None,
    request_metadata: Dict[str, Any] | None = None,
    external_data: Dict[str, Any] | None = None,
    memory_turns: List[MemoryTurn] | None = None,
) -> str:
    prompt_config = prompt_config or PromptOptimizationConfig()
    context_lines: List[str] = []
    chunk_metadata: Dict[str, Any] = {}
    for index, (score, metadata) in enumerate(retrieved_chunks, start=1):
        text = str(metadata.get("text", "")).strip()
        if not text:
            continue
        doc_ref = metadata.get("id") or metadata.get("source") or f"chunk_{index}"
        context_lines.append(f"[Doc {index} | ref={doc_ref} | score={score:.4f}]\n{text}")
        if prompt_config.include_metadata:
            chunk_metadata[f"doc_{index}"] = _normalize_metadata(metadata)

    merged_metadata = dict(request_metadata or {})
    if prompt_config.include_metadata and chunk_metadata:
        merged_metadata["retrieved_chunk_metadata"] = chunk_metadata

    return _build_prompt_body(
        query=query,
        context="\n\n".join(context_lines),
        prompt_config=prompt_config,
        metadata=merged_metadata if merged_metadata else None,
        external_data=external_data if prompt_config.include_external_data else None,
        memory_turns=memory_turns,
    )


@dataclass
class LLMTwin:
    model: Any
    prompt_config: PromptOptimizationConfig = field(default_factory=PromptOptimizationConfig)
    memory: PromptMemory = field(default_factory=PromptMemory)

    def __post_init__(self) -> None:
        self.memory.max_turns = self.prompt_config.max_memory_turns

    def answer(
        self,
        query: str,
        retrieved_chunks: List[Tuple[float, Dict]],
        *,
        session_id: str | None = None,
        request_metadata: Dict[str, Any] | None = None,
        external_data: Dict[str, Any] | None = None,
    ) -> str:
        prompt = build_prompt(
            query,
            retrieved_chunks,
            prompt_config=self.prompt_config,
            request_metadata=request_metadata,
            external_data=external_data,
            memory_turns=self.memory.get(session_id),
        )
        if hasattr(self.model, "generate"):
            result = self.model.generate(prompt)
        elif hasattr(self.model, "invoke"):
            result = self.model.invoke(prompt)
        else:
            raise TypeError("LLMTwin model must implement either `generate` or `invoke`.")
        result_text = result.content if hasattr(result, "content") else str(result)
        self.memory.append(query, result_text, session_id=session_id)
        return result_text


@lru_cache(maxsize=1)
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


@dataclass
class QueryEngineResult:
    answer: str
    documents: List[Any]
    context: str
    prompt: str = ""


@dataclass
class CrossEncoderReranker:
    model: Any
    default_top_n: int = 3

    def rerank(self, query: str, docs: List[Any], top_n: int | None = None) -> List[Any]:
        if not docs:
            return []

        pairs = [(query, doc.page_content or "") for doc in docs]
        scores = self.model.predict(pairs)
        scored_docs: List[Tuple[Any, float]] = []
        for doc, score in zip(docs, scores):
            metadata = dict(getattr(doc, "metadata", {}) or {})
            metadata["rerank_score"] = float(score)
            scored_docs.append((type(doc)(page_content=doc.page_content, metadata=metadata), float(score)))

        scored_docs.sort(key=lambda item: item[1], reverse=True)
        limit = top_n if top_n is not None else self.default_top_n
        return [doc for doc, _ in scored_docs[:limit]]


def build_context(docs: List[Any], max_chars: int = 6000) -> str:
    blocks: List[str] = []
    total = 0
    for index, doc in enumerate(docs, start=1):
        text = (getattr(doc, "page_content", "") or "").strip()
        if not text:
            continue
        metadata = _normalize_metadata(dict(getattr(doc, "metadata", {}) or {}))
        metadata_summary = (
            ", ".join(f"{key}={value}" for key, value in metadata.items())
            if metadata
            else "no_metadata"
        )
        block = f"[Doc {index} | {metadata_summary}]\n{text}\n"
        if total + len(block) > max_chars:
            break
        blocks.append(block)
        total += len(block)
    return "\n".join(blocks)


@dataclass
class RAGQueryEngine:
    retriever: Any
    llm: Any
    reranker: Any | None = None
    default_retrieve_top_k: int = 5
    max_context_chars: int = 6000
    prompt_config: PromptOptimizationConfig = field(default_factory=PromptOptimizationConfig)
    memory: PromptMemory = field(default_factory=PromptMemory)

    def __post_init__(self) -> None:
        self.memory.max_turns = self.prompt_config.max_memory_turns

    def _configure_top_k(self, top_k: int) -> None:
        if top_k <= 0:
            return
        for attr in ("k", "vector_top_k", "bm25_top_k", "runtime_top_k"):
            if hasattr(self.retriever, attr):
                try:
                    setattr(self.retriever, attr, top_k)
                except Exception:
                    pass

    def _generate_answer(
        self,
        query: str,
        context: str,
        *,
        request_metadata: Dict[str, Any] | None = None,
        external_data: Dict[str, Any] | None = None,
        memory_turns: List[MemoryTurn] | None = None,
    ) -> Tuple[str, str]:
        prompt = _build_prompt_body(
            query=query,
            context=context,
            prompt_config=self.prompt_config,
            metadata=request_metadata if self.prompt_config.include_metadata else None,
            external_data=external_data if self.prompt_config.include_external_data else None,
            memory_turns=memory_turns,
        )
        if hasattr(self.llm, "invoke"):
            result = self.llm.invoke(prompt)
            return (result.content if hasattr(result, "content") else str(result), prompt)
        if hasattr(self.llm, "generate"):
            return (self.llm.generate(prompt), prompt)
        raise TypeError("Query engine LLM must implement either `invoke` or `generate`.")

    def run(
        self,
        query: str,
        retrieve_top_k: int | None = None,
        rerank_top_n: int | None = None,
        *,
        session_id: str | None = None,
        request_metadata: Dict[str, Any] | None = None,
        external_data: Dict[str, Any] | None = None,
    ) -> QueryEngineResult:
        top_k = retrieve_top_k if retrieve_top_k is not None else self.default_retrieve_top_k
        self._configure_top_k(top_k)
        docs = self.retriever.invoke(query)
        if top_k > 0:
            docs = docs[:top_k]
        if self.reranker is not None:
            docs = self.reranker.rerank(query, docs, top_n=rerank_top_n or top_k)
        context = build_context(docs, max_chars=self.max_context_chars)
        merged_metadata = dict(request_metadata or {})
        if self.prompt_config.include_metadata and docs:
            merged_metadata["document_metadata"] = {
                f"doc_{index}": _normalize_metadata(dict(getattr(doc, "metadata", {}) or {}))
                for index, doc in enumerate(docs, start=1)
            }
        answer, prompt = self._generate_answer(
            query,
            context,
            request_metadata=merged_metadata if merged_metadata else None,
            external_data=external_data,
            memory_turns=self.memory.get(session_id),
        )
        self.memory.append(query, answer, session_id=session_id)
        return QueryEngineResult(answer=answer, documents=docs, context=context, prompt=prompt)


def build_rag_chain(retriever: Any, llm: Any) -> Any:
    deps = _import_agent_dependencies()
    prompt = deps["ChatPromptTemplate"].from_messages(
        [
            (
                "system",
                "You answer questions using the provided context.\n"
                "Reason internally, but do not reveal hidden reasoning.\n"
                "Cite the document ids when possible.\n"
                "If the context is insufficient, say so clearly.\n"
                'Return JSON only with keys: "answer", "grounded", "confidence", "citations".',
            ),
            (
                "human",
                "<examples>\n"
                "<example><question>What is RAG?</question><context>[Doc 1] RAG combines retrieval and generation.</context>"
                '<ideal>{"answer":"RAG combines retrieval with generation.","grounded":true,"confidence":"high","citations":["Doc 1"]}</ideal></example>\n'
                "</examples>\n"
                "<query>{question}</query>\n"
                "<retrieved_context>{context}</retrieved_context>\n"
                "<instruction>Answer using only the retrieved context.</instruction>",
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

    def _few_shot_block(self, examples: List[Tuple[str, str]]) -> str:
        blocks = []
        for index, (example_input, example_output) in enumerate(examples, start=1):
            blocks.append(
                _to_xml_block(
                    "example",
                    {
                        "id": index,
                        "input": example_input,
                        "output": example_output,
                    },
                )
            )
        return "<few_shot_examples>\n" + "\n".join(blocks) + "\n</few_shot_examples>"

    def generate_hyde_context(self, query: str) -> str:
        deps = _import_agent_dependencies()
        examples = self._few_shot_block(
            [
                (
                    "Need a nice place for kids and somewhere to eat after.",
                    "The user likely wants family-friendly attractions suitable for children and nearby dining options for a follow-up meal.",
                ),
            ]
        )
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
                ("human", f"{examples}\n<original_query>{{query}}</original_query>"),
            ]
        )
        return self._invoke_text_model(prompt, {"query": query})

    def decompose_query(self, query: str) -> str:
        deps = _import_agent_dependencies()
        examples = self._few_shot_block(
            [
                (
                    "Plan a kid-friendly day and also find lunch nearby.",
                    "Find child-friendly attractions for a daytime visit.\nFind lunch options near the likely attraction area.",
                ),
            ]
        )
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
                ("human", f"{examples}\n<original_query>{{query}}</original_query>"),
            ]
        )
        return self._invoke_text_model(prompt, {"query": query})

    def rewrite_ambiguous_query(self, query: str) -> str:
        deps = _import_agent_dependencies()
        hyde_context = self.generate_hyde_context(query)
        examples = self._few_shot_block(
            [
                (
                    "Somewhere fun for kids and food too.",
                    "child-friendly attractions with nearby restaurants",
                ),
            ]
        )
        prompt = deps["ChatPromptTemplate"].from_messages(
            [
                (
                    "system",
                    "You rewrite ambiguous user queries for downstream tool and function calls.\n"
                    "Rules:\n"
                    "- Keep the rewrite concise and easy for a tool call to use.\n"
                    "- Preserve the user's intent.\n"
                    "- Do not add extra detail that the user did not provide.\n"
                    "- Use the hypothetical note only to disambiguate, not to add specifics.\n"
                    "- Prefer a short search-ready query that is easy for MCP-style tool calls to consume.\n"
                    "- Return only the rewritten query.",
                ),
                (
                    "human",
                    f"{examples}\n<original_query>{{query}}</original_query>\n<hypothetical_note>{{hyde_context}}</hypothetical_note>",
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
                "   - This tool uses a HyDE-style hypothetical note internally before producing the final rewrite.\n"
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
                "- Think step-by-step internally, but only output the final JSON action.\n"
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
                "- Use the latest rewritten query if one exists.\n"
                "Few-shot examples:\n"
                '{"action":"tool","tool_name":"RewriteAmbiguousQuery","tool_input":"somewhere fun for kids and food too"}\n'
                '{"action":"tool","tool_name":"SearchChildFriendlyAttractions","tool_input":"child-friendly attractions with nearby restaurants"}\n'
                '{"action":"final","answer":"Here are a few family-friendly options, followed by nearby food suggestions."}',
            ),
            (
                "human",
                "<planner_state>\n"
                "<user_request>{user_query}</user_request>\n"
                "<active_query>{active_query}</active_query>\n"
                "<scratchpad>{scratchpad}</scratchpad>\n"
                "</planner_state>\n",
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
