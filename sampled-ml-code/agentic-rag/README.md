# Agentic RAG System

Agentic RAG prototype for customer-support workflows. The system combines routing, retrieval, planning, tool execution, reflection, memory, and monitoring in a modular package. It supports simple RAG-style question answering, tool-driven workflows, retry-aware task execution, and an optional PyTorch-based tool selector with Gumbel-Softmax exploration.

## Highlights

- Modular architecture: router, Q&A agent, planner, retrieval, memory, reflection, monitored tools, and workflow orchestration
- Fast paths for simple queries: lightweight routing can skip planning when a direct answer is enough
- Task-based execution: tool steps are executed through `TaskNode` dispatch with retries and early-exit handling
- Retrieval options: lexical fallback by default, plus optional LangChain + FAISS support
- Monitoring and observability: per-tool metrics, dashboard summaries, traces, and workflow-level performance stats
- Optional learned tool selection: PyTorch selector training, checkpoint loading, and Gumbel-Softmax sampling

## Built-in Tools

- `search_orders(order_id)` for order status lookup
- `create_ticket(issue, severity)` for support ticket creation
- `summarize_user_docs()` for user-document summarization
- `summarize_policy_docs()` for policy summarization

## Project Layout

```text
.
├── agentic_rag
│   ├── approval.py          # Human approval service for risky operations
│   ├── builder.py           # Workflow construction and dependency injection
│   ├── config.py            # Configuration management and feature flags
│   ├── gateway.py           # Request validation and preprocessing
│   ├── memory.py            # Conversation memory with async support
│   ├── monitoring.py        # Performance monitoring and dashboard
│   ├── observability.py     # Logging and tracing utilities
│   ├── planner.py           # Task planning and execution steps
│   ├── qa.py                # Question-answering agent
│   ├── reflection.py        # Response validation and refinement
│   ├── retrieval.py         # Document processing and retrieval
│   ├── router.py            # Intelligent request routing
│   ├── schema.py            # Data models, task nodes, and metrics
│   ├── tool_selection.py    # Tool selection policies
│   ├── tools.py             # Tool registry and BaseToolAgent
│   └── workflow.py          # Main orchestration with task management
├── data
│   ├── policy_docs.txt      # Policy documentation
│   └── user_docs.txt        # User documentation
├── main.py                  # CLI interface and demo
├── .env.example             # Environment variables template
└── README.md
```

## Setup

Run the demo:

```bash
python3 main.py
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Optional LangChain + vector-retrieval extras:

```bash
pip install python-dotenv langchain langchain-openai langchain-community faiss-cpu
```

Optional environment variables:

```bash
export OPENAI_API_KEY=your_key_here
export OPENAI_BASE_URL=https://your-compatible-endpoint.example.com
export ENABLE_GUMBEL_TOOL_SELECTION=true
export GUMBEL_TOOL_TEMPERATURE=0.5
export GUMBEL_TOOL_HARD=true
export TOOL_SELECTOR_MODEL_PATH=artifacts/tool_selector.pt
```

## Architecture

### Router model

- File: `agentic_rag/router.py`
- Purpose: lightweight control policy for deciding whether to invoke planning, retrieval, direct answering, tool usage, or another agent path
- Produces a structured `RouteDecision`
- In this repo, it decides whether a request should retrieve, plan, use tools, or go straight to Q&A
- Design options include a prompt-based router, a small intent classifier, or hybrid rules plus model
- Main cost benefit: simple requests skip the more expensive planner

### Q&A agent

- File: `agentic_rag/qa.py`
- Purpose: handle simple queries without requiring a multi-step tool workflow
- Model pattern: LLM with RAG
- Generates the final answer from memory, retrieved documents, and tool results

### Return planner agent

- File: `agentic_rag/planner.py`
- Purpose: decompose the task, decide the action sequence, and choose between direct response, tool usage, or multi-step execution
- Typical inputs: user request, order info, and policy context
- Typical outputs: structured plans or function calls such as `check_eligibility` and `initiate_return`
- Common implementation patterns: long-context LLM with function calling, optionally fine-tuned for planning traces
- Reasoning styles can include CoT, ToT, ReAct, or structured JSON planning
- In this repo, it builds the structured plan consumed by the workflow

### Tool selection model

- File: `agentic_rag/tool_selection.py`
- Purpose: choose the best tool or tool sequence for the current request
- Supports deterministic heuristic scoring, Gumbel-Softmax sampling, and optional trained PyTorch selector checkpoints
- Includes `train_model()`, `save_model()`, and `load_model()` helpers
- Design options include ranking models, multi-label classifiers, or LLM function-calling policies
- Useful features include query embeddings, tool-description embeddings, current plan step, and past success rate

Gumbel-Softmax is used as the exploration mechanism for tool choice:

1. Add Gumbel noise to the tool logits.
2. Apply softmax to the noisy logits after scaling by temperature `tau`.
3. Use `tau` to control exploration vs. exploitation.

Higher `tau` makes the distribution more uniform and exploratory. Lower `tau` makes the distribution sharper and more deterministic.

### Retrieval model

- File: `agentic_rag/retrieval.py`
- Purpose: fetch relevant grounded knowledge for the active request
- Loads, chunks, tokenizes, and retrieves from the local corpora
- Supports lexical fallback retrieval by default and optional vector retrieval
- Common components include query rewrite, dense retrieval, sparse retrieval, and reranking
- Common architectures include embeddings plus a vector database, hybrid BM25 plus dense retrieval, and a cross-encoder reranker

### Reflection model

- File: `agentic_rag/reflection.py`
- Purpose: evaluate intermediate or final results, detect hallucination or execution failure, and decide whether to revise
- Reviews the draft response for obvious issues before finalization
- Common forms include LLM-as-a-judge, groundedness or quality classifiers, and rule-based validators

### Memory model

- File: `agentic_rag/memory.py`
- Purpose: decide what to store, retrieve, summarize, promote, or forget across interactions
- Uses a hierarchical memory design:
  - short-term conversation turns are kept in a bounded deque for immediate context
  - vector memories are stored in a FAISS-backed short-term and long-term hierarchy
  - a `MemoryRetriever` performs semantic lookup over that hierarchy
- `MemoryTransformer` is the organizing layer for memory ingestion and querying
- `MemoryTransformer.process(content)` computes an importance score and routes content into short-term or long-term storage
- Higher-importance memories are promoted to long-term storage; lower-importance memories remain short-term unless promoted later
- Stores short-term conversation history, long-term key-value memory, and semantic vector memories in one module
- Possible components include summarization, memory retrieval, salience ranking, and memory promotion policies

### Orchestration

- File: `agentic_rag/workflow.py`
- Coordinates gateway checks, memory loading, routing, retrieval, planning, tool execution, reflection, caching, and final response assembly
- In a multi-agent setting, this layer decides which agent to invoke, when to transfer control, and when to merge outputs
- Typical agent roles can include planner, retrieval, code, review, and domain-specific agents

### Modeling strategy: predefined vs dynamically orchestrated agents

- Predefined agents use a fixed workflow graph with deterministic nodes; they are easier to debug and usually lower latency
- Dynamically orchestrated agents let an LLM decide the next node at runtime; they are more flexible but higher latency and less predictable
- This repo currently leans toward the predefined-agent approach

### Tool agents and monitoring

- Files: `agentic_rag/tools.py`, `agentic_rag/monitoring.py`
- Built-in tools run through monitored tool-agent wrappers
- Each tool execution can capture execution time, success rate, cost estimate, and memory estimate
- The workflow aggregates these metrics into a monitoring dashboard

## Runtime Behavior

- Standard mode uses local fallback retrieval over the text files in `data/`
- Enhanced mode uses LangChain when installed and configured
- Response caching deduplicates repeated requests in memory
- Async mode overlaps memory and retrieval work where useful
- Memory writes add both plain conversation turns and vectorized memories
- Semantic memory retrieval uses `MemoryRetriever` over `MemoryStorage`
- Short-term versus long-term vector placement is determined by memory importance

## Examples

### Task-Based Orchestration

The system uses a DAG-based task execution model for complex workflows:

```python
from agentic_rag.schema import TaskNode, ToolType

# Tasks are created with dependencies, priorities, and criticality
task = TaskNode(
    task_id="retrieval_1",
    tool_name="retrieve_user_docs",
    tool_type=ToolType.DATA_RETRIEVAL,
    params={"query": "refund policy"},
    dependencies=[],
    priority=1,
    is_critical=True,  # Fail fast if this task fails
    max_retries=3,
    timeout=30.0,
)
```

### Tool Agents with Performance Tracking

Create custom tool agents with built-in performance metrics:

```python
from agentic_rag.tools import BaseToolAgent
from agentic_rag.schema import ToolType

class OrderLookupAgent(BaseToolAgent):
    def __init__(self):
        super().__init__("order_lookup", ToolType.DATA_RETRIEVAL)
    
    def _execute_core(self, params):
        order_id = params.get("order_id")
        # Execute tool logic
        return {"order_id": order_id, "status": "shipped"}
```

Built-in tools already use this pattern through `FunctionToolAgent`, so monitoring works without needing a custom agent for every tool.

### Monitoring Dashboard

Real-time performance analytics:

```python
workflow = build_workflow()

# Process requests
result = workflow.run(request)

# Get comprehensive metrics
metrics = workflow.get_monitoring_metrics()
print(f"Success rate: {metrics['success_rate']:.2%}")
print(f"Avg execution time: {metrics['avg_execution_time']:.2f}s")

# Get detailed dashboard
dashboard = workflow.get_dashboard_summary()
print(f"Total tasks: {dashboard['system_metrics']['total_tasks']}")
print(f"Cost estimate: ${dashboard['system_metrics']['total_cost']:.4f}")
```

### Trainable Tool Selector

The tool selector can also be trained and loaded from checkpoints:

```python
from agentic_rag.tool_selection import train_model, save_model

selector = train_model(
    [
        ("check status for order ORD-123", "search_orders"),
        ("open a support ticket for my damaged order", "create_ticket"),
        ("summarize the user documentation", "summarize_user_docs"),
        ("summarize the refund policy", "summarize_policy_docs"),
    ],
    epochs=5,
)

save_model(selector, "artifacts/tool_selector.pt")
```

At runtime, set `TOOL_SELECTOR_MODEL_PATH` to load the checkpoint into the planner's `ToolSelectionModel`.

### Memory Transformer

The memory layer also includes a `MemoryTransformer` wrapper that organizes memory storage and retrieval:

```python
from agentic_rag.memory import MemoryTransformer

memory = MemoryTransformer(short_limit=15, long_limit=100)

memory.process("Important contract clause about exclusivity and termination.")
results = memory.query("exclusivity clause", top_k=3)
```

`MemoryTransformer` computes an importance score for each content item. High-importance memories are routed into long-term FAISS storage, while lower-importance memories remain in short-term storage unless later promoted.

### Retry Logic & Early Exit

- Tasks marked `is_critical=True` can trigger early exit on failure
- Non-critical tasks get fewer retries
- The workflow also stops if error counts exceed a safety threshold

## Usage

### CLI demo

```bash
python3 main.py
python3 main.py --async
```

The demo prints both workflow-level performance stats and aggregated monitoring metrics from tool executions.

### Programmatic usage

```python
from agentic_rag import UserRequest, build_workflow

workflow = build_workflow()

request = UserRequest(
    user_id="user-123",
    channel="web",
    message="What is the refund policy for delayed orders?"
)

state = workflow.run(request)
print(state.final_response)
```

### Batch usage

```python
requests = [
    UserRequest(user_id="user-1", channel="web", message="Check my order status"),
    UserRequest(user_id="user-2", channel="web", message="Summarize the user documentation"),
]

states = workflow.batch_run(requests)
```

### Async usage

```python
import asyncio

async def main():
    state = await workflow.run_async(
        UserRequest(
            user_id="user-123",
            channel="web",
            message="What is the refund policy?"
        )
    )
    print(state.final_response)

asyncio.run(main())
```

## Built-in Tools

- `search_orders(order_id: str)`
- `create_ticket(issue: str, severity: str = "medium")`
- `summarize_user_docs()`
- `summarize_policy_docs()`

## Performance Notes

- Retrieved document chunks are tokenized once and reused for fallback retrieval
- Response caching uses a bounded LRU-style cache in the workflow
- Async workflow mode runs memory and retrieval operations concurrently where useful
- Performance stats are exposed via `workflow.get_performance_stats()`
- Planned tool steps are executed through `TaskNode` dispatch with retry and early-exit support
- Tool selection can combine heuristic candidate filtering with trained-model scoring when a checkpoint is provided
- Memory search uses FAISS inner-product indexes when available and falls back cleanly when it is not

## Operational Notes

- Refund requests mentioning amounts over `$500` trigger the human-approval path
- Retrieval quality depends on the contents of `data/user_docs.txt` and `data/policy_docs.txt`
- Duplicate requests from the same user and channel can be served from the in-memory response cache
- Monitoring summaries are most informative for requests that actually execute tools

## Extending

- Add tools in `agentic_rag/tools.py`
- Update routing logic in `agentic_rag/router.py`
- Extend planning behavior in `agentic_rag/planner.py`
- Add new corpora in `agentic_rag/config.py` and `agentic_rag/retrieval.py`
