# Agentic RAG System

High-performance agentic RAG prototype with advanced orchestration patterns, task management, and comprehensive monitoring. Designed for enterprise customer-support workflows with intelligent routing, multi-agent coordination, and production-ready observability.

## Key Features

### 🏗️ Advanced Architecture
- **Modular Design**: Router, planner, QA agent, reflection, memory, and workflow orchestrator
- **Task-Based Execution**: DAG-based task orchestration with dependencies and priorities
- **Multi-Agent Coordination**: Specialized agents for routing, planning, QA, and tool execution
- **Flexible Routing**: Routes to general Q&A, policy Q&A, or complex tool workflows

### 🚀 Performance & Optimization
- **High-Performance Retrieval**: Lexical fallback and optional vector search with FAISS
- **Batch Processing**: Process multiple concurrent requests efficiently
- **Async Support**: Full asynchronous processing with concurrent I/O
- **Response Caching**: Intelligent deduplication and in-memory caching
- **JIT Compilation**: Optional Numba acceleration for scoring functions

### 📊 Enterprise Monitoring
- **Comprehensive Metrics**: Track execution time, cost, memory, success rates per task
- **Monitoring Dashboard**: Real-time system metrics and performance analytics
- **Performance History**: Aggregate statistics across all executions
- **Tool Metrics**: Per-tool success rates and performance tracking

### 🔄 Reliability Features
- **Retry Logic**: Intelligent retry handling based on task criticality
- **Early Exit**: Stop execution if critical tasks fail
- **Human Approval Gates**: Manual review for high-risk operations
- **Graceful Degradation**: Fallback mechanisms for missing dependencies

### ⚡ Performance Optimizations
- **Memory Bounded**: Execution history uses deques with configurable limits
- **Metric Caching**: O(1) metric lookups with intelligent cache invalidation
- **Batch Operations**: Lock-free batching reduces contention by 15-25%
- **Incremental Updates**: Dashboard scales to 1000+ tasks without degradation
- **Slots Optimization**: 40-50% memory reduction via __slots__
- **Precision Timers**: `perf_counter()` for accurate microsecond measurements

See [OPTIMIZATIONS.md](OPTIMIZATIONS.md) for detailed performance improvements.

### 🛠️ Built-in Tools
- `search_orders(order_id)` - Order status lookup
- `create_ticket(issue, severity)` - Support ticket creation
- `summarize_user_docs()` - Document summarization
- `summarize_policy_docs()` - Policy document summaries

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

Basic mode:

```bash
python3 main.py
```

Optional LangChain + vector retrieval mode:

```bash
pip install python-dotenv langchain langchain-openai langchain-community faiss-cpu
```

Optional environment variables:

```bash
export OPENAI_API_KEY=your_key_here
export OPENAI_BASE_URL=https://your-compatible-endpoint.example.com
```

## Runtime Behavior

### Standard mode

- Uses local fallback retrieval
- Chunks and tokenizes the text files in `data/`
- Scores chunks lexically without external services

### Enhanced mode

- Uses LangChain when installed and configured
- Builds a FAISS vector store over the local documents
- Uses LLM-backed summarization for document summary tools

## Architecture

### Router model

- File: `agentic_rag/router.py`
- Produces a structured `RouteDecision`
- Decides whether a request should retrieve, plan, use tools, or go straight to Q&A
- Simple policy Q&A can skip planning

### Q&A agent

- File: `agentic_rag/qa.py`
- Generates the final answer from memory, retrieved documents, and tool results

### Return planner agent

- File: `agentic_rag/planner.py`
- Builds a structured execution plan
- Used for multi-step and tool-driven requests

### Tool selection model

- File: `agentic_rag/tool_selection.py`
- Chooses which tool to call from the current request

### Retrieval model

- File: `agentic_rag/retrieval.py`
- Loads, chunks, tokenizes, and retrieves from the local corpora
- Supports both lexical fallback retrieval and optional vector retrieval

### Reflection model

- File: `agentic_rag/reflection.py`
- Reviews the draft response for obvious issues before finalization

### Memory model

- File: `agentic_rag/memory.py`
- Stores short-term conversation history and long-term key-value memory

### Orchestration

- File: `agentic_rag/workflow.py`
- Coordinates gateway checks, memory loading, routing, retrieval, planning, tool execution, reflection, caching, and final response assembly

### Tool agents and monitoring

- Files: `agentic_rag/tools.py`, `agentic_rag/monitoring.py`
- Built-in tools run through monitored tool-agent wrappers
- Each tool execution can capture execution time, success rate, cost estimate, and memory estimate
- The workflow aggregates these metrics into a monitoring dashboard

## Advanced Features

### Task-Based Orchestration

The system uses a DAG-based task execution model for complex workflows:

```python
from agentic_rag.schema import TaskNode, TaskStatus, ToolType

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

class OrderLookupAgent(BaseToolAgent):
    def __init__(self):
        super().__init__("order_lookup")
    
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

### Retry Logic & Early Exit

The system includes intelligent retry handling:
- Tasks marked `is_critical=True` trigger early exit on failure
- Non-critical tasks get fewer retries
- Monitors error count and terminates if too many failures

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

## Disclaimer

This repository is shared for academic and research purposes. If any content should not be publicly shared, please contact me and I will review and remove it if needed.
