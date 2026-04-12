# Agentic RAG System

Modular agentic RAG prototype for customer-support style workflows. The project is organized around routing, retrieval, planning, tool use, reflection, memory, and a workflow orchestrator.

## Overview

- Routes requests into `general_qa`, `policy_qa`, or `tool_workflow`
- Retrieves grounded context from local text documents under `data/`
- Supports built-in tools for order lookup, ticket creation, and document summaries
- Uses reflection and optional human approval for risky cases
- Exposes both synchronous and asynchronous workflow entrypoints

## Project Layout

```text
.
├── agentic_rag
│   ├── approval.py
│   ├── builder.py
│   ├── config.py
│   ├── gateway.py
│   ├── memory.py
│   ├── observability.py
│   ├── planner.py
│   ├── qa.py
│   ├── reflection.py
│   ├── retrieval.py
│   ├── router.py
│   ├── schema.py
│   ├── tool_selection.py
│   ├── tools.py
│   └── workflow.py
├── data
│   ├── policy_docs.txt
│   └── user_docs.txt
├── main.py
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

## Usage

### CLI demo

```bash
python3 main.py
python3 main.py --async
```

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

## Extending

- Add tools in `agentic_rag/tools.py`
- Update routing logic in `agentic_rag/router.py`
- Extend planning behavior in `agentic_rag/planner.py`
- Add new corpora in `agentic_rag/config.py` and `agentic_rag/retrieval.py`

## Disclaimer

This repository is shared for academic and research purposes. If any content should not be publicly shared, please contact me and I will review and remove it if needed.
