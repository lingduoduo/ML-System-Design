from __future__ import annotations

import asyncio
import time
from agentic_rag import UserRequest, build_workflow

DEMO_QUERIES = [
    "What is the refund policy for delayed orders, and can you check my order status?",
    "Summarize the user documentation",
    "What are the policy guidelines for refunds?",
    "I have an issue with my order ORD-123",
]


def _print_state(index: int, query: str, state) -> None:
    print(f"\n=== Test Query {index} ===")
    print(f"Query: {query}")
    print("FINAL RESPONSE")
    print("-" * 40)
    print(state.final_response)
    print("\nTRACE (last 5 events)")
    print("-" * 40)
    for item in state.trace[-5:]:
        print(item)
    print("=" * 80)


def _print_workflow_summary(workflow) -> None:
    print(f"Performance stats: {workflow.get_performance_stats()}")
    print(f"Monitoring summary: {workflow.get_monitoring_metrics()}")


async def main_async() -> None:
    workflow = build_workflow()
    print("Running async version for better performance...")
    start_time = time.perf_counter()

    for index, query in enumerate(DEMO_QUERIES, start=1):
        request = UserRequest(
            user_id=f"user-{index}",
            channel="web",
            message=query,
        )
        state = await workflow.run_async(request)
        _print_state(index, query, state)

    elapsed = time.perf_counter() - start_time
    print(f"Elapsed: {elapsed:.2f}s")
    _print_workflow_summary(workflow)


def main_sync() -> None:
    workflow = build_workflow()
    print("Running sync version...")
    start_time = time.perf_counter()

    for index, query in enumerate(DEMO_QUERIES, start=1):
        request = UserRequest(
            user_id=f"user-{index}",
            channel="web",
            message=query,
        )
        state = workflow.run(request)
        _print_state(index, query, state)

    elapsed = time.perf_counter() - start_time
    print(f"Elapsed: {elapsed:.2f}s")
    _print_workflow_summary(workflow)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--async":
        asyncio.run(main_async())
    else:
        main_sync()
