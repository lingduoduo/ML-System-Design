from __future__ import annotations

import asyncio
import time
from agentic_rag import UserRequest, build_workflow


async def main_async() -> None:
    workflow = build_workflow()
    test_queries = [
        "What is the refund policy for delayed orders, and can you check my order status?",
        "Summarize the user documentation",
        "What are the policy guidelines for refunds?",
        "I have an issue with my order ORD-123",
    ]

    print("Running async version for better performance...")
    start_time = time.time()

    for index, query in enumerate(test_queries, start=1):
        print(f"\n=== Test Query {index} ===")
        print(f"Query: {query}")

        request = UserRequest(
            user_id=f"user-{index}",
            channel="web",
            message=query,
        )
        state = await workflow.run_async(request)

        print("FINAL RESPONSE")
        print("-" * 40)
        print(state.final_response)

        print("\nTRACE (last 5 events)")
        print("-" * 40)
        for item in state.trace[-5:]:
            print(item)
        print("=" * 80)

    end_time = time.time()
    print(".2f")
    print(f"Performance stats: {workflow.get_performance_stats()}")


def main_sync() -> None:
    workflow = build_workflow()
    test_queries = [
        "What is the refund policy for delayed orders, and can you check my order status?",
        "Summarize the user documentation",
        "What are the policy guidelines for refunds?",
        "I have an issue with my order ORD-123",
    ]

    print("Running sync version...")
    start_time = time.time()

    for index, query in enumerate(test_queries, start=1):
        print(f"\n=== Test Query {index} ===")
        print(f"Query: {query}")

        request = UserRequest(
            user_id=f"user-{index}",
            channel="web",
            message=query,
        )
        state = workflow.run(request)

        print("FINAL RESPONSE")
        print("-" * 40)
        print(state.final_response)

        print("\nTRACE (last 5 events)")
        print("-" * 40)
        for item in state.trace[-5:]:
            print(item)
        print("=" * 80)

    end_time = time.time()
    print(".2f")
    print(f"Performance stats: {workflow.get_performance_stats()}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--async":
        asyncio.run(main_async())
    else:
        main_sync()
