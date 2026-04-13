"""Advanced usage examples for the Agentic RAG system.

This module demonstrates task-based orchestration, custom tool agents,
and comprehensive performance monitoring.
"""

from agentic_rag import UserRequest, build_workflow
from agentic_rag.tools import BaseToolAgent
from agentic_rag.schema import PerformanceMetrics, TaskNode, TaskStatus, ToolType


class CustomAnalysisAgent(BaseToolAgent):
    """Example custom tool agent with performance tracking."""

    def __init__(self):
        super().__init__("custom_analyzer", ToolType.ANALYSIS)

    def _execute_core(self, params: dict) -> str:
        """Execute custom analysis logic."""
        text = params.get("text", "")
        return f"Analysis of '{text[:50]}...': Content looks relevant for customer support."

    def _estimate_cost(self, params: dict) -> float:
        """Estimate this tool's cost."""
        return 0.05  # Higher cost than default

    def _get_memory_usage(self) -> float:
        """Memory usage estimate in MB."""
        return 25.0


def example_basic_monitoring():
    """Example: Monitor performance metrics."""
    print("=" * 60)
    print("Example 1: Basic Monitoring")
    print("=" * 60)

    workflow = build_workflow()

    # Process a request
    request = UserRequest(
        user_id="user-123",
        channel="web",
        message="What is the refund policy for delayed orders?"
    )

    result = workflow.run(request)

    # Get performance metrics
    metrics = workflow.get_monitoring_metrics()
    print(f"\n📊 Performance Metrics:")
    print(f"  Success rate: {metrics['success_rate']:.2%}")
    print(f"  Avg execution time: {metrics['avg_execution_time']:.3f}s")
    print(f"  Total cost estimate: ${metrics['total_cost']:.4f}")
    print(f"  Total tasks: {metrics['total_tasks']}")

    # Get detailed dashboard
    dashboard = workflow.get_dashboard_summary()
    system_metrics = dashboard['system_metrics']
    print(f"\n📈 System Summary:")
    print(f"  Completed tasks: {system_metrics['completed_tasks']}")
    print(f"  Failed tasks: {system_metrics['failed_tasks']}")
    print(f"  Skipped tasks: {system_metrics['skipped_tasks']}")
    print(f"  Success rate: {system_metrics['success_rate']:.2%}")


def example_batch_processing():
    """Example: Batch process multiple requests with monitoring."""
    print("\n" + "=" * 60)
    print("Example 2: Batch Processing with Monitoring")
    print("=" * 60)

    workflow = build_workflow()

    # Create batch requests
    requests = [
        UserRequest(
            user_id=f"user-{i}",
            channel="web",
            message=msg
        )
        for i, msg in enumerate([
            "What is your refund policy?",
            "How do I create a support ticket?",
            "What is the status of order ORD-12345?"
        ], 1)
    ]

    # Process batch
    print(f"\nProcessing {len(requests)} requests...")
    results = workflow.batch_run(requests)

    # Show results with metrics
    for i, result in enumerate(results, 1):
        print(f"\nRequest {i}:")
        print(f"  Status: {'✓ OK' if result.final_response else '✗ Failed'}")
        print(f"  Response length: {len(result.final_response) if result.final_response else 0} chars")

    # Get overall metrics
    metrics = workflow.get_monitoring_metrics()
    print(f"\n📊 Batch Metrics:")
    print(f"  Total tasks processed: {metrics['total_tasks']}")
    print(f"  Success rate: {metrics['success_rate']:.2%}")
    print(f"  Total execution time: {metrics['avg_execution_time']}s")


def example_task_node_creation():
    """Example: Create task nodes for orchestration."""
    print("\n" + "=" * 60)
    print("Example 3: Task Node Creation")
    print("=" * 60)

    # Create retrieval task
    retrieval_task = TaskNode(
        task_id="task_retrieve_docs",
        tool_name="retrieve_user_docs",
        tool_type=ToolType.DATA_RETRIEVAL,
        params={"query": "refund policy", "top_k": 3},
        dependencies=[],
        priority=1,
        max_retries=3,
        timeout=30.0,
        is_critical=True,  # This is critical - fail fast if it fails
    )

    # Create planning task that depends on retrieval
    planning_task = TaskNode(
        task_id="task_plan",
        tool_name="planner_agent",
        tool_type=ToolType.ANALYSIS,
        params={"include_tools": True},
        dependencies=["task_retrieve_docs"],  # Depends on retrieval
        priority=2,
        max_retries=2,
        timeout=30.0,
        is_critical=True,
    )

    # Create tool execution task that depends on planning
    tool_task = TaskNode(
        task_id="task_execute_tools",
        tool_name="tool_executor",
        tool_type=ToolType.GENERATION,
        params={"timeout": 10},
        dependencies=["task_plan"],  # Depends on plan
        priority=3,
        max_retries=2,
        timeout=10.0,
        is_critical=False,  # Non-critical - can degrade
    )

    # Create response generation task that depends on everything
    response_task = TaskNode(
        task_id="task_generate_response",
        tool_name="response_generator",
        tool_type=ToolType.GENERATION,
        params={},
        dependencies=["task_retrieve_docs", "task_plan", "task_execute_tools"],
        priority=4,
        max_retries=1,
        timeout=30.0,
        is_critical=True,
    )

    print("\n✓ Task DAG created with dependencies:")
    tasks = [retrieval_task, planning_task, tool_task, response_task]
    for task in tasks:
        deps_str = f" (depends on {', '.join(task.dependencies)})" if task.dependencies else ""
        print(f"  {task.task_id}: {task.tool_name}{deps_str}")
        print(f"    - Type: {task.tool_type.value}, Priority: {task.priority}, Critical: {task.is_critical}")


def example_custom_tool_agent():
    """Example: Create and use custom tool agent."""
    print("\n" + "=" * 60)
    print("Example 4: Custom Tool Agent")
    print("=" * 60)

    # Create custom agent
    analyzer = CustomAnalysisAgent()

    # Execute the tool
    result = analyzer.execute({"text": "This is a test message about order refunds"})

    print(f"\n✓ Custom Agent Execution:")
    print(f"  Tool name: {result.tool_name}")
    print(f"  Status: {result.status.value}")
    print(f"  Result: {result.result}")
    print(f"\n  Performance Metrics:")
    print(f"    - Execution time: {result.performance.execution_time:.3f}s")
    print(f"    - Cost estimate: ${result.performance.cost_estimate:.4f}")
    print(f"    - Memory usage: {result.performance.memory_usage:.1f} MB")
    print(f"    - Success rate: {result.performance.success_rate:.2%}")

    # Get agent summary
    metrics = analyzer.get_metrics()
    print(f"\n  Agent Summary:")
    print(f"    - Total calls: {int(metrics['call_count'])}")
    print(f"    - Success rate: {metrics['success_rate']:.2%}")
    print(f"    - Avg execution time: {metrics['avg_execution_time']:.3f}s")


if __name__ == "__main__":
    print("\n" + "🚀 " * 20)
    print("Advanced Agentic RAG System Examples")
    print("🚀 " * 20)

    try:
        # Run examples
        example_basic_monitoring()
        example_batch_processing()
        example_task_node_creation()
        example_custom_tool_agent()

        print("\n" + "=" * 60)
        print("✅ All examples completed successfully!")
        print("=" * 60 + "\n")

    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()
