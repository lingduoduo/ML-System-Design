from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class ExperimentTracker:
    runs: List[Dict] = field(default_factory=list)

    def log_run(self, params: Dict, metrics: Dict) -> None:
        self.runs.append({"params": params, "metrics": metrics})


@dataclass
class FineTunedLLM:
    model_name: str

    def generate(self, prompt: str) -> str:
        return f"[{self.model_name}] Response to: {prompt}"


def fine_tune_llm(base_model: str, dataset: List[Dict]) -> FineTunedLLM:
    return FineTunedLLM(model_name=f"{base_model}-finetuned")


def evaluate_candidate(model: FineTunedLLM, test_data: List[Dict]) -> Dict:
    score = min(1.0, 0.5 + len(test_data) * 0.01)
    return {"accuracy": score, "samples_evaluated": len(test_data)}


@dataclass
class ModelRegistry:
    models: Dict[str, FineTunedLLM] = field(default_factory=dict)

    def register(self, model_name: str, model_obj: FineTunedLLM) -> None:
        self.models[model_name] = model_obj

    def get(self, model_name: str) -> FineTunedLLM | None:
        return self.models.get(model_name)


def train_and_register_model(
    instruct_dataset: List[Dict],
    base_model: str = "base-llm",
    acceptance_threshold: float = 0.5,
) -> tuple[ExperimentTracker, ModelRegistry, FineTunedLLM | None, Dict]:
    tracker = ExperimentTracker()
    registry = ModelRegistry()

    candidate = fine_tune_llm(base_model, instruct_dataset)
    eval_slice = instruct_dataset[: min(5, len(instruct_dataset))]
    metrics = evaluate_candidate(candidate, eval_slice)

    tracker.log_run(
        params={"base_model": base_model, "train_size": len(instruct_dataset)},
        metrics=metrics,
    )

    accepted_model = None
    if metrics["accuracy"] > acceptance_threshold:
        registry.register("accepted-llm", candidate)
        accepted_model = candidate

    return tracker, registry, accepted_model, metrics
