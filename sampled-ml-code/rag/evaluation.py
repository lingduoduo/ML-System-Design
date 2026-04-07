from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


@dataclass
class EvalExample:
    original_q: str
    rewritten_q: str
    ground_truth_answer: str
    gold_doc_ids: List[str]


@dataclass
class EvaluationDataset:
    examples: List[EvalExample] = field(default_factory=list)

    def add_example(
        self,
        original_q: str,
        rewritten_q: str,
        ground_truth_answer: str,
        gold_doc_ids: List[str],
    ) -> None:
        self.examples.append(
            EvalExample(
                original_q=original_q,
                rewritten_q=rewritten_q,
                ground_truth_answer=ground_truth_answer,
                gold_doc_ids=gold_doc_ids,
            )
        )
        logging.info("Added a new evaluation example")

    def get_all_examples(self) -> List[EvalExample]:
        return list(self.examples)


class GroundTruthDataset(EvaluationDataset):
    """Backward-compatible alias for older imports."""


def _import_ragas_dependencies() -> Dict[str, Any]:
    try:
        import pandas as pd
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness
    except ImportError as exc:
        raise ImportError(
            "RAGAS-based evaluation requires `ragas`, `datasets`, and `pandas`."
        ) from exc

    return {
        "Dataset": Dataset,
        "answer_relevancy": answer_relevancy,
        "context_precision": context_precision,
        "context_recall": context_recall,
        "evaluate": evaluate,
        "faithfulness": faithfulness,
        "pd": pd,
    }


@dataclass
class EvaluationTracker:
    history: List[Dict[str, Any]] = field(default_factory=list)

    def record_run(self, query: str, retrieved: List[Tuple[float, Dict[str, Any]]], response: str) -> Dict[str, Any]:
        event = {
            "query": query,
            "retrieved_chunks": [metadata.get("text", "") for _, metadata in retrieved],
            "response_preview": response[:200],
        }
        self.history.append(event)
        return event

    def log_request(self, query: str, retrieved: List[Tuple[float, Dict[str, Any]]], response: str) -> Dict[str, Any]:
        return self.record_run(query, retrieved, response)


class Monitor(EvaluationTracker):
    """Backward-compatible alias for older imports."""


def recall_at_k(retrieved_doc_ids: List[str], gold_doc_ids: List[str], k: int) -> float:
    if not gold_doc_ids:
        return 0.0
    retrieved_top_k = set(retrieved_doc_ids[:k])
    gold = set(gold_doc_ids)
    return len(retrieved_top_k & gold) / len(gold)


def get_retrieved_doc_ids(docs: List[Any], id_key: str = "source") -> List[str]:
    doc_ids: List[str] = []
    for doc in docs:
        metadata = getattr(doc, "metadata", {}) or {}
        doc_ids.append(str(metadata.get(id_key, "")))
    return doc_ids


def _import_judge_dependencies() -> Dict[str, Any]:
    try:
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.prompts import ChatPromptTemplate
    except ImportError as exc:
        raise ImportError("Judge-based evaluation requires `langchain-core`.") from exc

    return {
        "ChatPromptTemplate": ChatPromptTemplate,
        "StrOutputParser": StrOutputParser,
    }


def create_judge_prompt() -> Any:
    deps = _import_judge_dependencies()
    return deps["ChatPromptTemplate"].from_messages(
        [
            (
                "system",
                "You are a strict evaluator (judge) for a RAG QA system.\n"
                "Given a question, a candidate answer, and a reference answer,\n"
                "decide if the candidate answer is correct.\n\n"
                "Return ONLY valid JSON with keys:\n"
                '  "verdict": "correct" | "incorrect"\n'
                '  "score": number between 0 and 1\n'
                '  "rationale": short explanation\n\n'
                "Be conservative: if the candidate misses key facts from the reference, mark incorrect.\n"
                "Do not output any extra text besides JSON.",
            ),
            (
                "human",
                "Question:\n{question}\n\n"
                "Candidate Answer:\n{answer}\n\n"
                "Reference Answer:\n{reference}\n",
            ),
        ]
    )


def build_judge_chain(judge_llm: Any) -> Any:
    deps = _import_judge_dependencies()
    return create_judge_prompt() | judge_llm | deps["StrOutputParser"]()


def _extract_json_object(text: str) -> Optional[str]:
    match = re.search(r"\{.*\}", text.strip(), flags=re.DOTALL)
    if match:
        return match.group(0)
    return None


def safe_json_loads(text: str) -> Dict[str, Any]:
    extracted = _extract_json_object(text)
    return json.loads(extracted if extracted is not None else text.strip())


def _parse_non_json_judge(text: str) -> Dict[str, Any]:
    normalized = text.strip()
    verdict_match = re.search(r"verdict\s*:\s*(correct|incorrect)", normalized, flags=re.I)
    verdict = verdict_match.group(1).lower() if verdict_match else None

    if verdict is None:
        if re.search(r"\bincorrect\b", normalized, flags=re.I):
            verdict = "incorrect"
        elif re.search(r"\bcorrect\b", normalized, flags=re.I):
            verdict = "correct"
        else:
            verdict = "incorrect"

    rationale_match = re.search(r"rationale\s*:\s*(.*)", normalized, flags=re.I | re.S)
    rationale = rationale_match.group(1).strip() if rationale_match else normalized[:300]
    score = 1.0 if verdict == "correct" else 0.0
    return {"verdict": verdict, "score": score, "rationale": rationale}


def judge_answer(judge_chain: Any, question: str, answer: str, reference: str) -> Dict[str, Any]:
    raw = judge_chain.invoke({"question": question, "answer": answer, "reference": reference})

    try:
        parsed = safe_json_loads(raw)
        return {
            "verdict": str(parsed.get("verdict", "incorrect")).lower(),
            "score": float(parsed.get("score", 0.0)),
            "rationale": str(parsed.get("rationale", "")).strip(),
            "raw": raw,
        }
    except Exception as exc:
        logging.warning("Judge JSON parse failed: %s; falling back to heuristic parsing.", exc)

    parsed = _parse_non_json_judge(raw)
    parsed["raw"] = raw
    return parsed


@dataclass
class RagasEvaluator:
    metrics: List[Any] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.metrics:
            deps = _import_ragas_dependencies()
            self.metrics = [
                deps["answer_relevancy"],
                deps["context_precision"],
                deps["context_recall"],
                deps["faithfulness"],
            ]

    def build_dataset(
        self,
        questions: List[str],
        answers: List[str],
        contexts: List[List[str]],
        ground_truths: List[str],
    ) -> Any:
        deps = _import_ragas_dependencies()
        dataframe = deps["pd"].DataFrame(
            {
                "question": questions,
                "answer": answers,
                "contexts": contexts,
                "ground_truth": ground_truths,
            }
        )
        return deps["Dataset"].from_pandas(dataframe)

    def evaluate_dataset(self, dataset: Any) -> Dict[str, Any]:
        deps = _import_ragas_dependencies()
        result = deps["evaluate"](dataset, metrics=self.metrics)
        if hasattr(result, "to_pandas"):
            return result.to_pandas().to_dict(orient="list")
        if hasattr(result, "dict"):
            return result.dict()
        return dict(result)

    def evaluate_examples(self, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        dataset = self.build_dataset(
            questions=[row["question"] for row in rows],
            answers=[row["answer"] for row in rows],
            contexts=[row["contexts"] for row in rows],
            ground_truths=[row["ground_truth"] for row in rows],
        )
        return self.evaluate_dataset(dataset)


@dataclass
class RAGEvaluator:
    retriever: Any
    answer_generator: Callable[[str], str]
    judge_chain: Any
    id_key: str = "source"
    k_list: List[int] = field(default_factory=lambda: [3, 5, 10])
    judge_correct_threshold: float = 0.5

    def _retrieve_docs(self, query: str) -> List[Any]:
        if hasattr(self.retriever, "invoke"):
            return self.retriever.invoke(query)
        if hasattr(self.retriever, "get_relevant_documents"):
            return self.retriever.get_relevant_documents(query)
        raise TypeError("Retriever must support .invoke(query) or .get_relevant_documents(query)")

    def evaluate_dataset(self, dataset: EvaluationDataset) -> Dict[str, Any]:
        examples = dataset.get_all_examples()
        total = len(examples)
        judge_correct = 0
        recall_sums = {k: 0.0 for k in self.k_list}
        per_example: List[Dict[str, Any]] = []

        for example in examples:
            query = example.rewritten_q
            docs = self._retrieve_docs(query)
            retrieved_ids = get_retrieved_doc_ids(docs, id_key=self.id_key)

            recall_scores = {
                k: recall_at_k(retrieved_ids, example.gold_doc_ids, k)
                for k in self.k_list
            }
            for k, recall in recall_scores.items():
                recall_sums[k] += recall

            answer = self.answer_generator(query)
            judged = judge_answer(
                self.judge_chain,
                question=example.original_q,
                answer=answer,
                reference=example.ground_truth_answer,
            )
            is_correct = (
                judged["verdict"] == "correct"
                and judged["score"] >= self.judge_correct_threshold
            )
            judge_correct += int(is_correct)

            logging.info(
                "Q: %s\nRewritten: %s\nJudge: %s (score=%.2f)\nRecall: %s",
                example.original_q,
                example.rewritten_q,
                judged["verdict"],
                judged["score"],
                ", ".join(f"@{k}={recall_scores[k]:.2f}" for k in self.k_list),
            )

            per_example.append(
                {
                    "original_q": example.original_q,
                    "rewritten_q": example.rewritten_q,
                    "gold_doc_ids": example.gold_doc_ids,
                    "retrieved_doc_ids": retrieved_ids,
                    "recall_at_k": recall_scores,
                    "answer": answer,
                    "judge": judged,
                    "is_correct": is_correct,
                }
            )

        avg_recall = {
            k: (recall_sums[k] / total if total else 0.0)
            for k in self.k_list
        }
        return {
            "total": total,
            "judge_correct": judge_correct,
            "judge_accuracy": (judge_correct / total if total else 0.0),
            "avg_recall_at_k": avg_recall,
            "examples": per_example,
        }

    @staticmethod
    def generate_report(metrics: Dict[str, Any]) -> str:
        lines = [
            "Evaluation Report",
            f"- Total examples: {metrics['total']}",
            f"- Judge accuracy: {metrics['judge_accuracy']:.2%} ({metrics['judge_correct']}/{metrics['total']})",
            "- Average Recall@K:",
        ]
        for k, value in metrics["avg_recall_at_k"].items():
            lines.append(f"  - Recall@{k}: {value:.2%}")
        return "\n".join(lines)
