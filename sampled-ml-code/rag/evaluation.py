from __future__ import annotations

from functools import lru_cache
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


@lru_cache(maxsize=1)
def _import_retrieval_comparison_dependencies() -> Dict[str, Any]:
    try:
        import numpy as np
        import pandas as pd
        import torch
        from datasets import Dataset
        from sentence_transformers import SentenceTransformer
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
    except ImportError as exc:
        raise ImportError(
            "Retrieval comparison utilities require `numpy`, `pandas`, `torch`, "
            "`datasets`, `sentence-transformers`, and `scikit-learn`."
        ) from exc

    return {
        "Dataset": Dataset,
        "SentenceTransformer": SentenceTransformer,
        "TfidfVectorizer": TfidfVectorizer,
        "cosine_similarity": cosine_similarity,
        "np": np,
        "pd": pd,
        "torch": torch,
    }


@lru_cache(maxsize=1)
def _import_ragas_dependencies() -> Dict[str, Any]:
    try:
        import pandas as pd
        from datasets import Dataset
        from ragas import evaluate
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms import LangchainLLMWrapper
        from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness
    except ImportError as exc:
        raise ImportError(
            "RAGAS-based evaluation requires `ragas`, `datasets`, and `pandas`."
        ) from exc

    return {
        "Dataset": Dataset,
        "LangchainEmbeddingsWrapper": LangchainEmbeddingsWrapper,
        "LangchainLLMWrapper": LangchainLLMWrapper,
        "answer_relevancy": answer_relevancy,
        "context_precision": context_precision,
        "context_recall": context_recall,
        "evaluate": evaluate,
        "faithfulness": faithfulness,
        "pd": pd,
    }


@lru_cache(maxsize=1)
def _import_chunking_experiment_dependencies() -> Dict[str, Any]:
    try:
        import pandas as pd
        from datasets import Dataset
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except ImportError as exc:
        raise ImportError(
            "Chunking experiments require `langchain-openai`, `langchain-text-splitters`, "
            "`datasets`, and `pandas`."
        ) from exc

    ragas_deps = _import_ragas_dependencies()
    return {
        "ChatOpenAI": ChatOpenAI,
        "Dataset": Dataset,
        "LangchainEmbeddingsWrapper": ragas_deps["LangchainEmbeddingsWrapper"],
        "LangchainLLMWrapper": ragas_deps["LangchainLLMWrapper"],
        "OpenAIEmbeddings": OpenAIEmbeddings,
        "RecursiveCharacterTextSplitter": RecursiveCharacterTextSplitter,
        "evaluate": ragas_deps["evaluate"],
        "metrics": [
            ragas_deps["answer_relevancy"],
            ragas_deps["context_precision"],
            ragas_deps["context_recall"],
            ragas_deps["faithfulness"],
        ],
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


@lru_cache(maxsize=1)
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
class RetrievalComparisonSample:
    question: str
    ground_truth: str


@dataclass
class RetrievalMethodComparison:
    corpus: List[str]
    embedding_model_name: str = "all-MiniLM-L6-v2"
    inverted_top_k: int = 2
    vector_top_k: int = 2
    ragas_evaluator: RagasEvaluator = field(default_factory=RagasEvaluator)
    _vectorizer: Any = field(init=False, repr=False)
    _tfidf_matrix: Any = field(init=False, repr=False)
    _embedding_model: Any = field(init=False, repr=False)
    _corpus_embeddings: Any = field(init=False, repr=False)
    _deps: Dict[str, Any] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._deps = _import_retrieval_comparison_dependencies()
        self._vectorizer = self._deps["TfidfVectorizer"]()
        self._tfidf_matrix = self._vectorizer.fit_transform(self.corpus)
        self._embedding_model = self._deps["SentenceTransformer"](self.embedding_model_name)
        self._corpus_embeddings = self._embedding_model.encode(self.corpus, convert_to_tensor=True)

    def retrieve_with_inverted_index(self, query: str, top_k: int | None = None) -> List[str]:
        limit = top_k if top_k is not None else self.inverted_top_k
        query_vector = self._vectorizer.transform([query])
        scores = self._deps["cosine_similarity"](query_vector, self._tfidf_matrix).flatten()
        top_indices = self._deps["np"].argsort(scores)[-limit:][::-1]
        return [self.corpus[index] for index in top_indices if scores[index] > 0]

    def retrieve_with_vector_index(self, query: str, top_k: int | None = None) -> List[str]:
        limit = top_k if top_k is not None else self.vector_top_k
        query_embedding = self._embedding_model.encode(query, convert_to_tensor=True)
        scores = self._deps["torch"].nn.functional.cosine_similarity(
            query_embedding.unsqueeze(0),
            self._corpus_embeddings,
            dim=1,
        ).cpu().numpy()
        top_indices = self._deps["np"].argsort(scores)[-limit:][::-1]
        return [self.corpus[index] for index in top_indices if scores[index] > 0]

    def build_comparison_rows(
        self,
        samples: List[RetrievalComparisonSample],
        answer_builder: Callable[[str, List[str], str], str] | None = None,
    ) -> List[Dict[str, Any]]:
        def default_answer_builder(question: str, contexts: List[str], method: str) -> str:
            if not contexts:
                return "No relevant information found."
            prefix = "keyword retrieval" if method == "Inverted Index" else "semantic retrieval"
            return f"Based on {prefix}, the answer is supported by: {contexts[0]}"

        build_answer = answer_builder or default_answer_builder
        rows: List[Dict[str, Any]] = []

        for sample in samples:
            inverted_contexts = self.retrieve_with_inverted_index(sample.question)
            vector_contexts = self.retrieve_with_vector_index(sample.question)

            rows.append(
                {
                    "question": sample.question,
                    "contexts": inverted_contexts,
                    "answer": build_answer(sample.question, inverted_contexts, "Inverted Index"),
                    "ground_truth": sample.ground_truth,
                    "retrieval_method": "Inverted Index",
                }
            )
            rows.append(
                {
                    "question": sample.question,
                    "contexts": vector_contexts,
                    "answer": build_answer(sample.question, vector_contexts, "Vector Index"),
                    "ground_truth": sample.ground_truth,
                    "retrieval_method": "Vector Index",
                }
            )

        return rows

    def build_dataset(self, rows: List[Dict[str, Any]]) -> Any:
        dataframe = self._deps["pd"].DataFrame(rows)
        return self._deps["Dataset"].from_pandas(dataframe)

    def evaluate_samples(
        self,
        samples: List[RetrievalComparisonSample],
        answer_builder: Callable[[str, List[str], str], str] | None = None,
    ) -> Dict[str, Any]:
        rows = self.build_comparison_rows(samples, answer_builder=answer_builder)
        ragas_metrics = self.ragas_evaluator.evaluate_examples(rows)
        metric_frame = self._deps["pd"].DataFrame(ragas_metrics)
        original_frame = self._deps["pd"].DataFrame(rows)
        combined = self._deps["pd"].concat(
            [original_frame.reset_index(drop=True), metric_frame.reset_index(drop=True)],
            axis=1,
        )
        return {
            "rows": rows,
            "metrics": ragas_metrics,
            "combined": combined.to_dict(orient="list"),
        }


@dataclass
class ChunkingEvaluationSample:
    question: str
    answer: str
    ground_truth: str


@dataclass
class ChunkingStrategyEvaluator:
    llm_model_name: str = "gpt-4o-mini"
    embedding_model_name: str = "text-embedding-3-small"
    temperature: float = 0.0
    separators: List[str] = field(default_factory=lambda: ["\n", ".", " "])

    def _build_wrapped_judges(self) -> Tuple[Any, Any]:
        deps = _import_chunking_experiment_dependencies()
        llm = deps["LangchainLLMWrapper"](
            deps["ChatOpenAI"](model=self.llm_model_name, temperature=self.temperature)
        )
        embeddings = deps["LangchainEmbeddingsWrapper"](
            deps["OpenAIEmbeddings"](model=self.embedding_model_name)
        )
        return llm, embeddings

    def chunk_text(self, text: str, chunk_size: int = 200, chunk_overlap: int = 50) -> List[str]:
        deps = _import_chunking_experiment_dependencies()
        splitter = deps["RecursiveCharacterTextSplitter"](
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=self.separators,
        )
        return splitter.split_text(text)

    def evaluate_chunking_strategy(
        self,
        text: str,
        samples: List[ChunkingEvaluationSample],
        chunk_sizes: List[int],
        chunk_overlaps: List[int],
    ) -> Dict[str, Any]:
        deps = _import_chunking_experiment_dependencies()
        evaluator_llm, evaluator_embeddings = self._build_wrapped_judges()
        result_frames = []

        for size in chunk_sizes:
            for overlap in chunk_overlaps:
                chunks = self.chunk_text(text, chunk_size=size, chunk_overlap=overlap)
                dataframe = deps["pd"].DataFrame(
                    {
                        "question": [sample.question for sample in samples],
                        "answer": [sample.answer for sample in samples],
                        "contexts": [chunks for _ in samples],
                        "ground_truth": [sample.ground_truth for sample in samples],
                    }
                )
                dataset = deps["Dataset"].from_pandas(dataframe)
                result = deps["evaluate"](
                    dataset,
                    metrics=deps["metrics"],
                    llm=evaluator_llm,
                    embeddings=evaluator_embeddings,
                )

                result_frame = result.to_pandas()
                result_frame["chunk_size"] = size
                result_frame["chunk_overlap"] = overlap
                result_frames.append(result_frame)

        if not result_frames:
            return {"results": []}

        combined = deps["pd"].concat(result_frames, ignore_index=True)
        return {"results": combined.to_dict(orient="list")}


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
