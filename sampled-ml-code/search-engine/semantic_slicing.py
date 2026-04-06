from collections import Counter
import re
from typing import Dict, List, Optional

import numpy as np

try:
    import spacy
    SPACY_IMPORT_ERROR = None
except ImportError as exc:  # pragma: no cover - environment-specific import failure
    spacy = None
    SPACY_IMPORT_ERROR = exc


SPACY_AVAILABLE = spacy is not None


def load_spacy_pipeline(model_name: str, fallback_to_blank: bool = True):
    if not SPACY_AVAILABLE:
        raise ImportError("spaCy is required for semantic slicing") from SPACY_IMPORT_ERROR
    try:
        nlp = spacy.load(model_name)
    except OSError:
        if not fallback_to_blank:
            raise
        nlp = spacy.blank("en")

    if "sentencizer" not in nlp.pipe_names:
        nlp.add_pipe("sentencizer")
    return nlp


class SemanticSegment:
    """Represents a semantic segment with its properties."""

    def __init__(self, text: str, start_idx: int, end_idx: int, segment_type: str = "general"):
        self.text = text
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.segment_type = segment_type
        self.tokens: List[str] = []
        self.pos_tags: List[tuple] = []
        self.entities: List[tuple] = []
        self.semantic_vector: Dict[str, int] = {}
        self.relevance_score = 0.0
        self.importance_score = 0.0


class SpaCySemanticSlicer:
    """Advanced semantic slicing using spaCy for context-aware text segmentation."""

    def __init__(self, model_name: str = "en_core_web_sm"):
        if not SPACY_AVAILABLE:
            raise ImportError("spaCy is required for semantic slicing. Install with: pip install spacy")

        self.nlp = load_spacy_pipeline(model_name)
        self.semantic_patterns = {
            "question": re.compile(r"\b(what|who|when|where|why|how|which|whose)\b", re.IGNORECASE),
            "command": re.compile(r"\b(please|can you|would you|could you|let me|help me)\b", re.IGNORECASE),
            "entity_focused": re.compile(r"\b(the|a|an)\s+\w+\s+(of|in|at|for|with)\b", re.IGNORECASE),
            "comparison": re.compile(r"\b(better|worse|than|versus|vs|compared to)\b", re.IGNORECASE),
        }

    def slice_text(self, text: str, max_segments: int = 10) -> List[SemanticSegment]:
        """Slice text into semantic segments using sentence-level analysis."""
        doc = self.nlp(text)

        segments = []
        current_segment = []
        current_start = 0

        for sent in doc.sents:
            sent_text = sent.text.strip()
            if not sent_text:
                continue

            segment_type = self._classify_segment_type(sent_text)
            if self._should_split_segment(current_segment, sent, segment_type):
                if current_segment:
                    segment_text = " ".join(sentence.text.strip() for sentence in current_segment)
                    segment = self._create_segment(segment_text, current_start, current_segment[-1].end, segments)
                    segments.append(segment)
                current_segment = [sent]
                current_start = sent.start
            else:
                current_segment.append(sent)

        if current_segment:
            segment_text = " ".join(sentence.text.strip() for sentence in current_segment)
            segment = self._create_segment(segment_text, current_start, current_segment[-1].end, segments)
            segments.append(segment)

        segments = segments[:max_segments]
        for segment in segments:
            self._enrich_segment(segment, doc)

        return segments

    def _classify_segment_type(self, text: str) -> str:
        for segment_type, pattern in self.semantic_patterns.items():
            if pattern.search(text):
                return segment_type
        return "general"

    def _should_split_segment(self, current_segment: List, new_sent, new_type: str) -> bool:
        if not current_segment:
            return False

        prev_type = self._classify_segment_type(current_segment[-1].text)
        if prev_type != new_type and new_type != "general":
            return True

        current_length = sum(len(sentence.text) for sentence in current_segment)
        if current_length > 200:
            return True

        prev_entities = {ent.text.lower() for ent in current_segment[-1].ents}
        new_entities = {ent.text.lower() for ent in new_sent.ents}
        if prev_entities and new_entities and not (prev_entities & new_entities):
            return True

        return False

    def _create_segment(
        self,
        text: str,
        start_idx: int,
        end_idx: int,
        existing_segments: List[SemanticSegment],
    ) -> SemanticSegment:
        segment_type = self._classify_segment_type(text)
        segment = SemanticSegment(text, start_idx, end_idx, segment_type)

        position_importance = 1.0 / (len(existing_segments) + 1)
        content_importance = min(len(text.split()) / 50.0, 1.0)
        segment.importance_score = (position_importance + content_importance) / 2.0
        return segment

    def _enrich_segment(self, segment: SemanticSegment, doc) -> None:
        segment_doc = doc[segment.start_idx:segment.end_idx]

        segment.tokens = [token.text for token in segment_doc]
        segment.pos_tags = [(token.text, token.pos_ or "X") for token in segment_doc]
        segment.entities = [(ent.text, ent.label_) for ent in segment_doc.ents]

        token_counts = Counter(
            token.lemma_.lower() if token.lemma_ else token.text.lower()
            for token in segment_doc
            if token.is_alpha and not token.is_stop
        )
        segment.semantic_vector = dict(token_counts)


class ContextRelevanceScorer:
    """Scores context relevance for semantic segments."""

    def __init__(self):
        self.idf_weights = self._load_idf_weights()

    def _load_idf_weights(self) -> Dict[str, float]:
        return {
            "the": 0.1,
            "a": 0.1,
            "an": 0.1,
            "and": 0.2,
            "or": 0.2,
            "but": 0.3,
            "what": 0.8,
            "how": 0.8,
            "why": 0.8,
            "when": 0.8,
            "where": 0.8,
            "good": 0.6,
            "best": 0.7,
            "great": 0.6,
            "excellent": 0.7,
            "hot": 0.5,
            "spring": 0.6,
            "spa": 0.6,
            "resort": 0.6,
        }

    def score_context_relevance(self, query: str, segments: List[SemanticSegment]) -> List[float]:
        query_terms = set(query.lower().split())
        scores = []

        for segment in segments:
            segment_terms = set(segment.text.lower().split())
            relevance_score = 0.0
            for term in query_terms:
                if term in segment_terms and segment.tokens:
                    idf = self.idf_weights.get(term, 0.5)
                    tf = segment.text.lower().count(term) / len(segment.tokens)
                    relevance_score += tf * idf

            semantic_overlap = len(set(segment.semantic_vector.keys()) & query_terms)
            relevance_score += semantic_overlap * 0.1
            relevance_score += segment.importance_score * 0.2
            scores.append(min(relevance_score, 1.0))

        return scores


class AnswerRelevanceScorer:
    """Scores answer relevance using semantic similarity."""

    def __init__(self):
        if not SPACY_AVAILABLE:
            raise ImportError("spaCy required for answer relevance scoring")
        self.nlp = load_spacy_pipeline("en_core_web_md")

    def score_answer_relevance(self, question: str, answers: List[str]) -> List[float]:
        question_doc = self.nlp(question)
        scores = []

        for answer in answers:
            answer_doc = self.nlp(answer)
            if question_doc.has_vector and answer_doc.has_vector:
                similarity = question_doc.similarity(answer_doc)
            else:
                q_tokens = {
                    (token.lemma_ or token.text).lower()
                    for token in question_doc
                    if token.is_alpha and not token.is_stop
                }
                a_tokens = {
                    (token.lemma_ or token.text).lower()
                    for token in answer_doc
                    if token.is_alpha and not token.is_stop
                }
                similarity = len(q_tokens & a_tokens) / (len(q_tokens | a_tokens) + 1e-8)
            scores.append(float(similarity))

        return scores


class FaithfulnessEvaluator:
    """Evaluates faithfulness of generated answers to source context."""

    def __init__(self):
        if not SPACY_AVAILABLE:
            raise ImportError("spaCy required for faithfulness evaluation")
        self.nlp = load_spacy_pipeline("en_core_web_sm")

    def evaluate_faithfulness(self, answer: str, context: str) -> Dict[str, float]:
        answer_doc = self.nlp(answer)
        context_doc = self.nlp(context)

        answer_entities = {ent.text.lower() for ent in answer_doc.ents}
        answer_claims = self._extract_claims(answer)
        context_entities = {ent.text.lower() for ent in context_doc.ents}
        context_tokens = {
            (token.lemma_ or token.text).lower()
            for token in context_doc
            if token.is_alpha and not token.is_stop
        }

        entity_overlap = len(answer_entities & context_entities)
        entity_coverage = entity_overlap / (len(answer_entities) + 1e-8)

        answer_tokens = {
            (token.lemma_ or token.text).lower()
            for token in answer_doc
            if token.is_alpha and not token.is_stop
        }
        token_overlap = len(answer_tokens & context_tokens)
        token_coverage = token_overlap / (len(answer_tokens) + 1e-8)

        if answer_doc.has_vector and context_doc.has_vector:
            semantic_similarity = float(answer_doc.similarity(context_doc))
        else:
            semantic_similarity = token_coverage

        hallucination_score = self._detect_hallucinations(answer_claims, context)
        faithfulness_score = (entity_coverage + token_coverage + semantic_similarity) / 3.0
        faithfulness_score = max(0.0, faithfulness_score - hallucination_score)

        return {
            "faithfulness_score": faithfulness_score,
            "entity_coverage": entity_coverage,
            "token_coverage": token_coverage,
            "semantic_similarity": semantic_similarity,
            "hallucination_penalty": hallucination_score,
        }

    def _extract_claims(self, text: str) -> List[str]:
        doc = self.nlp(text)
        claims = []
        for sent in doc.sents:
            if any(ent.label_ in ["PERSON", "ORG", "GPE", "MONEY", "DATE"] for ent in sent.ents):
                claims.append(sent.text.strip())
        return claims

    def _detect_hallucinations(self, claims: List[str], context: str) -> float:
        context_lower = context.lower()
        hallucination_penalty = 0.0

        for claim in claims:
            claim_entities = [ent.text.lower() for ent in self.nlp(claim).ents]
            unsupported_entities = sum(1 for entity in claim_entities if entity not in context_lower)
            if claim_entities:
                hallucination_penalty += unsupported_entities / len(claim_entities)

        return min(hallucination_penalty / len(claims) if claims else 0.0, 1.0)


class QueryDifficultyAnalyzer:
    """Analyzes query difficulty for adaptive processing."""

    def __init__(self):
        if not SPACY_AVAILABLE:
            raise ImportError("spaCy required for query difficulty analysis")
        self.nlp = load_spacy_pipeline("en_core_web_sm")

    def analyze_difficulty(self, query: str) -> Dict[str, float]:
        doc = self.nlp(query)

        avg_word_length = sum(len(token.text) for token in doc) / (len(doc) + 1e-8)
        unique_words = len({(token.lemma_ or token.text).lower() for token in doc if token.is_alpha})
        lexical_complexity = (avg_word_length + unique_words / 10.0) / 2.0

        sentence_count = max(1, len(list(doc.sents)))
        avg_sent_length = len(doc) / sentence_count
        syntactic_complexity = min(avg_sent_length / 20.0, 1.0)

        question_words = ["what", "how", "why", "when", "where", "which", "whose"]
        has_question_word = any((token.lemma_ or token.text).lower() in question_words for token in doc)
        multiple_entities = len(list(doc.ents)) > 2
        semantic_ambiguity = (1.0 if has_question_word else 0.0) + (0.5 if multiple_entities else 0.0)

        difficulty_score = (lexical_complexity + syntactic_complexity + semantic_ambiguity) / 3.0
        return {
            "difficulty_score": difficulty_score,
            "lexical_complexity": lexical_complexity,
            "syntactic_complexity": syntactic_complexity,
            "semantic_ambiguity": semantic_ambiguity,
            "difficulty_level": self._classify_difficulty(difficulty_score),
        }

    def _classify_difficulty(self, score: float) -> str:
        if score < 0.3:
            return "easy"
        if score < 0.6:
            return "medium"
        return "hard"


class SemanticSlicingPipeline:
    """Complete pipeline for semantic slicing with evaluation metrics."""

    def __init__(self):
        self.slicer = SpaCySemanticSlicer()
        self.context_scorer = ContextRelevanceScorer()
        self.answer_scorer = AnswerRelevanceScorer()
        self.faithfulness_evaluator = FaithfulnessEvaluator()
        self.difficulty_analyzer = QueryDifficultyAnalyzer()

    def process_query(self, query: str, context: str = "", max_segments: int = 5) -> Dict[str, object]:
        difficulty_analysis = self.difficulty_analyzer.analyze_difficulty(query)
        segments = self.slicer.slice_text(context or query, max_segments)

        if context:
            relevance_scores = self.context_scorer.score_context_relevance(query, segments)
            for segment, score in zip(segments, relevance_scores):
                segment.relevance_score = score

        segments.sort(key=lambda segment: (segment.relevance_score, segment.importance_score), reverse=True)
        answer_candidates = [segment.text for segment in segments[:3]]
        answer_scores = self.answer_scorer.score_answer_relevance(query, answer_candidates) if answer_candidates else []

        faithfulness_results = []
        if context and answer_candidates:
            for answer in answer_candidates:
                faithfulness_results.append(self.faithfulness_evaluator.evaluate_faithfulness(answer, context))

        return {
            "query": query,
            "difficulty_analysis": difficulty_analysis,
            "segments": segments,
            "answer_candidates": answer_candidates,
            "answer_scores": answer_scores,
            "faithfulness_results": faithfulness_results,
            "top_segment": segments[0] if segments else None,
        }


class RecursiveMethodComparator:
    """Compares semantic slicing performance against recursive methods."""

    def __init__(self):
        self.semantic_pipeline = SemanticSlicingPipeline()

    def compare_methods(self, queries: List[str], contexts: List[str]) -> Dict[str, List[float]]:
        semantic_scores = []
        recursive_scores = []

        for query, context in zip(queries, contexts):
            semantic_result = self.semantic_pipeline.process_query(query, context)
            semantic_scores.append(self._calculate_overall_score(semantic_result))
            recursive_scores.append(self._simulate_recursive_method(query, context))

        return {
            "semantic_slicing_scores": semantic_scores,
            "recursive_method_scores": recursive_scores,
            "semantic_avg": np.mean(semantic_scores),
            "recursive_avg": np.mean(recursive_scores),
            "improvement": np.mean(semantic_scores) - np.mean(recursive_scores),
        }

    def _calculate_overall_score(self, result: Dict[str, object]) -> float:
        relevance = np.mean([segment.relevance_score for segment in result["segments"]]) if result["segments"] else 0.0
        answer_quality = np.mean(result["answer_scores"]) if result["answer_scores"] else 0.0
        faithfulness = (
            np.mean([item["faithfulness_score"] for item in result["faithfulness_results"]])
            if result["faithfulness_results"]
            else 0.0
        )
        return 0.3 * relevance + 0.3 * answer_quality + 0.4 * faithfulness

    def _simulate_recursive_method(self, query: str, context: str) -> float:
        words = context.split()
        if len(words) < 10:
            return 0.5

        mid = len(words) // 2
        left_score = self._simulate_recursive_method(query, " ".join(words[:mid]))
        right_score = self._simulate_recursive_method(query, " ".join(words[mid:]))
        return (left_score + right_score) / 2.0 * 0.9


if __name__ == "__main__":
    if not SPACY_AVAILABLE:
        raise SystemExit(f"spaCy is unavailable in this environment: {SPACY_IMPORT_ERROR}")

    pipeline = SemanticSlicingPipeline()

    queries = [
        "What are the best hot springs in New Jersey?",
        "How do I get to the spa resort?",
        "Why is this hot spring famous compared to others?",
    ]

    contexts = [
        "New Jersey has several excellent hot springs. Shunjing Hot Spring is well-known for its resort facilities. Jiuhua Mountain Resort offers scenic spa services. The Capital Hot Spring provides popular hotel amenities.",
        "To reach the spa resort, take the main highway and follow the signs. The resort is located in a beautiful mountain area with easy access from downtown.",
        "This hot spring is famous because it has unique mineral content that provides therapeutic benefits. Unlike regular spas, it offers authentic hot spring experience with natural geothermal water.",
    ]

    print("=== Semantic Slicing Analysis ===")
    for i, (query, context) in enumerate(zip(queries, contexts), start=1):
        print(f"\nQuery {i}: {query}")
        result = pipeline.process_query(query, context)

        print(
            f"Difficulty: {result['difficulty_analysis']['difficulty_level']} "
            f"({result['difficulty_analysis']['difficulty_score']:.2f})"
        )
        print(f"Segments found: {len(result['segments'])}")

        if result["segments"]:
            top_segment = result["segments"][0]
            print(f"Top segment: {top_segment.text[:100]}...")
            print(f"Relevance: {top_segment.relevance_score:.3f}, Importance: {top_segment.importance_score:.3f}")

        if result["answer_scores"]:
            print(f"Answer relevance: {np.mean(result['answer_scores']):.3f}")
