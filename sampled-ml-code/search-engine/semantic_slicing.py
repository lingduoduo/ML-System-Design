import torch
import torch.nn as nn
from typing import List, Dict, Tuple, Optional
import numpy as np
from collections import Counter
import re


import spacy
from spacy.lang.en import English


SPACY_AVAILABLE = spacy is not None


class SemanticSegment:
    """Represents a semantic segment with its properties."""

    def __init__(self, text: str, start_idx: int, end_idx: int, segment_type: str = "general"):
        self.text = text
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.segment_type = segment_type
        self.tokens = []
        self.pos_tags = []
        self.entities = []
        self.semantic_vector = None
        self.relevance_score = 0.0
        self.importance_score = 0.0


class SpaCySemanticSlicer:
    """Advanced semantic slicing using SpaCy for context-aware text segmentation."""

    def __init__(self, model_name: str = "en_core_web_sm"):
        if not SPACY_AVAILABLE:
            raise ImportError("spaCy is required for semantic slicing. Install with: pip install spacy")

        try:
            self.nlp = spacy.load(model_name)
        except OSError:
            print(f"Model {model_name} not found. Downloading...")
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", model_name])
            self.nlp = spacy.load(model_name)

        # Configure sentence boundary detection
        self.nlp.add_pipe("sentencizer")

        # Semantic patterns for different segment types
        self.semantic_patterns = {
            "question": re.compile(r'\b(what|who|when|where|why|how|which|whose)\b', re.IGNORECASE),
            "command": re.compile(r'\b(please|can you|would you|could you|let me|help me)\b', re.IGNORECASE),
            "entity_focused": re.compile(r'\b(the|a|an)\s+\w+\s+(of|in|at|for|with)\b', re.IGNORECASE),
            "comparison": re.compile(r'\b(better|worse|than|versus|vs|compared to)\b', re.IGNORECASE),
        }

    def slice_text(self, text: str, max_segments: int = 10) -> List[SemanticSegment]:
        """Slice text into semantic segments using SpaCy analysis."""
        doc = self.nlp(text)

        segments = []
        current_segment = []
        current_start = 0

        for sent in doc.sents:
            # Analyze sentence semantics
            sent_text = sent.text.strip()
            if not sent_text:
                continue

            # Determine segment type
            segment_type = self._classify_segment_type(sent_text)

            # Check if we should start a new segment
            if self._should_split_segment(current_segment, sent, segment_type):
                if current_segment:
                    segment_text = " ".join([s.text for s in current_segment])
                    segment = self._create_segment(segment_text, current_start, sent.start, segments)
                    segments.append(segment)

                current_segment = [sent]
                current_start = sent.start
            else:
                current_segment.append(sent)

        # Add final segment
        if current_segment:
            segment_text = " ".join([s.text for s in current_segment])
            end_idx = current_segment[-1].end
            segment = self._create_segment(segment_text, current_start, end_idx, segments)
            segments.append(segment)

        # Limit number of segments and enrich with semantic features
        segments = segments[:max_segments]
        for segment in segments:
            self._enrich_segment(segment, doc)

        return segments

    def _classify_segment_type(self, text: str) -> str:
        """Classify segment type based on semantic patterns."""
        for segment_type, pattern in self.semantic_patterns.items():
            if pattern.search(text):
                return segment_type
        return "general"

    def _should_split_segment(self, current_segment: List, new_sent, new_type: str) -> bool:
        """Determine if we should start a new segment."""
        if not current_segment:
            return False

        # Split on topic changes
        if len(current_segment) > 0:
            prev_type = self._classify_segment_type(current_segment[-1].text)
            if prev_type != new_type and new_type != "general":
                return True

        # Split on sentence length (too long segments)
        current_length = sum(len(s.text) for s in current_segment)
        if current_length > 200:  # Characters
            return True

        # Split on semantic coherence (entity changes)
        if current_segment:
            prev_entities = set([ent.text.lower() for ent in current_segment[-1].ents])
            new_entities = set([ent.text.lower() for ent in new_sent.ents])
            if prev_entities and new_entities and not (prev_entities & new_entities):
                return True

        return False

    def _create_segment(self, text: str, start_idx: int, end_idx: int, existing_segments: List[SemanticSegment]) -> SemanticSegment:
        """Create a semantic segment with basic properties."""
        segment_type = self._classify_segment_type(text)
        segment = SemanticSegment(text, start_idx, end_idx, segment_type)

        # Calculate importance based on position and content
        position_importance = 1.0 / (len(existing_segments) + 1)  # Earlier segments more important
        content_importance = len(text.split()) / 50.0  # Length-based importance
        segment.importance_score = (position_importance + content_importance) / 2.0

        return segment

    def _enrich_segment(self, segment: SemanticSegment, doc):
        """Enrich segment with detailed linguistic features."""
        # Get tokens and POS tags for the segment span
        segment_doc = doc[segment.start_idx:segment.end_idx]

        segment.tokens = [token.text for token in segment_doc]
        segment.pos_tags = [(token.text, token.pos_) for token in segment_doc]
        segment.entities = [(ent.text, ent.label_) for ent in segment_doc.ents]

        # Calculate semantic vector (simple TF-IDF like representation)
        token_counts = Counter([token.lemma_.lower() for token in segment_doc
                               if not token.is_stop and token.is_alpha])
        segment.semantic_vector = dict(token_counts)


class ContextRelevanceScorer:
    """Scores context relevance for semantic segments."""

    def __init__(self):
        self.idf_weights = self._load_idf_weights()

    def _load_idf_weights(self) -> Dict[str, float]:
        """Load or compute IDF weights for common terms."""
        # Simplified IDF weights - in practice, compute from large corpus
        common_terms = {
            "the": 0.1, "a": 0.1, "an": 0.1, "and": 0.2, "or": 0.2, "but": 0.3,
            "what": 0.8, "how": 0.8, "why": 0.8, "when": 0.8, "where": 0.8,
            "good": 0.6, "best": 0.7, "great": 0.6, "excellent": 0.7,
            "hot": 0.5, "spring": 0.6, "spa": 0.6, "resort": 0.6
        }
        return common_terms

    def score_context_relevance(self, query: str, segments: List[SemanticSegment]) -> List[float]:
        """Score how relevant each segment is to the query context."""
        query_terms = set(query.lower().split())
        scores = []

        for segment in segments:
            segment_terms = set(segment.text.lower().split())
            term_overlap = len(query_terms & segment_terms)

            # TF-IDF style scoring
            relevance_score = 0.0
            for term in query_terms:
                if term in segment_terms:
                    idf = self.idf_weights.get(term, 0.5)
                    tf = segment.text.lower().count(term) / len(segment.tokens)
                    relevance_score += tf * idf

            # Boost for semantic matches
            if hasattr(segment, 'semantic_vector'):
                semantic_overlap = len(set(segment.semantic_vector.keys()) & query_terms)
                relevance_score += semantic_overlap * 0.1

            # Position and importance bonus
            relevance_score += segment.importance_score * 0.2

            scores.append(min(relevance_score, 1.0))  # Normalize to [0,1]

        return scores


class AnswerRelevanceScorer:
    """Scores answer relevance using semantic similarity."""

    def __init__(self):
        if not SPACY_AVAILABLE:
            raise ImportError("spaCy required for answer relevance scoring")

        try:
            self.nlp = spacy.load("en_core_web_md")  # Need medium model for word vectors
        except OSError:
            print("en_core_web_md not found, using en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")

    def score_answer_relevance(self, question: str, answers: List[str]) -> List[float]:
        """Score how relevant each answer is to the question."""
        question_doc = self.nlp(question)
        scores = []

        for answer in answers:
            answer_doc = self.nlp(answer)

            # Cosine similarity of document vectors
            if question_doc.has_vector and answer_doc.has_vector:
                similarity = question_doc.similarity(answer_doc)
            else:
                # Fallback to token overlap
                q_tokens = set([token.lemma_.lower() for token in question_doc
                               if not token.is_stop and token.is_alpha])
                a_tokens = set([token.lemma_.lower() for token in answer_doc
                               if not token.is_stop and token.is_alpha])
                similarity = len(q_tokens & a_tokens) / (len(q_tokens | a_tokens) + 1e-8)

            scores.append(float(similarity))

        return scores


class FaithfulnessEvaluator:
    """Evaluates faithfulness of generated answers to source context."""

    def __init__(self):
        if not SPACY_AVAILABLE:
            raise ImportError("spaCy required for faithfulness evaluation")

        self.nlp = spacy.load("en_core_web_sm")

    def evaluate_faithfulness(self, answer: str, context: str) -> Dict[str, float]:
        """Evaluate how faithful the answer is to the provided context."""
        answer_doc = self.nlp(answer)
        context_doc = self.nlp(context)

        # Extract key facts from answer
        answer_entities = set([ent.text.lower() for ent in answer_doc.ents])
        answer_claims = self._extract_claims(answer)

        # Check against context
        context_entities = set([ent.text.lower() for ent in context_doc.ents])
        context_tokens = set([token.lemma_.lower() for token in context_doc
                             if not token.is_stop and token.is_alpha])

        # Entity consistency
        entity_overlap = len(answer_entities & context_entities)
        entity_coverage = entity_overlap / (len(answer_entities) + 1e-8)

        # Token overlap (factual consistency)
        answer_tokens = set([token.lemma_.lower() for token in answer_doc
                            if not token.is_stop and token.is_alpha])
        token_overlap = len(answer_tokens & context_tokens)
        token_coverage = token_overlap / (len(answer_tokens) + 1e-8)

        # Semantic similarity
        if answer_doc.has_vector and context_doc.has_vector:
            semantic_similarity = answer_doc.similarity(context_doc)
        else:
            semantic_similarity = token_coverage

        # Hallucination detection (claims not supported by context)
        hallucination_score = self._detect_hallucinations(answer_claims, context)

        faithfulness_score = (entity_coverage + token_coverage + semantic_similarity) / 3.0
        faithfulness_score = max(0.0, faithfulness_score - hallucination_score)

        return {
            "faithfulness_score": faithfulness_score,
            "entity_coverage": entity_coverage,
            "token_coverage": token_coverage,
            "semantic_similarity": semantic_similarity,
            "hallucination_penalty": hallucination_score
        }

    def _extract_claims(self, text: str) -> List[str]:
        """Extract factual claims from text."""
        doc = self.nlp(text)
        claims = []

        for sent in doc.sents:
            # Simple claim extraction - sentences with entities
            if any(ent.label_ in ['PERSON', 'ORG', 'GPE', 'MONEY', 'DATE'] for ent in sent.ents):
                claims.append(sent.text.strip())

        return claims

    def _detect_hallucinations(self, claims: List[str], context: str) -> float:
        """Detect potential hallucinations in claims."""
        context_lower = context.lower()
        hallucination_penalty = 0.0

        for claim in claims:
            claim_lower = claim.lower()
            # Check if key entities/claims appear in context
            claim_entities = [ent.text.lower() for ent in self.nlp(claim).ents]
            unsupported_entities = 0

            for entity in claim_entities:
                if entity not in context_lower:
                    unsupported_entities += 1

            if claim_entities:
                hallucination_penalty += unsupported_entities / len(claim_entities)

        return min(hallucination_penalty / len(claims) if claims else 0.0, 1.0)


class QueryDifficultyAnalyzer:
    """Analyzes query difficulty for adaptive processing."""

    def __init__(self):
        if not SPACY_AVAILABLE:
            raise ImportError("spaCy required for query difficulty analysis")

        self.nlp = spacy.load("en_core_web_sm")

    def analyze_difficulty(self, query: str) -> Dict[str, float]:
        """Analyze query difficulty across multiple dimensions."""
        doc = self.nlp(query)

        # Lexical complexity
        avg_word_length = sum(len(token.text) for token in doc) / (len(doc) + 1e-8)
        unique_words = len(set(token.lemma_.lower() for token in doc if token.is_alpha))
        lexical_complexity = (avg_word_length + unique_words / 10.0) / 2.0

        # Syntactic complexity
        avg_sent_length = len(doc) / max(1, len(list(doc.sents)))
        syntactic_complexity = min(avg_sent_length / 20.0, 1.0)

        # Semantic ambiguity
        question_words = ['what', 'how', 'why', 'when', 'where', 'which', 'whose']
        has_question_word = any(token.lemma_.lower() in question_words for token in doc)
        multiple_entities = len([ent for ent in doc.ents]) > 2
        semantic_ambiguity = (1.0 if has_question_word else 0.0) + (0.5 if multiple_entities else 0.0)

        # Overall difficulty score
        difficulty_score = (lexical_complexity + syntactic_complexity + semantic_ambiguity) / 3.0

        return {
            "difficulty_score": difficulty_score,
            "lexical_complexity": lexical_complexity,
            "syntactic_complexity": syntactic_complexity,
            "semantic_ambiguity": semantic_ambiguity,
            "difficulty_level": self._classify_difficulty(difficulty_score)
        }

    def _classify_difficulty(self, score: float) -> str:
        """Classify query difficulty level."""
        if score < 0.3:
            return "easy"
        elif score < 0.6:
            return "medium"
        else:
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
        """Process query with semantic slicing and comprehensive evaluation."""
        # Analyze query difficulty
        difficulty_analysis = self.difficulty_analyzer.analyze_difficulty(query)

        # Slice text into semantic segments
        segments = self.slicer.slice_text(context or query, max_segments)

        # Score context relevance
        if context:
            relevance_scores = self.context_scorer.score_context_relevance(query, segments)
            for segment, score in zip(segments, relevance_scores):
                segment.relevance_score = score

        # Sort segments by relevance and importance
        segments.sort(key=lambda s: (s.relevance_score, s.importance_score), reverse=True)

        # Generate answer candidates from top segments
        answer_candidates = [seg.text for seg in segments[:3]]

        # Score answer relevance
        if answer_candidates:
            answer_scores = self.answer_scorer.score_answer_relevance(query, answer_candidates)
        else:
            answer_scores = []

        # Evaluate faithfulness if we have context
        faithfulness_results = []
        if context and answer_candidates:
            for answer in answer_candidates:
                faith_eval = self.faithfulness_evaluator.evaluate_faithfulness(answer, context)
                faithfulness_results.append(faith_eval)

        return {
            "query": query,
            "difficulty_analysis": difficulty_analysis,
            "segments": segments,
            "answer_candidates": answer_candidates,
            "answer_scores": answer_scores,
            "faithfulness_results": faithfulness_results,
            "top_segment": segments[0] if segments else None
        }


class RecursiveMethodComparator:
    """Compares semantic slicing performance against recursive methods."""

    def __init__(self):
        self.semantic_pipeline = SemanticSlicingPipeline()

    def compare_methods(self, queries: List[str], contexts: List[str]) -> Dict[str, List[float]]:
        """Compare semantic slicing vs recursive methods across queries."""
        semantic_scores = []
        recursive_scores = []

        for query, context in zip(queries, contexts):
            # Semantic slicing approach
            semantic_result = self.semantic_pipeline.process_query(query, context)
            semantic_score = self._calculate_overall_score(semantic_result)
            semantic_scores.append(semantic_score)

            # Recursive approach (simplified simulation)
            recursive_score = self._simulate_recursive_method(query, context)
            recursive_scores.append(recursive_score)

        return {
            "semantic_slicing_scores": semantic_scores,
            "recursive_method_scores": recursive_scores,
            "semantic_avg": np.mean(semantic_scores),
            "recursive_avg": np.mean(recursive_scores),
            "improvement": np.mean(semantic_scores) - np.mean(recursive_scores)
        }

    def _calculate_overall_score(self, result: Dict[str, object]) -> float:
        """Calculate overall performance score for semantic slicing result."""
        difficulty = result["difficulty_analysis"]["difficulty_score"]
        relevance = np.mean([s.relevance_score for s in result["segments"]]) if result["segments"] else 0.0
        answer_quality = np.mean(result["answer_scores"]) if result["answer_scores"] else 0.0
        faithfulness = np.mean([r["faithfulness_score"] for r in result["faithfulness_results"]]) if result["faithfulness_results"] else 0.0

        # Weighted combination
        score = (0.3 * relevance + 0.3 * answer_quality + 0.4 * faithfulness)
        return score

    def _simulate_recursive_method(self, query: str, context: str) -> float:
        """Simulate recursive method performance (simplified)."""
        # Simple recursive text splitting simulation
        words = context.split()
        if len(words) < 10:
            return 0.5  # Base case

        # Recursive split
        mid = len(words) // 2
        left_score = self._simulate_recursive_method(query, " ".join(words[:mid]))
        right_score = self._simulate_recursive_method(query, " ".join(words[mid:]))

        # Combine with some degradation
        return (left_score + right_score) / 2.0 * 0.9


if __name__ == "__main__":
    # Example usage
    pipeline = SemanticSlicingPipeline()

    queries = [
        "What are the best hot springs in New Jersey?",
        "How do I get to the spa resort?",
        "Why is this hot spring famous compared to others?"
    ]

    contexts = [
        "New Jersey has several excellent hot springs. Shunjing Hot Spring is well-known for its resort facilities. Jiuhua Mountain Resort offers scenic spa services. The Capital Hot Spring provides popular hotel amenities.",
        "To reach the spa resort, take the main highway and follow the signs. The resort is located in a beautiful mountain area with easy access from downtown.",
        "This hot spring is famous because it has unique mineral content that provides therapeutic benefits. Unlike regular spas, it offers authentic hot spring experience with natural geothermal water."
    ]

    print("=== Semantic Slicing Analysis ===")
    for i, (query, context) in enumerate(zip(queries, contexts)):
        print(f"\nQuery {i+1}: {query}")
        result = pipeline.process_query(query, context)

        print(f"Difficulty: {result['difficulty_analysis']['difficulty_level']} ({result['difficulty_analysis']['difficulty_score']:.2f})")
        print(f"Segments found: {len(result['segments'])}")

        if result['segments']:
            top_segment = result['segments'][0]
            print(f"Top segment: {top_segment.text[:100]}...")
            print(f"Relevance: {top_segment.relevance_score:.3f}, Importance: {top_segment.importance_score:.3f}")

        if result['answer_scores']:
            print(f"Answer relevance: {np.mean(result['answer_scores']):.3f}")

        if result['faithfulness_results']:
            avg_faithfulness = np.mean([r['faithfulness_score'] for r in result['faithfulness_results']])
            print(f"Average faithfulness: {avg_faithfulness:.3f}")

    # Compare with recursive methods
    print("\n=== Method Comparison ===")
    comparator = RecursiveMethodComparator()
    comparison = comparator.compare_methods(queries, contexts)

    print(f"Semantic Slicing Average: {comparison['semantic_avg']:.3f}")
    print(f"Recursive Method Average: {comparison['recursive_avg']:.3f}")
    print(f"Improvement: {comparison['improvement']:.3f}")
