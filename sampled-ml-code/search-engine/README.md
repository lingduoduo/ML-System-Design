# ML-Based Search Engine System

A comprehensive, production-ready search engine implementation with query understanding, multi-stage retrieval, and neural ranking. This system combines traditional information retrieval techniques (BM25) with modern deep learning approaches (LTR, cross-encoders) for optimal search quality.

## Table of Contents

1. [System Architecture](#system-architecture)
2. [Dependencies](#dependencies)
3. [Code Overview](#code-overview)
4. [Module Details](#module-details)
5. [Usage Examples](#usage-examples)

---

## System Architecture

```
Query Input
    ↓
Query Understanding (preprocessing, tokenization, NER, intent detection)
    ↓
Search Recall (main pool → backup pool → fallback)
    ↓
Retrieval (BM25 sparse + dense embedding)
    ↓
Ranking (LTR: pointwise & pairwise)
    ↓
Reranking (cross-encoder, dense semantic, hybrid)
    ↓
Final Results
```

---

## Dependencies

```
torch>=1.9.0
spacy>=3.0.0
nltk>=3.6
numpy>=1.19.0
```

Install models:
```bash
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('punkt'); nltk.download('averaged_perceptron_tagger'); nltk.download('wordnet'); nltk.download('stopwords')"
```

---

## Code Overview

### 1. search_engine.py - Query Preprocessing
Legacy base module for query preprocessing and initial analysis.

```python
import re
import unicodedata
from collections import Counter
from typing import List, Tuple, Dict

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.util import ngrams

try:
    import spacy
except ImportError:
    spacy = None

SPACY_AVAILABLE = spacy is not None

class SearchQueryProcessor:
    EMOJI_PATTERN = re.compile(
        "["
        "\U0001F600-\U0001F64F"
        "\U0001F300-\U0001F5FF"
        "\U0001F680-\U0001F6FF"
        "\U0001F1E0-\U0001F1FF"
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "]+",
        flags=re.UNICODE,
    )
    PUNCTUATION_PATTERN = re.compile(r"[^\w\s]+")
    WHITESPACE_PATTERN = re.compile(r"\s+")
    DATE_PATTERN = re.compile(r"\d{4}[/-]\d{1,2}[/-]\d{1,2}|\d{1,2}:\d{2}")
    NUMBER_PATTERN = re.compile(r"\d+\.?\d*")

    def __init__(self, language: str = "en"):
        self.language = language
        self.stop_words = set(stopwords.words("english"))
        self.lemmatizer = WordNetLemmatizer()
        self.intent_keywords = {
            "purchase": ["buy", "purchase", "order", "shop", "checkout"],
            "navigate": ["map", "directions", "navigate", "route", "go to"],
            "qa": ["what", "who", "when", "how", "where", "why"],
            "recommendation": ["recommend", "suggest", "advice", "best"],
        }
        self.synonym_map = {
            "tv": ["television", "smart tv"],
            "notebook": ["laptop", "ultrabook"],
            "spa": ["hot spring", "resort"],
            "phone": ["smartphone", "mobile"],
        }
        self.click_log = Counter()
        self.idf = Counter()
        self.spacy_nlp = self._load_spacy_model() if SPACY_AVAILABLE else None

    def _load_spacy_model(self):
        try:
            return spacy.load("en_core_web_sm")
        except OSError:
            return None

    def normalize_whitespace(self, text: str) -> str:
        return self.WHITESPACE_PATTERN.sub(" ", text).strip()

    def remove_invalid_characters(self, text: str) -> str:
        return "".join(ch for ch in text if not unicodedata.category(ch).startswith(("C", "S")))

    def remove_emojis(self, text: str) -> str:
        return self.EMOJI_PATTERN.sub("", text)

    def lowercase(self, text: str) -> str:
        return text.lower()

    def remove_punctuation(self, text: str) -> str:
        return self.PUNCTUATION_PATTERN.sub("", text)

    def truncate_text(self, text: str, max_chars: int) -> str:
        return text if max_chars is None else text[:max_chars]

    def preprocess(self, text: str, max_chars: int = 64) -> str:
        text = text or ""
        text = self.remove_emojis(text)
        text = self.remove_invalid_characters(text)
        text = self.normalize_whitespace(text)
        text = self.lowercase(text)
        text = self.remove_punctuation(text)
        return self.truncate_text(text, max_chars)

    def tokenize(self, text: str) -> List[str]:
        if self.spacy_nlp:
            doc = self.spacy_nlp(text)
            return [token.text for token in doc if not token.is_space]
        return nltk.word_tokenize(text)

    def pos_tag(self, tokens: List[str]) -> List[Tuple[str, str]]:
        if self.spacy_nlp:
            doc = self.spacy_nlp(" ".join(tokens))
            return [(token.text, token.pos_) for token in doc if not token.is_space]
        return nltk.pos_tag(tokens)

    def named_entities(self, text: str) -> List[Tuple[str, str]]:
        if self.spacy_nlp:
            doc = self.spacy_nlp(text)
            return [(ent.text, ent.label_) for ent in doc.ents]
        entities = [(item, "DATE") for item in self.DATE_PATTERN.findall(text)]
        entities.extend((item, "NUMBER") for item in self.NUMBER_PATTERN.findall(text))
        return entities

    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        return [token for token in tokens if token.lower() not in self.stop_words]

    def lemmatize(self, tokens: List[str]) -> List[str]:
        if self.spacy_nlp:
            doc = self.spacy_nlp(" ".join(tokens))
            return [token.lemma_ for token in doc if not token.is_space]
        return [self.lemmatizer.lemmatize(token) for token in tokens]

    def generate_ngrams(self, tokens: List[str], n: int = 2) -> List[Tuple[str, ...]]:
        return list(ngrams(tokens, n))

    def calculate_term_weights(self, tokens: List[str]) -> Dict[str, float]:
        return {token: float(self.click_log.get(token, 1)) * self.idf.get(token, 1.0) for token in tokens}

    def update_click_log(self, clicked_terms: List[str]) -> None:
        self.click_log.update(clicked_terms)

    def set_idf(self, idf_mapping: Dict[str, float]) -> None:
        self.idf.update(idf_mapping)

    def vocabulary(self) -> set:
        synonym_terms = {term for values in self.synonym_map.values() for term in values}
        return set(self.synonym_map.keys()) | synonym_terms | self.stop_words

    def expand_synonyms(self, tokens: List[str]) -> List[str]:
        expanded = list(tokens)
        for token in tokens:
            expanded.extend(self.synonym_map.get(token, []))
        return list(dict.fromkeys(expanded))

    def detect_intents(self, text: str) -> List[Tuple[str, float]]:
        lower_text = text.lower()
        scores = [
            (intent, sum(keyword in lower_text for keyword in keywords) / len(keywords))
            for intent, keywords in self.intent_keywords.items()
            if any(keyword in lower_text for keyword in keywords)
        ]
        return sorted(scores, key=lambda item: item[1], reverse=True)

    def recognize_entities(self, text: str) -> List[Tuple[str, str]]:
        entities = self.named_entities(text)
        if not entities and "beijing" in text.lower():
            entities.append(("Beijing", "GPE"))
        return entities

    def simple_error_correction(self, tokens: List[str]) -> List[str]:
        vocabulary = self.vocabulary()
        corrected = []
        for token in tokens:
            lower_token = token.lower()
            if lower_token in vocabulary or len(token) <= 1:
                corrected.append(token)
                continue
            candidates = [word for word in vocabulary if abs(len(word) - len(lower_token)) <= 2]
            if not candidates:
                corrected.append(token)
                continue
            best = min(candidates, key=lambda cand: nltk.edit_distance(lower_token, cand.lower()))
            corrected.append(best if nltk.edit_distance(lower_token, best.lower()) <= 2 else token)
        return corrected

    def generate_drop_candidates(self, tokens: List[str], term_weights: Dict[str, float]) -> List[str]:
        if len(tokens) <= 1:
            return [" ".join(tokens)]

        sorted_tokens = sorted(tokens, key=lambda token: term_weights.get(token, 0.0))
        candidates = [" ".join(tokens)]
        for k in range(1, min(3, len(sorted_tokens))):
            dropped = [token for token in tokens if token not in set(sorted_tokens[:k])]
            if dropped:
                candidates.append(" ".join(dropped))
        return list(dict.fromkeys(candidates))

    def rewrite_query(self, tokens: List[str], term_weights: Dict[str, float]) -> List[str]:
        candidates = [" ".join(tokens)]
        candidates.extend(self.generate_drop_candidates(tokens, term_weights))
        candidates.append(" ".join(self.expand_synonyms(tokens)))
        return [query for query in dict.fromkeys(candidates) if query]

    def process_query(self, text: str, max_chars: int = 64) -> Dict[str, object]:
        raw = self.preprocess(text, max_chars=max_chars)
        tokens = self.tokenize(raw)
        corrected_tokens = self.simple_error_correction(tokens)
        filtered_tokens = self.remove_stopwords(corrected_tokens)
        lemmas = self.lemmatize(filtered_tokens)
        pos_tags = self.pos_tag(filtered_tokens)
        term_weights = self.calculate_term_weights(lemmas)
        intents = self.detect_intents(raw)
        entities = self.recognize_entities(raw)
        rewrites = self.rewrite_query(lemmas, term_weights)
        bigrams = self.generate_ngrams(lemmas, 2)

        return {
            "raw": raw,
            "tokens": tokens,
            "corrected_tokens": corrected_tokens,
            "filtered_tokens": filtered_tokens,
            "lemmas": lemmas,
            "pos_tags": pos_tags,
            "term_weights": term_weights,
            "intents": intents,
            "entities": entities,
            "rewrites": rewrites,
            "bigrams": bigrams,
        }


if __name__ == "__main__":
    processor = SearchQueryProcessor()
    processor.set_idf({"beijing": 2.5, "hot": 1.5, "spring": 1.8, "famous": 1.2})
    processor.update_click_log(["beijing", "hot", "spring"])
    result = processor.process_query("Beijing famous hot spring 😊 I want to see scenery", max_chars=64)
    for key, value in result.items():
        print(f"{key}: {value}")
```

---

### 2. query_understanding.py - Enhanced Query Processing
Advanced query understanding module with chunk analysis and intent detection.

```python
import re
import unicodedata
from collections import Counter
from typing import List, Tuple, Dict

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.util import ngrams

try:
    import spacy
except ImportError:
    spacy = None

SPACY_AVAILABLE = spacy is not None

class SearchQueryProcessor:
    EMOJI_PATTERN = re.compile(
        "["
        "\U0001F600-\U0001F64F"
        "\U0001F300-\U0001F5FF"
        "\U0001F680-\U0001F6FF"
        "\U0001F1E0-\U0001F1FF"
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "]+",
        flags=re.UNICODE,
    )
    PUNCTUATION_PATTERN = re.compile(r"[^\w\s]+")
    WHITESPACE_PATTERN = re.compile(r"\s+")
    DATE_PATTERN = re.compile(r"\d{4}[/-]\d{1,2}[/-]\d{1,2}|\d{1,2}:\d{2}")
    NUMBER_PATTERN = re.compile(r"\d+\.?\d*")

    def __init__(self, language: str = "en"):
        self.language = language
        self.stop_words = set(stopwords.words("english"))
        self.lemmatizer = WordNetLemmatizer()
        self.intent_keywords = {
            "purchase": ["buy", "purchase", "order", "shop", "checkout"],
            "navigate": ["map", "directions", "navigate", "route", "go to"],
            "qa": ["what", "who", "when", "how", "where", "why"],
            "recommendation": ["recommend", "suggest", "advice", "best"],
        }
        self.synonym_map = {
            "tv": ["television", "smart tv"],
            "notebook": ["laptop", "ultrabook"],
            "spa": ["hot spring", "resort"],
            "phone": ["smartphone", "mobile"],
        }
        self.click_log = Counter()
        self.idf = Counter()
        self.spacy_nlp = self._load_spacy_model() if SPACY_AVAILABLE else None

    def _load_spacy_model(self):
        try:
            return spacy.load("en_core_web_sm")
        except OSError:
            return None

    def normalize_whitespace(self, text: str) -> str:
        return self.WHITESPACE_PATTERN.sub(" ", text).strip()

    def remove_invalid_characters(self, text: str) -> str:
        return "".join(ch for ch in text if not unicodedata.category(ch).startswith(("C", "S")))

    def remove_emojis(self, text: str) -> str:
        return self.EMOJI_PATTERN.sub("", text)

    def lowercase(self, text: str) -> str:
        return text.lower()

    def remove_punctuation(self, text: str) -> str:
        return self.PUNCTUATION_PATTERN.sub("", text)

    def preprocess(self, text: str, max_chars: int = 64) -> str:
        text = text or ""
        text = self.remove_emojis(text)
        text = self.remove_invalid_characters(text)
        text = self.normalize_whitespace(text)
        text = self.lowercase(text)
        text = self.remove_punctuation(text)
        return self.truncate_text(text, max_chars)

    def truncate_text(self, text: str, max_chars: int) -> str:
        return text if max_chars is None else text[:max_chars]

    def tokenize(self, text: str) -> List[str]:
        if self.spacy_nlp:
            doc = self.spacy_nlp(text)
            return [token.text for token in doc if not token.is_space]
        return nltk.word_tokenize(text)

    def pos_tag(self, tokens: List[str]) -> List[Tuple[str, str]]:
        if self.spacy_nlp:
            doc = self.spacy_nlp(" ".join(tokens))
            return [(token.text, token.pos_) for token in doc if not token.is_space]
        return nltk.pos_tag(tokens)

    def named_entities(self, text: str) -> List[Tuple[str, str]]:
        if self.spacy_nlp:
            doc = self.spacy_nlp(text)
            return [(ent.text, ent.label_) for ent in doc.ents]
        entities = [(item, "DATE") for item in self.DATE_PATTERN.findall(text)]
        entities.extend((item, "NUMBER") for item in self.NUMBER_PATTERN.findall(text))
        return entities

    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        return [token for token in tokens if token.lower() not in self.stop_words]

    def lemmatize(self, tokens: List[str]) -> List[str]:
        if self.spacy_nlp:
            doc = self.spacy_nlp(" ".join(tokens))
            return [token.lemma_ for token in doc if not token.is_space]
        return [self.lemmatizer.lemmatize(token) for token in tokens]

    def generate_ngrams(self, tokens: List[str], n: int = 2) -> List[Tuple[str, ...]]:
        return list(ngrams(tokens, n))

    def calculate_term_weights(self, tokens: List[str]) -> Dict[str, float]:
        return {token: float(self.click_log.get(token, 1)) * self.idf.get(token, 1.0) for token in tokens}

    def update_click_log(self, clicked_terms: List[str]) -> None:
        self.click_log.update(clicked_terms)

    def set_idf(self, idf_mapping: Dict[str, float]) -> None:
        self.idf.update(idf_mapping)

    def vocabulary(self) -> set:
        synonym_terms = {term for values in self.synonym_map.values() for term in values}
        return set(self.synonym_map.keys()) | synonym_terms | self.stop_words

    def expand_synonyms(self, tokens: List[str]) -> List[str]:
        expanded = list(tokens)
        for token in tokens:
            expanded.extend(self.synonym_map.get(token, []))
        return list(dict.fromkeys(expanded))

    def analyze_chunks(self, tokens: List[str], pos_tags: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        location_words = {"beijing", "shanghai", "new york", "london"}
        chunks = []
        for token, pos in pos_tags:
            lower_token = token.lower()
            if lower_token in location_words or pos == "PROPN":
                chunks.append((token, "LOCATION"))
            elif pos == "ADJ":
                chunks.append((token, "MODIFIER"))
            elif pos in {"NOUN", "PROPN"}:
                chunks.append((token, "CATEGORY"))
            else:
                chunks.append((token, "OTHER"))
        return chunks

    def detect_category_intent(self, chunk_tags: List[Tuple[str, str]]) -> str:
        categories = [tag for _, tag in chunk_tags if tag == "CATEGORY"]
        modifiers = [tag for _, tag in chunk_tags if tag == "MODIFIER"]
        if categories and modifiers:
            return "category_search"
        if categories:
            return "exact_search"
        return "keyword_search"

    def detect_intents(self, text: str, tokens: List[str]) -> List[Tuple[str, float]]:
        lower_text = text.lower()
        scores = [
            (intent, sum(keyword in lower_text for keyword in keywords) / len(keywords))
            for intent, keywords in self.intent_keywords.items()
            if any(keyword in lower_text for keyword in keywords)
        ]
        return sorted(scores, key=lambda item: item[1], reverse=True)

    def recognize_entities(self, text: str) -> List[Tuple[str, str]]:
        entities = self.named_entities(text)
        if not entities and "beijing" in text.lower():
            entities.append(("Beijing", "GPE"))
        return entities

    def simple_error_correction(self, tokens: List[str]) -> List[str]:
        vocabulary = self.vocabulary()
        corrected = []
        for token in tokens:
            lower_token = token.lower()
            if lower_token in vocabulary or len(token) <= 1:
                corrected.append(token)
                continue
            candidates = [word for word in vocabulary if abs(len(word) - len(lower_token)) <= 2]
            if not candidates:
                corrected.append(token)
                continue
            best = min(candidates, key=lambda cand: nltk.edit_distance(lower_token, cand.lower()))
            corrected.append(best if nltk.edit_distance(lower_token, best.lower()) <= 2 else token)
        return corrected

    def generate_drop_candidates(self, tokens: List[str], term_weights: Dict[str, float]) -> List[str]:
        if len(tokens) <= 1:
            return [" ".join(tokens)]

        sorted_tokens = sorted(tokens, key=lambda token: term_weights.get(token, 0.0))
        candidates = [" ".join(tokens)]
        for k in range(1, min(3, len(sorted_tokens))):
            dropped = [token for token in tokens if token not in set(sorted_tokens[:k])]
            if dropped:
                candidates.append(" ".join(dropped))
        return list(dict.fromkeys(candidates))

    def rewrite_query(self, tokens: List[str], term_weights: Dict[str, float]) -> List[str]:
        candidates = [" ".join(tokens)]
        candidates.extend(self.generate_drop_candidates(tokens, term_weights))
        candidates.append(" ".join(self.expand_synonyms(tokens)))
        return [query for query in dict.fromkeys(candidates) if query]

    def process_query(self, text: str, max_chars: int = 64) -> Dict[str, object]:
        raw = self.preprocess(text, max_chars=max_chars)
        tokens = self.tokenize(raw)
        corrected_tokens = self.simple_error_correction(tokens)
        filtered_tokens = self.remove_stopwords(corrected_tokens)
        lemmas = self.lemmatize(filtered_tokens)
        pos_tags = self.pos_tag(filtered_tokens)
        term_weights = self.calculate_term_weights(lemmas)
        intents = self.detect_intents(raw, tokens)
        entities = self.recognize_entities(raw)
        rewrites = self.rewrite_query(lemmas, term_weights)
        bigrams = self.generate_ngrams(lemmas, 2)

        return {
            "raw": raw,
            "tokens": tokens,
            "corrected_tokens": corrected_tokens,
            "filtered_tokens": filtered_tokens,
            "lemmas": lemmas,
            "pos_tags": pos_tags,
            "term_weights": term_weights,
            "intents": intents,
            "entities": entities,
            "rewrites": rewrites,
            "bigrams": bigrams,
        }


if __name__ == "__main__":
    processor = SearchQueryProcessor()
    result = processor.process_query("Beijing famous hot spring 😊 I want to see scenery", max_chars=64)
    for key, value in result.items():
        print(f"{key}: {value}")
```

---

### 3. search_recall.py - Multi-Stage Retrieval
Online search recall engine with multi-stage fallback strategy.

```python
from typing import List, Dict, Tuple

from query_understanding import SearchQueryProcessor


class SearchRecallEngine:
    def __init__(self, language: str = "en"):
        self.processor = SearchQueryProcessor(language=language)
        self.main_poi_index: List[Dict[str, object]] = []
        self.backup_poi_index: List[Dict[str, object]] = []
        self.initialize_poi_index()

    def initialize_poi_index(self, poi_index: List[Dict[str, object]] = None) -> None:
        poi_index = poi_index if poi_index is not None else self._default_poi_index()
        self.main_poi_index = [poi for poi in poi_index if poi.get("partner", False)]
        self.backup_poi_index = [poi for poi in poi_index if not poi.get("partner", False)]

    def _default_poi_index(self) -> List[Dict[str, object]]:
        return [
            {"id": 1, "name": "Shunjing Hot Spring", "category": "Hot Spring", "city": "New Jersey", "partner": True, "description": "well-known resort"},
            {"id": 2, "name": "Jiuhua Mountain Resort", "category": "Hot Spring", "city": "New Jersey", "partner": True, "description": "scenic spa resort"},
            {"id": 3, "name": "City Spa Center", "category": "Spa", "city": "New York", "partner": False, "description": "relaxing services"},
            {"id": 4, "name": "Capital Hot Spring", "category": "Hot Spring", "city": "New York", "partner": False, "description": "popular hotel"},
        ]

    def _field_match_score(self, poi: Dict[str, object], terms: List[str]) -> float:
        field_weights = {
            "name": 3.0,
            "category": 2.5,
            "city": 2.0,
            "description": 1.0,
        }
        source = {field: str(poi.get(field, "")).lower() for field in field_weights}
        return sum(
            weight
            for term in terms
            for field, weight in field_weights.items()
            if term.lower() in source[field]
        )

    def _search_pool(self, terms: List[str], pool: List[Dict[str, object]]) -> List[Dict[str, object]]:
        scored = [(self._field_match_score(poi, terms), poi) for poi in pool]
        scored = [(score, poi) for score, poi in scored if score > 0.0]
        return [poi for score, poi in sorted(scored, key=lambda item: item[0], reverse=True)]

    def _drop_modifiers(self, lemmas: List[str], chunk_tags: List[Tuple[str, str]]) -> List[str]:
        modifiers = {token.lower() for token, tag in chunk_tags if tag == "MODIFIER"}
        return [term for term in lemmas if term.lower() not in modifiers]

    def _expand_search_terms(self, lemmas: List[str]) -> List[str]:
        return list(dict.fromkeys(lemmas + self.processor.expand_synonyms(lemmas)))

    def _build_recall_result(
        self,
        main_recall: List[Dict[str, object]],
        backup_recall: List[Dict[str, object]],
        final_recall: List[Dict[str, object]],
        recall_steps: List[str],
    ) -> Dict[str, object]:
        return {
            "main_recall": main_recall,
            "backup_recall": backup_recall,
            "final_recall": final_recall,
            "recall_steps": recall_steps,
        }

    def recall(self, query: str, max_chars: int = 64) -> Dict[str, object]:
        query_info = self.processor.process_query(query, max_chars=max_chars)
        query_info["chunk_tags"] = self.processor.analyze_chunks(query_info["tokens"], query_info["pos_tags"])
        query_info["category_intent"] = self.processor.detect_category_intent(query_info["chunk_tags"])
        query_info["search_terms"] = self._expand_search_terms(query_info["lemmas"])

        recall_steps: List[str] = []
        main_hits = self._search_pool(query_info["search_terms"], self.main_poi_index)
        recall_steps.append("main_pool_recall")

        if main_hits:
            query_info["recall"] = self._build_recall_result(main_hits, [], main_hits, recall_steps)
            return query_info

        backup_hits = self._search_pool(query_info["search_terms"], self.backup_poi_index)
        recall_steps.append("backup_pool_recall")

        if backup_hits:
            query_info["recall"] = self._build_recall_result([], backup_hits, backup_hits, recall_steps)
            return query_info

        if query_info["category_intent"] == "category_search":
            reduced_terms = self._drop_modifiers(query_info["lemmas"], query_info["chunk_tags"])
            reduced_terms = self._expand_search_terms(reduced_terms)
            recall_steps.append("modifier_drop_recall")
            third_hits = self._search_pool(reduced_terms, self.main_poi_index + self.backup_poi_index)
            query_info["recall"] = self._build_recall_result([], [], third_hits, recall_steps)
            return query_info

        query_info["recall"] = self._build_recall_result([], [], [], recall_steps)
        return query_info


if __name__ == "__main__":
    engine = SearchRecallEngine()
    engine.processor.set_idf({"new jersey": 0.48, "famous": 0.39, "hot": 0.55, "spring": 0.55})
    engine.processor.update_click_log(["new jersey", "famous", "hot", "spring"])

    query = "New Jersey famous hot spring"
    result = engine.recall(query)

    print("Query:", query)
    print("Chunk tags:", result["chunk_tags"])
    print("Category intent:", result["category_intent"])
    print("Search terms:", result["search_terms"])
    print("Recall steps:", result["recall"]["recall_steps"])
    print("Final recall:")
    for poi in result["recall"]["final_recall"]:
        print("-", poi["name"], f"({poi['city']} - {poi['category']})")
```

---

### 4. intention_classifier.py - User Intent Classification
PyTorch-based neural classifier for 4-class intent prediction.

```python
import torch
import torch.nn as nn
from typing import List, Dict, Tuple
from collections import Counter


class IntentionClassifier(nn.Module):
    """PyTorch-based intention classifier using a simple feedforward neural network."""

    def __init__(self, vocab_size: int = 5000, embedding_dim: int = 128, hidden_dim: int = 256, num_classes: int = 4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, num_classes)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(token_ids)
        pooled = embedded.mean(dim=1)
        hidden = self.dropout(self.relu(self.fc1(pooled)))
        hidden = self.dropout(self.relu(self.fc2(hidden)))
        output = self.fc3(hidden)
        return output

    def predict(self, token_ids: torch.Tensor) -> Tuple[List[str], List[float]]:
        self.eval()
        with torch.no_grad():
            logits = self.forward(token_ids)
            probs = torch.softmax(logits, dim=1)
            predictions = torch.argmax(logits, dim=1)
        
        intent_labels = ["purchase", "navigate", "qa", "recommendation"]
        predicted_intents = [intent_labels[pred.item()] for pred in predictions]
        predicted_probs = [prob.item() for prob in probs.max(dim=1).values]
        return predicted_intents, predicted_probs


class TokenVocabulary:
    """Simple vocabulary for token-to-id conversion."""

    def __init__(self):
        self.token2id = {"<PAD>": 0, "<UNK>": 1}
        self.id2token = {0: "<PAD>", 1: "<UNK>"}
        self.counter = Counter()

    def build(self, tokens_list: List[List[str]], min_freq: int = 2) -> None:
        for tokens in tokens_list:
            self.counter.update(tokens)
        
        for token, freq in self.counter.items():
            if freq >= min_freq and token not in self.token2id:
                token_id = len(self.token2id)
                self.token2id[token] = token_id
                self.id2token[token_id] = token

    def encode(self, tokens: List[str], max_len: int = 32) -> torch.Tensor:
        token_ids = [self.token2id.get(token, self.token2id["<UNK>"]) for token in tokens]
        token_ids = token_ids[:max_len]
        token_ids += [self.token2id["<PAD>"]] * (max_len - len(token_ids))
        return torch.tensor([token_ids], dtype=torch.long)

    def decode(self, token_ids: List[int]) -> List[str]:
        return [self.id2token.get(token_id, "<UNK>") for token_id in token_ids]


class IntentionClassificationPipeline:
    """Pipeline for intention classification with vocabulary management."""

    def __init__(self, vocab_size: int = 5000, embedding_dim: int = 128, hidden_dim: int = 256):
        self.vocab = TokenVocabulary()
        self.model = IntentionClassifier(vocab_size, embedding_dim, hidden_dim, num_classes=4)
        self.is_trained = False

    def train_model(self, train_data: List[Tuple[List[str], str]], epochs: int = 10, lr: float = 0.001) -> None:
        self.vocab.build([tokens for tokens, _ in train_data])
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        intent_labels = ["purchase", "navigate", "qa", "recommendation"]
        label_to_id = {label: idx for idx, label in enumerate(intent_labels)}

        self.model.train()
        for epoch in range(epochs):
            total_loss = 0.0
            for tokens, label in train_data:
                token_ids = self.vocab.encode(tokens)
                token_ids = token_ids.to(self.model.device)
                label_id = torch.tensor([label_to_id[label]], dtype=torch.long).to(self.model.device)
                
                optimizer.zero_grad()
                logits = self.model(token_ids)
                loss = criterion(logits, label_id)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            if (epoch + 1) % 2 == 0:
                avg_loss = total_loss / len(train_data)
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")
        
        self.is_trained = True

    def predict(self, tokens: List[str]) -> Tuple[str, float]:
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call train_model first.")
        
        token_ids = self.vocab.encode(tokens)
        token_ids = token_ids.to(self.model.device)
        predicted_intents, predicted_probs = self.model.predict(token_ids)
        
        return predicted_intents[0], predicted_probs[0]


if __name__ == "__main__":
    training_data = [
        (["buy", "phone", "online"], "purchase"),
        (["order", "pizza", "delivery"], "purchase"),
        (["directions", "to", "restaurant"], "navigate"),
        (["map", "nearest", "hotel"], "navigate"),
        (["what", "is", "weather"], "qa"),
        (["how", "to", "cook", "pasta"], "qa"),
        (["recommend", "movie"], "recommendation"),
        (["suggest", "best", "restaurant"], "recommendation"),
    ]

    pipeline = IntentionClassificationPipeline()
    pipeline.train_model(training_data, epochs=10)

    test_tokens = ["buy", "laptop", "now"]
    intent, confidence = pipeline.predict(test_tokens)
    print(f"Predicted intent: {intent} (confidence: {confidence:.4f})")
```

---

### 5. bm25_retriever.py - Sparse Retrieval
BM25 algorithm implementation for efficient sparse retrieval.

```python
import math
from typing import List, Dict, Tuple
from collections import Counter


class BM25:
    """BM25 sparse retrieval algorithm implementation."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.docs: List[List[str]] = []
        self.doc_freqs: List[Dict[str, int]] = []
        self.idf: Dict[str, float] = {}
        self.avg_doc_len = 0.0
        self.doc_len: List[int] = []
        self.num_docs = 0
        self.vocab: set = set()

    def index(self, documents: List[List[str]]) -> None:
        self.docs = documents
        self.num_docs = len(documents)
        
        # Calculate document frequencies
        for doc in documents:
            self.doc_freqs.append(Counter(doc))
            self.doc_len.append(len(doc))
            self.vocab.update(doc)
        
        # Calculate average document length
        self.avg_doc_len = sum(self.doc_len) / self.num_docs if self.num_docs > 0 else 0
        
        # Calculate IDF for each term
        for term in self.vocab:
            doc_count = sum(1 for doc_freq in self.doc_freqs if term in doc_freq)
            self.idf[term] = math.log((self.num_docs - doc_count + 0.5) / (doc_count + 0.5) + 1)

    def search(self, query: List[str], top_k: int = 5) -> List[Tuple[int, float]]:
        if not self.docs:
            return []
        
        query_freq = Counter(query)
        scores = []
        
        for doc_id, doc_freq in enumerate(self.doc_freqs):
            score = 0.0
            for term in query_freq:
                if term not in self.idf:
                    continue
                
                term_freq = doc_freq.get(term, 0)
                idf = self.idf[term]
                doc_len = self.doc_len[doc_id]
                
                numerator = idf * term_freq * (self.k1 + 1)
                denominator = term_freq + self.k1 * (1 - self.b + self.b * (doc_len / self.avg_doc_len))
                score += numerator / denominator
            
            if score > 0:
                scores.append((doc_id, score))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def batch_search(self, queries: List[List[str]], top_k: int = 5) -> List[List[Tuple[int, float]]]:
        return [self.search(query, top_k) for query in queries]


class BM25Retriever:
    """High-level BM25 retriever for POI/document ranking."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.bm25 = BM25(k1=k1, b=b)
        self.documents: List[Dict[str, object]] = []

    def index_documents(self, documents: List[Dict[str, object]], text_fields: List[str] = None) -> None:
        if text_fields is None:
            text_fields = ["name", "description", "category"]
        
        self.documents = documents
        indexed_docs = []
        
        for doc in documents:
            tokens = []
            for field in text_fields:
                text = str(doc.get(field, "")).lower().split()
                tokens.extend(text)
            indexed_docs.append(tokens)
        
        self.bm25.index(indexed_docs)

    def retrieve(self, query: List[str], top_k: int = 5) -> List[Dict[str, object]]:
        query_lower = [token.lower() for token in query]
        results = self.bm25.search(query_lower, top_k=top_k)
        
        retrieved_docs = []
        for doc_id, score in results:
            doc = self.documents[doc_id].copy()
            doc["bm25_score"] = score
            retrieved_docs.append(doc)
        
        return retrieved_docs

    def batch_retrieve(self, queries: List[List[str]], top_k: int = 5) -> List[List[Dict[str, object]]]:
        return [self.retrieve(query, top_k) for query in queries]


if __name__ == "__main__":
    # Example with POI documents
    pois = [
        {"id": 1, "name": "Shunjing Hot Spring", "category": "Hot Spring", "description": "well-known resort"},
        {"id": 2, "name": "Jiuhua Mountain Resort", "category": "Hot Spring", "description": "scenic spa resort"},
        {"id": 3, "name": "City Spa Center", "category": "Spa", "description": "relaxing services"},
        {"id": 4, "name": "Capital Hot Spring", "category": "Hot Spring", "description": "popular hotel"},
    ]

    retriever = BM25Retriever()
    retriever.index_documents(pois, text_fields=["name", "category", "description"])

    # Retrieve top-2 for a query
    query = ["hot", "spring"]
    results = retriever.retrieve(query, top_k=2)
    print(f"Query: {query}")
    for doc in results:
        print(f"  - {doc['name']} (BM25 score: {doc['bm25_score']:.4f})")

    # Batch retrieval
    queries = [["hot", "spring"], ["spa"]]
    batch_results = retriever.batch_retrieve(queries, top_k=2)
    for query, results in zip(queries, batch_results):
        print(f"Query: {query}")
        for doc in results:
            print(f"  - {doc['name']} (BM25 score: {doc['bm25_score']:.4f})")
```

---

### 6. learning_to_rank.py - Neural Ranking
Point-wise and pairwise Learning-to-Rank models for document ranking.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict
import numpy as np


class PointwiseLTRModel(nn.Module):
    """Point-wise Learning-to-Rank model treating ranking as a regression problem."""

    def __init__(self, feature_dim: int = 32, hidden_dim: int = 128):
        super().__init__()
        self.feature_dim = feature_dim
        self.fc1 = nn.Linear(feature_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, 1)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        hidden = self.dropout(self.relu(self.fc1(features)))
        hidden = self.dropout(self.relu(self.fc2(hidden)))
        score = torch.sigmoid(self.fc3(hidden))
        return score


class PairwiseLTRModel(nn.Module):
    """Pair-wise Learning-to-Rank model using pairwise preference classification."""

    def __init__(self, feature_dim: int = 32, hidden_dim: int = 128):
        super().__init__()
        self.feature_dim = feature_dim
        self.fc1 = nn.Linear(feature_dim * 2, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, 1)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, doc1_features: torch.Tensor, doc2_features: torch.Tensor) -> torch.Tensor:
        concat_features = torch.cat([doc1_features, doc2_features], dim=1)
        hidden = self.dropout(self.relu(self.fc1(concat_features)))
        hidden = self.dropout(self.relu(self.fc2(hidden)))
        preference = torch.sigmoid(self.fc3(hidden))
        return preference


class RankingFeatureExtractor:
    """Extract ranking features from documents and queries."""

    def extract_features(
        self, query: List[str], doc: Dict[str, object], bm25_score: float = 0.0
    ) -> np.ndarray:
        """
        Extract features for a query-document pair.
        Features: [bm25_score, query_len, doc_len, term_overlap, idf_sum]
        """
        query_set = set(query)
        doc_text = " ".join(
            [str(doc.get(field, "")).lower() for field in ["name", "description", "category"]]
        )
        doc_tokens = doc_text.split()
        doc_set = set(doc_tokens)

        term_overlap = len(query_set & doc_set) / (len(query_set) + 1e-8)
        query_len = len(query)
        doc_len = len(doc_tokens)

        features = [
            bm25_score,
            query_len,
            doc_len,
            term_overlap,
            len(query_set & doc_set),
        ]
        return np.array(features, dtype=np.float32)

    def extract_batch_features(
        self, queries: List[List[str]], docs: List[Dict[str, object]], scores: List[float] = None
    ) -> np.ndarray:
        if scores is None:
            scores = [0.0] * len(docs)

        batch_features = []
        for query, doc, score in zip(queries, docs, scores):
            features = self.extract_features(query, doc, score)
            batch_features.append(features)
        return np.array(batch_features, dtype=np.float32)


class PointwiseLTRRanker:
    """Point-wise LTR ranker for document ranking."""

    def __init__(self, feature_dim: int = 5, hidden_dim: int = 128):
        self.model = PointwiseLTRModel(feature_dim, hidden_dim)
        self.feature_extractor = RankingFeatureExtractor()
        self.is_trained = False

    def train(
        self,
        train_data: List[Tuple[List[str], List[Dict[str, object]], List[int]]],
        epochs: int = 10,
        lr: float = 0.001,
    ) -> None:
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        self.model.train()
        for epoch in range(epochs):
            total_loss = 0.0

            for query, docs, relevance_labels in train_data:
                for doc, label in zip(docs, relevance_labels):
                    features = self.feature_extractor.extract_features(query, doc)
                    features_tensor = torch.tensor([features]).to(self.model.device)
                    label_tensor = torch.tensor([[float(label) / 5.0]]).to(self.model.device)

                    optimizer.zero_grad()
                    pred = self.model(features_tensor)
                    loss = criterion(pred, label_tensor)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()

            if (epoch + 1) % 2 == 0:
                avg_loss = total_loss / max(sum(len(docs) for _, docs, _ in train_data), 1)
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

        self.is_trained = True

    def rank(self, query: List[str], docs: List[Dict[str, object]], top_k: int = 5) -> List[Dict[str, object]]:
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call train first.")

        self.model.eval()
        scores = []

        with torch.no_grad():
            for doc in docs:
                features = self.feature_extractor.extract_features(query, doc)
                features_tensor = torch.tensor([features]).to(self.model.device)
                score = self.model(features_tensor).item()
                scores.append(score)

        ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        result = []
        for doc, score in ranked[:top_k]:
            doc_copy = doc.copy()
            doc_copy["pointwise_ltr_score"] = score
            result.append(doc_copy)

        return result


class PairwiseLTRRanker:
    """Pair-wise LTR ranker using pairwise preference classification."""

    def __init__(self, feature_dim: int = 5, hidden_dim: int = 128):
        self.model = PairwiseLTRModel(feature_dim, hidden_dim)
        self.feature_extractor = RankingFeatureExtractor()
        self.is_trained = False

    def train(
        self,
        train_data: List[Tuple[List[str], List[Dict[str, object]], List[int]]],
        epochs: int = 10,
        lr: float = 0.001,
    ) -> None:
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.BCELoss()

        self.model.train()
        for epoch in range(epochs):
            total_loss = 0.0
            pair_count = 0

            for query, docs, relevance_labels in train_data:
                for i in range(len(docs)):
                    for j in range(i + 1, len(docs)):
                        doc1, doc2 = docs[i], docs[j]
                        label1, label2 = relevance_labels[i], relevance_labels[j]

                        features1 = self.feature_extractor.extract_features(query, doc1)
                        features2 = self.feature_extractor.extract_features(query, doc2)

                        features1_tensor = torch.tensor([features1]).to(self.model.device)
                        features2_tensor = torch.tensor([features2]).to(self.model.device)

                        if label1 > label2:
                            pair_label = torch.tensor([[1.0]]).to(self.model.device)
                        elif label1 < label2:
                            pair_label = torch.tensor([[0.0]]).to(self.model.device)
                        else:
                            continue

                        optimizer.zero_grad()
                        pred = self.model(features1_tensor, features2_tensor)
                        loss = criterion(pred, pair_label)
                        loss.backward()
                        optimizer.step()
                        total_loss += loss.item()
                        pair_count += 1

            if (epoch + 1) % 2 == 0:
                avg_loss = total_loss / max(pair_count, 1)
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

        self.is_trained = True

    def rank(self, query: List[str], docs: List[Dict[str, object]], top_k: int = 5) -> List[Dict[str, object]]:
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call train first.")

        self.model.eval()
        scores = [0.0] * len(docs)

        with torch.no_grad():
            for i in range(len(docs)):
                for j in range(len(docs)):
                    if i == j:
                        continue

                    features1 = self.feature_extractor.extract_features(query, docs[i])
                    features2 = self.feature_extractor.extract_features(query, docs[j])

                    features1_tensor = torch.tensor([features1]).to(self.model.device)
                    features2_tensor = torch.tensor([features2]).to(self.model.device)

                    preference = self.model(features1_tensor, features2_tensor).item()
                    scores[i] += preference

        ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        result = []
        for doc, score in ranked[:top_k]:
            doc_copy = doc.copy()
            doc_copy["pairwise_ltr_score"] = score
            result.append(doc_copy)

        return result


if __name__ == "__main__":
    pois = [
        {"id": 1, "name": "Shunjing Hot Spring", "category": "Hot Spring", "description": "well-known resort"},
        {"id": 2, "name": "Jiuhua Mountain Resort", "category": "Hot Spring", "description": "scenic spa resort"},
        {"id": 3, "name": "City Spa Center", "category": "Spa", "description": "relaxing services"},
        {"id": 4, "name": "Capital Hot Spring", "category": "Hot Spring", "description": "popular hotel"},
    ]

    training_data = [
        (["hot", "spring"], pois[:3], [3, 4, 1]),
        (["spa", "city"], pois, [2, 1, 5, 2]),
    ]

    print("=== Point-wise LTR ===")
    pointwise_ranker = PointwiseLTRRanker()
    pointwise_ranker.train(training_data, epochs=5)
    results = pointwise_ranker.rank(["hot", "spring"], pois, top_k=2)
    for doc in results:
        print(f"  {doc['name']}: {doc.get('pointwise_ltr_score', 0):.4f}")

    print("\n=== Pair-wise LTR ===")
    pairwise_ranker = PairwiseLTRRanker()
    pairwise_ranker.train(training_data, epochs=5)
    results = pairwise_ranker.rank(["hot", "spring"], pois, top_k=2)
    for doc in results:
        print(f"  {doc['name']}: {doc.get('pairwise_ltr_score', 0):.4f}")
```

---

### 7. reranker.py - Cross-Encoder & Deep Semantic
Reranking models for refining retrieved results.

```python
import torch
import torch.nn as nn
from typing import List, Dict, Tuple
import numpy as np


class CrossEncoderModel(nn.Module):
    """Cross-encoder model that jointly encodes query and document."""

    def __init__(self, input_dim: int = 64, hidden_dim: int = 256):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, 1)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: concatenated query and document features
        Returns:
            relevance score [0, 1]
        """
        hidden = self.dropout(self.relu(self.fc1(features)))
        hidden = self.dropout(self.relu(self.fc2(hidden)))
        score = torch.sigmoid(self.fc3(hidden))
        return score


class DenseSemanticModel(nn.Module):
    """Deep semantic model with separate query and document encoders."""

    def __init__(self, input_dim: int = 32, embedding_dim: int = 128, hidden_dim: int = 256):
        super().__init__()
        self.embedding_dim = embedding_dim

        # Query encoder
        self.query_fc1 = nn.Linear(input_dim, hidden_dim)
        self.query_relu = nn.ReLU()
        self.query_dropout = nn.Dropout(0.3)
        self.query_fc2 = nn.Linear(hidden_dim, embedding_dim)

        # Document encoder
        self.doc_fc1 = nn.Linear(input_dim, hidden_dim)
        self.doc_relu = nn.ReLU()
        self.doc_dropout = nn.Dropout(0.3)
        self.doc_fc2 = nn.Linear(hidden_dim, embedding_dim)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def encode_query(self, query_features: torch.Tensor) -> torch.Tensor:
        hidden = self.query_dropout(self.query_relu(self.query_fc1(query_features)))
        embedding = self.query_fc2(hidden)
        embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
        return embedding

    def encode_document(self, doc_features: torch.Tensor) -> torch.Tensor:
        hidden = self.doc_dropout(self.doc_relu(self.doc_fc1(doc_features)))
        embedding = self.doc_fc2(hidden)
        embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
        return embedding

    def forward(self, query_features: torch.Tensor, doc_features: torch.Tensor) -> torch.Tensor:
        query_emb = self.encode_query(query_features)
        doc_emb = self.encode_document(doc_features)
        similarity = torch.sum(query_emb * doc_emb, dim=1, keepdim=True)
        return similarity


class RerankerFeatureExtractor:
    """Extract features for reranking."""

    def extract_features(self, query: List[str], doc: Dict[str, object]) -> np.ndarray:
        """Extract semantic and relevance features."""
        query_set = set(query)
        doc_text = " ".join(
            [str(doc.get(field, "")).lower() for field in ["name", "description", "category"]]
        )
        doc_tokens = doc_text.split()
        doc_set = set(doc_tokens)

        term_overlap = len(query_set & doc_set) / (len(query_set) + 1e-8)
        query_len = len(query) / 10.0
        doc_len = len(doc_tokens) / 50.0
        shared_terms = len(query_set & doc_set) / 10.0

        features = [
            term_overlap,
            query_len,
            doc_len,
            shared_terms,
            len(query),
        ]
        return np.array(features, dtype=np.float32)


class CrossEncoderReranker:
    """Cross-encoder based reranker."""

    def __init__(self, input_dim: int = 64, hidden_dim: int = 256):
        self.model = CrossEncoderModel(input_dim, hidden_dim)
        self.feature_extractor = RerankerFeatureExtractor()
        self.is_trained = False

    def train(
        self,
        train_data: List[Tuple[List[str], List[Dict[str, object]], List[int]]],
        epochs: int = 10,
        lr: float = 0.001,
    ) -> None:
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.BCELoss()

        self.model.train()
        for epoch in range(epochs):
            total_loss = 0.0
            count = 0

            for query, docs, relevance_labels in train_data:
                for doc, label in zip(docs, relevance_labels):
                    features = self.feature_extractor.extract_features(query, doc)
                    normalized_label = min(label / 5.0, 1.0)

                    features_tensor = torch.tensor([features]).to(self.model.device)
                    label_tensor = torch.tensor([[normalized_label]]).to(self.model.device)

                    optimizer.zero_grad()
                    score = self.model(features_tensor)
                    loss = criterion(score, label_tensor)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                    count += 1

            if (epoch + 1) % 2 == 0:
                avg_loss = total_loss / max(count, 1)
                print(f"Cross-Encoder Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

        self.is_trained = True

    def rerank(self, query: List[str], docs: List[Dict[str, object]], top_k: int = 5) -> List[Dict[str, object]]:
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call train first.")

        self.model.eval()
        scores = []

        with torch.no_grad():
            for doc in docs:
                features = self.feature_extractor.extract_features(query, doc)
                features_tensor = torch.tensor([features]).to(self.model.device)
                score = self.model(features_tensor).item()
                scores.append(score)

        ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        result = []
        for doc, score in ranked[:top_k]:
            doc_copy = doc.copy()
            doc_copy["cross_encoder_score"] = score
            result.append(doc_copy)

        return result


class DenseSemanticReranker:
    """Dense semantic model based reranker."""

    def __init__(self, input_dim: int = 5, embedding_dim: int = 128, hidden_dim: int = 256):
        self.model = DenseSemanticModel(input_dim, embedding_dim, hidden_dim)
        self.feature_extractor = RerankerFeatureExtractor()
        self.is_trained = False

    def train(
        self,
        train_data: List[Tuple[List[str], List[Dict[str, object]], List[int]]],
        epochs: int = 10,
        lr: float = 0.001,
        margin: float = 0.3,
    ) -> None:
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        self.model.train()
        for epoch in range(epochs):
            total_loss = 0.0
            pair_count = 0

            for query, docs, relevance_labels in train_data:
                for i in range(len(docs)):
                    for j in range(i + 1, len(docs)):
                        if relevance_labels[i] == relevance_labels[j]:
                            continue

                        query_features = self.feature_extractor.extract_features(query, docs[i])
                        doc1_features = self.feature_extractor.extract_features(query, docs[i])
                        doc2_features = self.feature_extractor.extract_features(query, docs[j])

                        query_tensor = torch.tensor([query_features]).to(self.model.device)
                        doc1_tensor = torch.tensor([doc1_features]).to(self.model.device)
                        doc2_tensor = torch.tensor([doc2_features]).to(self.model.device)

                        optimizer.zero_grad()

                        score1 = self.model(query_tensor, doc1_tensor)
                        score2 = self.model(query_tensor, doc2_tensor)

                        if relevance_labels[i] > relevance_labels[j]:
                            loss = torch.clamp(margin - (score1 - score2), min=0.0).mean()
                        else:
                            loss = torch.clamp(margin - (score2 - score1), min=0.0).mean()

                        loss.backward()
                        optimizer.step()
                        total_loss += loss.item()
                        pair_count += 1

            if (epoch + 1) % 2 == 0:
                avg_loss = total_loss / max(pair_count, 1)
                print(f"Dense Semantic Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

        self.is_trained = True

    def rerank(self, query: List[str], docs: List[Dict[str, object]], top_k: int = 5) -> List[Dict[str, object]]:
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call train first.")

        self.model.eval()
        scores = []

        with torch.no_grad():
            query_features = self.feature_extractor.extract_features(query, {"name": "", "description": ""})
            query_tensor = torch.tensor([query_features]).to(self.model.device)

            for doc in docs:
                doc_features = self.feature_extractor.extract_features(query, doc)
                doc_tensor = torch.tensor([doc_features]).to(self.model.device)
                score = self.model(query_tensor, doc_tensor).item()
                scores.append(score)

        ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        result = []
        for doc, score in ranked[:top_k]:
            doc_copy = doc.copy()
            doc_copy["dense_semantic_score"] = score
            result.append(doc_copy)

        return result


class HybridReranker:
    """Hybrid reranker combining multiple models."""

    def __init__(self):
        self.cross_encoder = CrossEncoderReranker()
        self.dense_semantic = DenseSemanticReranker()

    def train(
        self,
        train_data: List[Tuple[List[str], List[Dict[str, object]], List[int]]],
        epochs: int = 10,
    ) -> None:
        print("Training Cross-Encoder...")
        self.cross_encoder.train(train_data, epochs=epochs)
        print("Training Dense Semantic Model...")
        self.dense_semantic.train(train_data, epochs=epochs)

    def rerank(self, query: List[str], docs: List[Dict[str, object]], top_k: int = 5, weights: Dict[str, float] = None) -> List[Dict[str, object]]:
        if weights is None:
            weights = {"cross_encoder": 0.5, "dense_semantic": 0.5}

        cross_results = self.cross_encoder.rerank(query, docs, top_k=len(docs))
        dense_results = self.dense_semantic.rerank(query, docs, top_k=len(docs))

        score_map = {}
        for doc in cross_results:
            doc_id = doc.get("id")
            score_map[doc_id] = score_map.get(doc_id, 0.0) + weights["cross_encoder"] * doc.get("cross_encoder_score", 0.0)

        for doc in dense_results:
            doc_id = doc.get("id")
            score_map[doc_id] = score_map.get(doc_id, 0.0) + weights["dense_semantic"] * doc.get("dense_semantic_score", 0.0)

        hybrid_scored = []
        for doc in docs:
            doc_copy = doc.copy()
            doc_copy["hybrid_score"] = score_map.get(doc.get("id"), 0.0)
            hybrid_scored.append(doc_copy)

        ranked = sorted(hybrid_scored, key=lambda x: x["hybrid_score"], reverse=True)
        return ranked[:top_k]


if __name__ == "__main__":
    pois = [
        {"id": 1, "name": "Shunjing Hot Spring", "category": "Hot Spring", "description": "well-known resort"},
        {"id": 2, "name": "Jiuhua Mountain Resort", "category": "Hot Spring", "description": "scenic spa resort"},
        {"id": 3, "name": "City Spa Center", "category": "Spa", "description": "relaxing services"},
        {"id": 4, "name": "Capital Hot Spring", "category": "Hot Spring", "description": "popular hotel"},
    ]

    training_data = [
        (["hot", "spring"], pois[:3], [3, 4, 1]),
        (["spa", "city"], pois, [2, 1, 5, 2]),
    ]

    print("=== Cross-Encoder Reranker ===")
    cross_reranker = CrossEncoderReranker()
    cross_reranker.train(training_data, epochs=5)
    results = cross_reranker.rerank(["hot", "spring"], pois, top_k=2)
    for doc in results:
        print(f"  {doc['name']}: {doc.get('cross_encoder_score', 0):.4f}")

    print("\n=== Dense Semantic Reranker ===")
    dense_reranker = DenseSemanticReranker()
    dense_reranker.train(training_data, epochs=5)
    results = dense_reranker.rerank(["hot", "spring"], pois, top_k=2)
    for doc in results:
        print(f"  {doc['name']}: {doc.get('dense_semantic_score', 0):.4f}")

    print("\n=== Hybrid Reranker ===")
    hybrid_reranker = HybridReranker()
    hybrid_reranker.train(training_data, epochs=5)
    results = hybrid_reranker.rerank(["hot", "spring"], pois, top_k=2)
    for doc in results:
        print(f"  {doc['name']}: {doc.get('hybrid_score', 0):.4f}")
```

---

## Module Details

| Module | Purpose | Key Classes | Dependencies |
|--------|---------|-------------|--------------|
| search_engine.py | Query preprocessing | SearchQueryProcessor | nltk, spacy, re |
| query_understanding.py | Query analysis & enhancement | SearchQueryProcessor | nltk, spacy, re |
| search_recall.py | Multi-stage retrieval | SearchRecallEngine | query_understanding |
| intention_classifier.py | Intent classification | IntentionClassifier, IntentionClassificationPipeline | torch |
| bm25_retriever.py | Sparse retrieval | BM25, BM25Retriever | math, typing |
| learning_to_rank.py | Neural ranking | PointwiseLTRRanker, PairwiseLTRRanker | torch, numpy |
| reranker.py | Result reranking | CrossEncoderReranker, DenseSemanticReranker, HybridReranker | torch, numpy |

---

## Usage Examples

### End-to-End Search Pipeline

```python
from query_understanding import SearchQueryProcessor
from search_recall import SearchRecallEngine
from bm25_retriever import BM25Retriever
from learning_to_rank import PointwiseLTRRanker
from reranker import HybridReranker

# 1. Process query
processor = SearchQueryProcessor()
query_result = processor.process_query("hot spring near new york")

# 2. Recall
recall_engine = SearchRecallEngine()
recall_result = recall_engine.recall("hot spring near new york")

# 3. Sparse retrieval (BM25)
retriever = BM25Retriever()
retrieved_docs = retriever.retrieve(query_result["lemmas"], top_k=10)

# 4. Neural ranking
ranker = PointwiseLTRRanker()
ranked_docs = ranker.rank(query_result["lemmas"], retrieved_docs, top_k=5)

# 5. Reranking
reranker = HybridReranker()
final_results = reranker.rerank(query_result["lemmas"], ranked_docs, top_k=3)
```

---

## Performance Characteristics

- **Query Processing**: O(n) where n = query length
- **BM25 Retrieval**: O(q*d) where q = query terms, d = documents
- **LTR Ranking**: O(d*f) where d = documents, f = features
- **Cross-Encoder Reranking**: O(d) one-pass encoding
- **Dense Semantic**: O(d) with separate encoders

---

## Future Enhancements

1. **BERT-based embeddings** for improved semantic understanding
2. **ListWise LTR** for group-level ranking optimization
3. **Multi-field indexing** with field-specific analysis
4. **Model serialization** for deployment
5. **Evaluation metrics** (NDCG, MRR, MAP)
6. **A/B testing framework** for ranking models

---

## References

- BM25: Okapi BM25 full text search algorithm
- Learning-to-Rank: Li, H., et al. Learning to Rank for Information Retrieval
- Cross-Encoders: Thakur, N., et al. BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation
- Dense Retrieval: Karpukhin, V., et al. Dense Passage Retrieval for Open-domain Question Answering
