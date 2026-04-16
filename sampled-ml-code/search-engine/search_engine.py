import re
import unicodedata
from collections import Counter
from typing import Dict, List, Optional, Set, Tuple

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.util import ngrams

from text_preprocessing import extract_companies
from vocabulary import Vocabulary

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
    LOCATION_WORDS = {
        "london",
        "manchester",
        "edinburgh",
        "oxford",
        "cambridge",
        "bristol",
    }

    def __init__(self, language: str = "en"):
        self.language = language
        self.stop_words = self._load_stop_words(language)
        self.lemmatizer = WordNetLemmatizer()
        self.vocab = Vocabulary()
        self.intent_keywords = {
            "purchase": ["buy", "purchase", "order", "shop", "checkout"],
            "navigate": ["map", "directions", "navigate", "route", "go to"],
            "qa": ["what", "who", "when", "how", "where", "why"],
            "recommendation": ["recommend", "suggest", "advice", "best"],
        }
        self.synonym_map = {
            "tv": ["television", "smart tv"],
            "notebook": ["laptop", "ultrabook"],
            "museum": ["gallery", "exhibition"],
            "gallery": ["museum", "exhibition"],
            "phone": ["smartphone", "mobile"],
        }
        self.click_log = Counter()
        self.idf = Counter()
        self.spacy_nlp = self._load_spacy_model() if SPACY_AVAILABLE else None

    def _load_stop_words(self, language: str) -> Set[str]:
        if language != "en":
            return set()
        try:
            return set(stopwords.words("english"))
        except LookupError:
            return {
                "a",
                "an",
                "and",
                "are",
                "for",
                "i",
                "in",
                "is",
                "of",
                "the",
                "to",
                "want",
                "with",
            }

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
        if not text:
            return []
        if self.spacy_nlp:
            doc = self.spacy_nlp(text)
            return [token.text for token in doc if not token.is_space]
        try:
            return nltk.word_tokenize(text)
        except LookupError:
            return text.split()

    def pos_tag(self, tokens: List[str]) -> List[Tuple[str, str]]:
        if not tokens:
            return []
        if self.spacy_nlp:
            doc = self.spacy_nlp(" ".join(tokens))
            return [(token.text, token.pos_) for token in doc if not token.is_space]
        try:
            return nltk.pos_tag(tokens)
        except LookupError:
            return [(token, "NOUN") for token in tokens]

    def named_entities(self, text: str) -> List[Tuple[str, str]]:
        if self.spacy_nlp:
            doc = self.spacy_nlp(text)
            return [(ent.text, ent.label_) for ent in doc.ents]
        entities = [(item, "DATE") for item in self.DATE_PATTERN.findall(text)]
        entities.extend((item, "NUMBER") for item in self.NUMBER_PATTERN.findall(text))
        return entities

    def extract_organizations(self, text: str, render: bool = False) -> List[str]:
        return extract_companies(text, render=render)

    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        return [token for token in tokens if token.lower() not in self.stop_words]

    def lemmatize(self, tokens: List[str]) -> List[str]:
        if not tokens:
            return []
        if self.spacy_nlp:
            doc = self.spacy_nlp(" ".join(tokens))
            return [token.lemma_.lower() for token in doc if not token.is_space]
        lemmas = []
        for token in tokens:
            try:
                lemmas.append(self.lemmatizer.lemmatize(token.lower()))
            except LookupError:
                lemmas.append(token.lower())
        return lemmas

    def generate_ngrams(self, tokens: List[str], n: int = 2) -> List[Tuple[str, ...]]:
        if len(tokens) < n:
            return []
        return list(ngrams(tokens, n))

    def calculate_term_weights(self, tokens: List[str]) -> Dict[str, float]:
        return {token: float(self.click_log.get(token, 1)) * self.idf.get(token, 1.0) for token in tokens}

    def update_click_log(self, clicked_terms: List[str]) -> None:
        self.click_log.update(term.lower() for term in clicked_terms)

    def set_idf(self, idf_mapping: Dict[str, float]) -> None:
        self.idf.update({term.lower(): weight for term, weight in idf_mapping.items()})

    def vocabulary(self) -> Set[str]:
        synonym_terms = {term for values in self.synonym_map.values() for term in values}
        return set(self.synonym_map.keys()) | synonym_terms | set(self.vocab.tokens) | self.stop_words | self.LOCATION_WORDS

    def expand_synonyms(self, tokens: List[str]) -> List[str]:
        expanded = list(tokens)
        for token in tokens:
            expanded.extend(self.synonym_map.get(token, []))
        return list(dict.fromkeys(expanded))

    def analyze_chunks(self, tokens: List[str], pos_tags: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        chunk_tags = []
        for token, pos in pos_tags:
            lower_token = token.lower()
            if lower_token in self.LOCATION_WORDS or pos in {"GPE", "LOC"}:
                chunk_tags.append((token, "LOCATION"))
            elif pos in {"ADJ", "ADV"}:
                chunk_tags.append((token, "MODIFIER"))
            elif pos in {"NOUN", "PROPN"}:
                chunk_tags.append((token, "CATEGORY"))
            else:
                chunk_tags.append((token, "OTHER"))
        return chunk_tags

    def detect_category_intent(self, chunk_tags: List[Tuple[str, str]]) -> str:
        category_count = sum(1 for _, tag in chunk_tags if tag == "CATEGORY")
        modifier_count = sum(1 for _, tag in chunk_tags if tag == "MODIFIER")
        if category_count and modifier_count:
            return "category_search"
        if category_count:
            return "exact_search"
        return "keyword_search"

    def detect_intents(self, text: str, tokens: Optional[List[str]] = None) -> List[Tuple[str, float]]:
        lower_text = text.lower()
        scores = [
            (intent, sum(keyword in lower_text for keyword in keywords) / len(keywords))
            for intent, keywords in self.intent_keywords.items()
            if any(keyword in lower_text for keyword in keywords)
        ]
        return sorted(scores, key=lambda item: item[1], reverse=True)

    def recognize_entities(self, text: str) -> List[Tuple[str, str]]:
        entities = self.named_entities(text)
        if not entities:
            for location in self.LOCATION_WORDS:
                if location in text.lower():
                    entities.append((location.title(), "GPE"))
        return entities

    def simple_error_correction(self, tokens: List[str]) -> List[str]:
        vocabulary = set(self.vocab.tokens)
        corrected = []
        for token in tokens:
            lower_token = token.lower()
            if lower_token in vocabulary or len(token) <= 1:
                corrected.append(lower_token)
                continue
            # First-char + length filter cuts candidate set ~10x before the expensive edit-distance scan
            first_char = lower_token[0]
            candidates = [
                word for word in vocabulary
                if abs(len(word) - len(lower_token)) <= 2 and word[:1] == first_char
            ]
            if not candidates:
                candidates = [word for word in vocabulary if abs(len(word) - len(lower_token)) <= 2]
            if not candidates:
                corrected.append(lower_token)
                continue
            best = min(candidates, key=lambda cand: nltk.edit_distance(lower_token, cand.lower()))
            corrected.append(best if nltk.edit_distance(lower_token, best.lower()) <= 2 else lower_token)
        return corrected

    def generate_drop_candidates(self, tokens: List[str], term_weights: Dict[str, float]) -> List[str]:
        if len(tokens) <= 1:
            return [" ".join(tokens)] if tokens else []

        sorted_tokens = sorted(tokens, key=lambda token: term_weights.get(token, 0.0))
        candidates = [" ".join(tokens)]
        for k in range(1, min(3, len(sorted_tokens))):
            dropped_terms = set(sorted_tokens[:k])
            dropped = [token for token in tokens if token not in dropped_terms]
            if dropped:
                candidates.append(" ".join(dropped))
        return list(dict.fromkeys(candidates))

    def rewrite_query(self, tokens: List[str], term_weights: Dict[str, float]) -> List[str]:
        candidates = [" ".join(tokens)] if tokens else []
        candidates.extend(self.generate_drop_candidates(tokens, term_weights))
        synonym_expansion = self.expand_synonyms(tokens)
        if synonym_expansion:
            candidates.append(" ".join(synonym_expansion))
        return [query for query in dict.fromkeys(candidates) if query]

    def process_query(self, text: str, max_chars: int = 64) -> Dict[str, object]:
        raw = self.preprocess(text, max_chars=max_chars)
        tokens = self.tokenize(raw)
        corrected_tokens = self.simple_error_correction(tokens)
        filtered_tokens = self.remove_stopwords(corrected_tokens)
        lemmas = self.lemmatize(filtered_tokens)
        pos_tags = self.pos_tag(filtered_tokens)
        term_weights = self.calculate_term_weights(lemmas)
        chunk_tags = self.analyze_chunks(filtered_tokens, pos_tags)
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
            "chunk_tags": chunk_tags,
            "category_intent": self.detect_category_intent(chunk_tags),
            "term_weights": term_weights,
            "intents": intents,
            "entities": entities,
            "rewrites": rewrites,
            "bigrams": bigrams,
        }


if __name__ == "__main__":
    processor = SearchQueryProcessor()
    processor.set_idf({"london": 2.5, "museum": 1.5, "historic": 1.8, "famous": 1.2})
    processor.update_click_log(["london", "museum", "historic"])
    result = processor.process_query("London famous museum 😊 I want to see history", max_chars=64)
    for key, value in result.items():
        print(f"{key}: {value}")
