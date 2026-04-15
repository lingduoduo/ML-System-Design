import re
from functools import lru_cache
from typing import List, Optional, Sequence

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

try:
    from num2words import num2words
except ImportError:
    num2words = None

try:
    from autocorrect import Speller
except ImportError:
    Speller = None

try:
    import spacy
    from spacy import displacy
except ImportError:
    spacy = None
    displacy = None

_CLEAN_RE = re.compile(r"[\n\r\t-]+")
_SUBJECT_RE = re.compile(r"<SUBJECT LINE>(.*?)<END>", flags=re.S)
_BODY_RE = re.compile(r"<BODY TEXT>(.*?)<END>", flags=re.S)
_DIGITS_RE = re.compile(r"\b(\d+)(st|nd|rd|th)?\b", flags=re.IGNORECASE)
_WORD_RE = re.compile(r"[A-Za-z]+")
_PUNCTUATION_RE = re.compile(r"[^ A-Za-z0-9]+")

_DEFAULT_STOPWORDS = frozenset({
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
})

try:
    _STOPWORDS = frozenset(stopwords.words("english"))
except Exception:
    _STOPWORDS = _DEFAULT_STOPWORDS

_STEMMER = PorterStemmer()
_LEMMATIZER = WordNetLemmatizer()

if Speller is not None:
    try:
        _SPELLER = Speller(lang="en")
    except Exception:
        _SPELLER = None
else:
    _SPELLER = None

_SPACY_MODEL = None


def ensure_nltk_resources() -> None:
    """Download the minimum set of NLTK resources required by this module."""
    resource_checks = {
        "punkt": "tokenizers/punkt",
        "stopwords": "corpora/stopwords",
        "wordnet": "corpora/wordnet",
    }
    for package, path in resource_checks.items():
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(package, quiet=True)


def decode(text: str) -> str:
    """Extract subject and body text from tagged input and normalize whitespace."""
    text = _CLEAN_RE.sub(" ", text)
    parts: List[str] = []

    subject_match = _SUBJECT_RE.search(text)
    if subject_match:
        parts.append(subject_match.group(1).strip())

    body_match = _BODY_RE.search(text)
    if body_match:
        parts.append(body_match.group(1).strip())

    if not parts:
        return text.strip()

    return ". ".join(parts).strip()


def digits_to_words(match: re.Match) -> str:
    """Convert digits to words, preserving ordinal form when present."""
    number = match.group(1)
    suffix = match.group(2)
    if num2words is None:
        return match.group(0)

    if suffix:
        return num2words(number, to="ordinal")
    return num2words(number, to="cardinal")


def spelling_correction(text: str) -> str:
    """Correct text spelling using autocorrect if available."""
    if _SPELLER is None:
        return text
    return _WORD_RE.sub(lambda m: _SPELLER(m.group(0)), text)


def remove_stop_words(text: str) -> str:
    """Remove English stopwords from a text string."""
    return " ".join(word for word in text.split() if word.lower() not in _STOPWORDS)


def stemming(text: str) -> str:
    """Stem each word individually."""
    return " ".join(_STEMMER.stem(word) for word in text.split())


def lemmatizing(text: str) -> str:
    """Lemmatize each word individually."""
    lemmas: List[str] = []
    for word in text.split():
        try:
            lemmas.append(_LEMMATIZER.lemmatize(word.lower()))
        except LookupError:
            lemmas.append(word.lower())
    return " ".join(lemmas)


def normalize_tokens(tokens: Sequence[str]) -> List[str]:
    """Normalize a token sequence with the same pipeline used for raw text."""
    return [token for token in preprocessing(" ".join(tokens), max_chars=None).split() if token]


def _get_spacy_model():
    global _SPACY_MODEL
    if spacy is None:
        return None
    if _SPACY_MODEL is None:
        try:
            _SPACY_MODEL = spacy.load("en_core_web_sm")
        except OSError:
            _SPACY_MODEL = None
    return _SPACY_MODEL


def extract_companies(text: str, render: bool = False) -> List[str]:
    """Extract organization names using spaCy NER, with optional Jupyter rendering."""
    model = _get_spacy_model()
    if model is None:
        return []
    doc = model(text)
    if render and displacy is not None:
        displacy.render(doc, style="ent", jupyter=True)
    return [ent.text for ent in doc.ents if ent.label_ == "ORG"]


@lru_cache(maxsize=4096)
def _preprocess_cached(input_text: str) -> str:
    output = decode(input_text or "")
    output = output.lower()
    output = _DIGITS_RE.sub(digits_to_words, output)
    output = _PUNCTUATION_RE.sub("", output)
    output = spelling_correction(output)
    output = remove_stop_words(output)
    return lemmatizing(output)


def preprocessing(input_text: str, debug: bool = False, max_chars: Optional[int] = 64) -> str:
    """Run a compact text preprocessing pipeline."""
    raw_text = input_text or ""

    if debug:
        output = decode(raw_text)
        print("\nDecode/remove encoding:\n        ", output)

        output = output.lower()
        print("\nLower casing:\n        ", output)

        output = _DIGITS_RE.sub(digits_to_words, output)
        print("\nDigits to words:\n        ", output)

        output = _PUNCTUATION_RE.sub("", output)
        print("\nRemove punctuations and other special characters:\n        ", output)

        output = spelling_correction(output)
        print("\nSpelling corrections:\n        ", output)

        output = remove_stop_words(output)
        print("\nRemove stop words:\n        ", output)

        output = lemmatizing(output)
        print("\nLemmatizing:\n        ", output)
    else:
        output = _preprocess_cached(raw_text)

    return output[:max_chars] if max_chars is not None else output
