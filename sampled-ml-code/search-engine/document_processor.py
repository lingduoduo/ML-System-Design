import json
import os
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence

try:
    import yaml
except ImportError:
    yaml = None


DEFAULT_CONFIG = {
    "dify": {
        "process_rules": {
            "pre_processing": [
                {"id": "remove_extra_spaces", "enabled": True},
                {"id": "remove_urls_emails", "enabled": True},
            ],
            "segmentation": {
                "separator": r"(?<=[.!?])\s+|\n+",
            },
        }
    }
}


class TextProcessor:
    """Config-driven document preprocessing for plain text and JSON payloads."""

    NON_CONTENT_FIELDS = {
        "id",
        "dataset_id",
        "document_id",
        "segment_id",
        "node_id",
        "document_enabled",
        "segment_enabled",
        "rank",
        "score",
        "bm25_score",
        "recall_score",
        "type",
        "partner",
    }

    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or os.path.join(os.path.dirname(__file__), "config.yaml")
        self.config = self._load_config()["dify"]["process_rules"]
        self._separator_pattern = re.compile(
            self.config.get("segmentation", {}).get("separator", r"(?<=[.!?])\s+|\n+")
        )
        self._url_email_pattern = re.compile(r"https?://\S+|\b\S+@\S+\.\S+\b")
        self._token_pattern = re.compile(r"\b\w+\b")

    def _load_config(self) -> Dict[str, Any]:
        if os.path.exists(self.config_path) and yaml is not None:
            with open(self.config_path, "r", encoding="utf-8") as handle:
                return yaml.safe_load(handle)
        return DEFAULT_CONFIG

    def _is_rule_enabled(self, rule_index: int, default: bool = False) -> bool:
        rules = self.config.get("pre_processing", [])
        if rule_index >= len(rules):
            return default
        return bool(rules[rule_index].get("enabled", default))

    def preprocess(self, text: str) -> str:
        """Apply basic cleanup rules before segmentation."""
        text = str(text or "")

        if self._is_rule_enabled(0, default=True):
            text = re.sub(r"\s+", " ", text).strip()

        if self._is_rule_enabled(1, default=True):
            text = self._url_email_pattern.sub("", text)
            text = re.sub(r"\s+", " ", text).strip()

        return text

    def segment(self, text: str) -> List[str]:
        """Split cleaned text into non-empty segments."""
        return [
            segment.strip()
            for segment in self._separator_pattern.split(text)
            if segment and segment.strip()
        ]

    def preprocess_json(self, json_data: Dict[str, Any]) -> List[str]:
        """Extract `rec_texts` from a JSON payload and clean each entry."""
        rec_texts = json_data.get("rec_texts", [])
        processed_texts = []

        for text in rec_texts:
            processed_text = self.preprocess(text)
            if processed_text:
                processed_texts.append(processed_text)

        return processed_texts

    def _extract_text_values(self, input_data: Any, text_fields: Optional[Sequence[str]] = None) -> List[str]:
        if isinstance(input_data, str):
            return [input_data]

        if not isinstance(input_data, dict):
            return []

        if text_fields:
            values: Iterable[Any] = (input_data.get(field, "") for field in text_fields)
        elif "rec_texts" in input_data:
            values = input_data.get("rec_texts", [])
        else:
            values = (
                value
                for key, value in input_data.items()
                if key not in self.NON_CONTENT_FIELDS and isinstance(value, (str, int, float)) and not isinstance(value, bool)
            )

        extracted = []
        for value in values:
            if value is None:
                continue
            text = self.preprocess(str(value))
            if text:
                extracted.append(text)
        return extracted

    def process(self, input_data: Any) -> List[str]:
        """
        Unified entry point for text or JSON payloads.

        Returns segmented text chunks.
        """
        extracted_texts = self._extract_text_values(input_data)
        combined_text = " ".join(extracted_texts).strip()
        if combined_text:
            return self.segment(combined_text)

        return []

    def normalize_document(self, input_data: Any, text_fields: Optional[Sequence[str]] = None) -> str:
        """Convert supported inputs into one cleaned document string."""
        extracted_texts = self._extract_text_values(input_data, text_fields=text_fields)
        return " ".join(extracted_texts).strip()

    def tokenize(self, input_data: Any, text_fields: Optional[Sequence[str]] = None) -> List[str]:
        """Normalize and tokenize text with one shared retrieval-friendly path."""
        normalized_text = self.normalize_document(input_data, text_fields=text_fields).lower()
        return self.tokenize_normalized(normalized_text)

    def tokenize_normalized(self, normalized_text: str) -> List[str]:
        """Tokenize already-normalized text without re-running preprocessing."""
        return self._token_pattern.findall(normalized_text)


if __name__ == "__main__":
    processor = TextProcessor()

    sample_json = {
        "rec_texts": [
            "This is a test email contact@example.com",
            "Visit https://example.com for details.",
        ]
    }

    print("Processing JSON input:")
    print(json.dumps(processor.process(sample_json), indent=2))

    sample_text = "This is a test text. It needs to be segmented."
    print("Processing plain text:")
    print(processor.process(sample_text))
