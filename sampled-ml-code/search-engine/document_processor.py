import json
import os
import re
from typing import Any, Dict, List, Optional

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

    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or os.path.join(os.path.dirname(__file__), "config.yaml")
        self.config = self._load_config()["dify"]["process_rules"]
        self._separator_pattern = re.compile(
            self.config.get("segmentation", {}).get("separator", r"(?<=[.!?])\s+|\n+")
        )

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
            text = re.sub(r"https?://\S+|\b\S+@\S+\.\S+\b", "", text)
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

    def process(self, input_data: Any) -> List[str]:
        """
        Unified entry point for text or JSON payloads.

        Returns segmented text chunks.
        """
        if isinstance(input_data, dict):
            cleaned_texts = self.preprocess_json(input_data)
            combined_text = " ".join(cleaned_texts).strip()
            return self.segment(combined_text) if combined_text else []

        if isinstance(input_data, str):
            cleaned_text = self.preprocess(input_data)
            return self.segment(cleaned_text)

        return []

    def normalize_document(self, input_data: Any) -> str:
        """Convert supported inputs into one cleaned document string."""
        return " ".join(self.process(input_data)).strip()


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
