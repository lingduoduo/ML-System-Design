from __future__ import annotations

from dataclasses import dataclass

from training_pipeline import FineTunedLLM


@dataclass
class Deployer:
    endpoint: str = "/generate"

    def deploy(self, model: FineTunedLLM) -> dict:
        return {"status": "deployed", "model": model.model_name, "endpoint": self.endpoint}
