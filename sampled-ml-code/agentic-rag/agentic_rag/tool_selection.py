from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import re
from typing import List, Optional, Sequence


import torch
import torch.nn as nn
import torch.nn.functional as F


TOKEN_PATTERN = re.compile(r"[a-z0-9]+")

from .retrieval import ORDER_ID_PATTERN
from .schema import ToolCall


@dataclass(frozen=True, slots=True)
class ToolCandidate:
    tool_call: ToolCall
    score: float


DEFAULT_TRAINABLE_TOOL_NAMES = (
    "search_orders",
    "create_ticket",
    "summarize_user_docs",
    "summarize_policy_docs",
)


class GumbelSoftmaxToolSelector(nn.Module):
    def forward(self, logits: torch.Tensor, tau: float = 0.5, hard: bool = True) -> torch.Tensor:
        # Gumbel-Softmax sampling:
        # 1. Add Gumbel noise to the logits.
        # 2. Apply softmax to the noisy logits scaled by temperature.
        # 3. Higher tau increases exploration; lower tau sharpens choices.
        return F.gumbel_softmax(logits, tau=tau, hard=hard, dim=-1)


class TrainableToolSelector(nn.Module):
    def __init__(self, tool_names: Sequence[str] | None = None, input_dim: int = 64):
        super().__init__()
        self.tool_names = list(tool_names or DEFAULT_TRAINABLE_TOOL_NAMES)
        self.tool_name_to_index = {tool_name: idx for idx, tool_name in enumerate(self.tool_names)}
        self.input_dim = input_dim
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, len(self.tool_names)),
        )

    def forward(self, x: torch.Tensor, tau: float = 0.5, hard: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.model(x)
        y_soft = F.gumbel_softmax(logits, tau=tau, hard=hard, dim=-1)
        return y_soft, logits

    def predict(self, x: torch.Tensor) -> tuple[int, torch.Tensor]:
        with torch.inference_mode():
            _, logits = self.forward(x)
            return torch.argmax(logits, dim=-1).item(), logits[0]


class ToolSelectionModel:
    def __init__(
        self,
        *,
        enable_gumbel: bool = False,
        temperature: float = 0.5,
        hard: bool = True,
        trained_selector: TrainableToolSelector | None = None,
    ):
        self.enable_gumbel = enable_gumbel
        self.temperature = temperature
        self.hard = hard
        self.gumbel_selector = GumbelSoftmaxToolSelector()
        self.trained_selector = trained_selector
        if self.trained_selector is not None:
            self.trained_selector.eval()
            self._trained_tool_name_to_index = dict(self.trained_selector.tool_name_to_index)
        else:
            self._trained_tool_name_to_index = {}

    def choose_tool(self, message: str) -> Optional[ToolCall]:
        candidates = self._score_candidates(message)
        if not candidates:
            return None

        if len(candidates) == 1:
            return candidates[0].tool_call

        selected_index = self._choose_candidate_index(message, candidates)
        return candidates[selected_index].tool_call

    def debug_scores(self, message: str) -> List[dict]:
        return [
            {
                "tool_name": candidate.tool_call.tool_name,
                "score": candidate.score,
                "arguments": dict(candidate.tool_call.arguments),
            }
            for candidate in self._score_candidates(message)
        ]

    def _choose_candidate_index(self, message: str, candidates: List[ToolCandidate]) -> int:
        if self.trained_selector is not None:
            return self._choose_with_trained_selector(candidates, message)

        if not self.enable_gumbel:
            return max(range(len(candidates)), key=lambda idx: candidates[idx].score)

        logits = torch.tensor([[candidate.score for candidate in candidates]], dtype=torch.float32)
        sampled = self.gumbel_selector.forward(logits, tau=self.temperature, hard=self.hard)
        return int(torch.argmax(sampled, dim=-1).item())

    def _choose_with_trained_selector(self, candidates: List[ToolCandidate], message: str) -> int:
        assert self.trained_selector is not None

        with torch.inference_mode():
            _, logits = self.trained_selector(
                encode_query(message, input_dim=self.trained_selector.input_dim)
            )

        candidate_indices = [
            self._trained_tool_name_to_index.get(candidate.tool_call.tool_name, -1)
            for candidate in candidates
        ]
        candidate_logits = logits.new_full((1, len(candidates)), float("-inf"))
        valid_positions = [
            (position, tool_index)
            for position, tool_index in enumerate(candidate_indices)
            if tool_index >= 0
        ]
        if valid_positions:
            positions = torch.tensor([position for position, _ in valid_positions], dtype=torch.long)
            tool_indices = torch.tensor([tool_index for _, tool_index in valid_positions], dtype=torch.long)
            candidate_logits[0, positions] = logits[0, tool_indices]
        else:
            return max(range(len(candidates)), key=lambda idx: candidates[idx].score)

        if self.enable_gumbel:
            sampled = self.gumbel_selector.forward(candidate_logits, tau=self.temperature, hard=self.hard)
            return int(torch.argmax(sampled, dim=-1).item())

        return int(torch.argmax(candidate_logits, dim=-1).item())

    def _score_candidates(self, message: str) -> List[ToolCandidate]:
        msg = message.lower()
        order_id = self._extract_order_id(message)

        candidates = [
            self._candidate(
                "search_orders",
                {"order_id": order_id},
                self._score_order_status(msg, order_id),
            ),
            self._candidate(
                "create_ticket",
                {"issue": message, "severity": "medium"},
                self._score_ticket_creation(msg),
            ),
            self._candidate(
                "summarize_user_docs",
                {},
                self._score_user_summary(msg),
            ),
            self._candidate(
                "summarize_policy_docs",
                {},
                self._score_policy_summary(msg),
            ),
        ]
        return [candidate for candidate in candidates if candidate is not None]

    @staticmethod
    def _candidate(tool_name: str, arguments: dict, score: float) -> Optional[ToolCandidate]:
        if score <= 0:
            return None
        return ToolCandidate(ToolCall(tool_name=tool_name, arguments=arguments), score)

    @staticmethod
    def _score_order_status(msg: str, order_id: str) -> float:
        score = 0.0
        if "order" in msg:
            score += 1.5
        if "status" in msg or "track" in msg or "where is" in msg:
            score += 2.5
        if order_id != "ORD-001":
            score += 1.5
        return score

    @staticmethod
    def _score_ticket_creation(msg: str) -> float:
        score = 0.0
        if "ticket" in msg:
            score += 3.0
        if "issue" in msg or "problem" in msg or "support" in msg:
            score += 2.0
        if "help" in msg and "order" in msg:
            score += 1.0
        return score

    @staticmethod
    def _score_user_summary(msg: str) -> float:
        if "summarize" not in msg and "summary" not in msg:
            return 0.0
        score = 2.0
        if "user" in msg:
            score += 2.0
        if "docs" in msg or "documentation" in msg:
            score += 1.5
        return score

    @staticmethod
    def _score_policy_summary(msg: str) -> float:
        if "summarize" not in msg and "summary" not in msg:
            return 0.0
        score = 2.0
        if "policy" in msg or "refund" in msg:
            score += 2.0
        return score

    def _extract_order_id(self, message: str) -> str:
        match = ORDER_ID_PATTERN.search(message)
        if match:
            return match.group(0).upper()
        return "ORD-001"


@lru_cache(maxsize=2048)
def _encode_query_cached(query: str, input_dim: int) -> tuple[float, ...]:
    features = [0.0] * input_dim
    for token in TOKEN_PATTERN.findall(query.lower()):
        features[hash(token) % input_dim] += 1.0

    norm = sum(value * value for value in features) ** 0.5
    if norm > 0:
        return tuple(value / norm for value in features)
    return tuple(features)


def encode_query(query: str, input_dim: int = 64) -> torch.Tensor:
    encoded = _encode_query_cached(query, input_dim)
    return torch.tensor(encoded, dtype=torch.float32).unsqueeze(0)


def train_model(
    train_data: Sequence[tuple[str, str]],
    epochs: int = 5,
    *,
    input_dim: int = 64,
    learning_rate: float = 0.01,
    tool_names: Sequence[str] | None = None,
) -> TrainableToolSelector:
    if not train_data:
        raise ValueError("train_data must not be empty")

    selector = TrainableToolSelector(tool_names=tool_names, input_dim=input_dim)
    optimizer = torch.optim.Adam(selector.parameters(), lr=learning_rate)
    tool_name_to_index = selector.tool_name_to_index

    print("\nPrediction before training:")
    test_query = train_data[0][0]
    idx, _ = selector.predict(encode_query(test_query, input_dim=input_dim))
    print(f"Query: '{test_query}'")
    print(f"Predicted tool: {selector.tool_names[idx]}")

    print("\nStart training...")
    for epoch in range(epochs):
        total_loss = 0.0
        tau = max(0.5, 1.0 - epoch * 0.1)

        for step, (query, target_tool) in enumerate(train_data):
            tool_idx = tool_name_to_index.get(target_tool)
            if tool_idx is None:
                raise ValueError(f"Unknown target tool '{target_tool}'")

            x = encode_query(query, input_dim=input_dim)
            _, logits = selector(x, tau=tau)

            target_index = torch.tensor([tool_idx], dtype=torch.long)
            loss = F.cross_entropy(logits, target_index)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if step % 5 == 0 or step == len(train_data) - 1:
                probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()[0]
                formatted_probs = " ".join(
                    f"{tool_name}:{prob:.3f}"
                    for tool_name, prob in zip(selector.tool_names, probs)
                )
                print(
                    f"[Epoch {epoch}, Step {step}] Query='{query}' | "
                    f"{formatted_probs} | Loss: {loss.item():.4f} | tau={tau:.2f}"
                )

        print(f"Epoch {epoch} average loss: {total_loss / len(train_data):.4f}")

    return selector


def save_model(selector: TrainableToolSelector, path: str | Path) -> None:
    checkpoint = {
        "tool_names": selector.tool_names,
        "input_dim": selector.input_dim,
        "state_dict": selector.state_dict(),
    }
    torch.save(checkpoint, Path(path))


def load_model(path: str | Path) -> TrainableToolSelector:
    checkpoint = torch.load(Path(path), map_location="cpu")
    selector = TrainableToolSelector(
        tool_names=checkpoint["tool_names"],
        input_dim=checkpoint["input_dim"],
    )
    selector.load_state_dict(checkpoint["state_dict"])
    selector.eval()
    return selector
