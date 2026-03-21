import math
import random
from dataclasses import dataclass
from typing import Dict, List, Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ============================================================
# Config
# ============================================================

@dataclass
class Config:
    num_channels: int = 4                 # e.g. email, push, in-app, sms
    seq_len: int = 20
    action_vocab_size: int = 50           # action/event ids
    action_embed_dim: int = 16

    user_dense_dim: int = 16
    item_dense_dim: int = 12
    campaign_dense_dim: int = 10
    graph_embed_dim: int = 32

    hidden_dim: int = 64
    transformer_heads: int = 4
    transformer_layers: int = 2
    dropout: float = 0.1

    batch_size: int = 64
    lr: float = 1e-3
    epochs: int = 3
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    lambda_unsub: float = 2.0
    lambda_cost: float = 0.3


CFG = Config()


# ============================================================
# Fake sampled data
# Replace this with your real feature pipeline / parquet / warehouse reads
# ============================================================

def generate_fake_row(cfg: Config) -> Dict[str, Any]:
    """
    One row = one candidate send/impression opportunity
    for (user, item/campaign, channel, time).
    """
    user_dense = np.random.randn(cfg.user_dense_dim).astype(np.float32)
    item_dense = np.random.randn(cfg.item_dense_dim).astype(np.float32)
    campaign_dense = np.random.randn(cfg.campaign_dense_dim).astype(np.float32)
    graph_embed = np.random.randn(cfg.graph_embed_dim).astype(np.float32)

    channel_id = np.random.randint(0, cfg.num_channels)

    seq_actions = np.random.randint(1, cfg.action_vocab_size, size=cfg.seq_len)
    seq_mask = np.ones(cfg.seq_len, dtype=np.float32)

    # Simulate some dependency for labels
    base_click_logit = (
        0.25 * user_dense[0]
        + 0.15 * item_dense[0]
        + 0.10 * campaign_dense[0]
        + 0.10 * (channel_id == 0)
    )
    p_click = 1 / (1 + np.exp(-base_click_logit))

    click = np.random.binomial(1, min(max(p_click, 0.01), 0.95))

    base_conv_logit = (
        -1.0
        + 0.30 * click
        + 0.20 * user_dense[1]
        + 0.20 * item_dense[1]
    )
    p_conv = 1 / (1 + np.exp(-base_conv_logit))
    conversion = np.random.binomial(1, min(max(p_conv, 0.005), 0.8))

    base_unsub_logit = (
        -2.0
        + 0.25 * campaign_dense[1]
        + 0.15 * (channel_id == 3)
        - 0.10 * user_dense[2]
    )
    p_unsub = 1 / (1 + np.exp(-base_unsub_logit))
    unsubscribe = np.random.binomial(1, min(max(p_unsub, 0.001), 0.5))

    margin = np.float32(np.random.uniform(5.0, 50.0))
    send_cost = np.float32([0.02, 0.01, 0.005, 0.03][channel_id])

    return {
        "user_dense": user_dense,
        "item_dense": item_dense,
        "campaign_dense": campaign_dense,
        "graph_embed": graph_embed,
        "channel_id": np.int64(channel_id),
        "seq_actions": seq_actions.astype(np.int64),
        "seq_mask": seq_mask,
        "click": np.float32(click),
        "conversion": np.float32(conversion),
        "unsubscribe": np.float32(unsubscribe),
        "margin": margin,
        "send_cost": send_cost,
    }


def generate_fake_dataset(n: int, cfg: Config) -> List[Dict[str, Any]]:
    return [generate_fake_row(cfg) for _ in range(n)]


# ============================================================
# Dataset
# ============================================================

class CampaignDataset(Dataset):
    def __init__(self, rows: List[Dict[str, Any]]):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.rows[idx]
        return {
            "user_dense": torch.tensor(row["user_dense"], dtype=torch.float32),
            "item_dense": torch.tensor(row["item_dense"], dtype=torch.float32),
            "campaign_dense": torch.tensor(row["campaign_dense"], dtype=torch.float32),
            "graph_embed": torch.tensor(row["graph_embed"], dtype=torch.float32),
            "channel_id": torch.tensor(row["channel_id"], dtype=torch.long),
            "seq_actions": torch.tensor(row["seq_actions"], dtype=torch.long),
            "seq_mask": torch.tensor(row["seq_mask"], dtype=torch.float32),
            "click": torch.tensor(row["click"], dtype=torch.float32),
            "conversion": torch.tensor(row["conversion"], dtype=torch.float32),
            "unsubscribe": torch.tensor(row["unsubscribe"], dtype=torch.float32),
            "margin": torch.tensor(row["margin"], dtype=torch.float32),
            "send_cost": torch.tensor(row["send_cost"], dtype=torch.float32),
        }


# ============================================================
# Model parts
# ============================================================

class SequenceEncoder(nn.Module):
    """
    Encodes recent user actions using a small Transformer encoder.
    """

    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int,
                 n_heads: int, n_layers: int, dropout: float):
        super().__init__()
        self.action_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.proj = nn.Linear(embed_dim, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def forward(self, seq_actions: torch.Tensor, seq_mask: torch.Tensor) -> torch.Tensor:
        """
        seq_actions: [B, T]
        seq_mask:    [B, T], 1 for valid, 0 for padded
        """
        x = self.action_embedding(seq_actions)      # [B, T, E]
        x = self.proj(x)                            # [B, T, H]

        key_padding_mask = seq_mask == 0            # True means ignore
        x = self.encoder(x, src_key_padding_mask=key_padding_mask)

        # masked mean pooling
        mask = seq_mask.unsqueeze(-1)               # [B, T, 1]
        x = x * mask
        pooled = x.sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        return pooled                               # [B, H]


class DenseTower(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MultiTaskCampaignModel(nn.Module):
    """
    Multi-task model:
      - click probability
      - conversion probability
      - unsubscribe probability
    """
    def __init__(self, cfg: Config):
        super().__init__()

        self.channel_embedding = nn.Embedding(cfg.num_channels, 8)

        self.user_tower = DenseTower(cfg.user_dense_dim, cfg.hidden_dim, cfg.dropout)
        self.item_tower = DenseTower(cfg.item_dense_dim, cfg.hidden_dim, cfg.dropout)
        self.campaign_tower = DenseTower(cfg.campaign_dense_dim, cfg.hidden_dim, cfg.dropout)
        self.graph_tower = DenseTower(cfg.graph_embed_dim, cfg.hidden_dim, cfg.dropout)

        self.seq_encoder = SequenceEncoder(
            vocab_size=cfg.action_vocab_size,
            embed_dim=cfg.action_embed_dim,
            hidden_dim=cfg.hidden_dim,
            n_heads=cfg.transformer_heads,
            n_layers=cfg.transformer_layers,
            dropout=cfg.dropout,
        )

        fused_dim = cfg.hidden_dim * 5 + 8

        self.shared_mlp = nn.Sequential(
            nn.Linear(fused_dim, cfg.hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim * 2, cfg.hidden_dim),
            nn.ReLU(),
        )

        self.click_head = nn.Linear(cfg.hidden_dim, 1)
        self.conv_head = nn.Linear(cfg.hidden_dim, 1)
        self.unsub_head = nn.Linear(cfg.hidden_dim, 1)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        user_vec = self.user_tower(batch["user_dense"])
        item_vec = self.item_tower(batch["item_dense"])
        campaign_vec = self.campaign_tower(batch["campaign_dense"])
        graph_vec = self.graph_tower(batch["graph_embed"])
        seq_vec = self.seq_encoder(batch["seq_actions"], batch["seq_mask"])
        channel_vec = self.channel_embedding(batch["channel_id"])

        fused = torch.cat(
            [user_vec, item_vec, campaign_vec, graph_vec, seq_vec, channel_vec],
            dim=-1
        )

        shared = self.shared_mlp(fused)

        click_logit = self.click_head(shared).squeeze(-1)
        conv_logit = self.conv_head(shared).squeeze(-1)
        unsub_logit = self.unsub_head(shared).squeeze(-1)

        click_prob = torch.sigmoid(click_logit)
        conv_prob = torch.sigmoid(conv_logit)
        unsub_prob = torch.sigmoid(unsub_logit)

        return {
            "click_logit": click_logit,
            "conv_logit": conv_logit,
            "unsub_logit": unsub_logit,
            "click_prob": click_prob,
            "conv_prob": conv_prob,
            "unsub_prob": unsub_prob,
        }


# ============================================================
# Loss + scoring
# ============================================================

class MultiTaskLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        click_loss = self.bce(outputs["click_logit"], batch["click"])
        conv_loss = self.bce(outputs["conv_logit"], batch["conversion"])
        unsub_loss = self.bce(outputs["unsub_logit"], batch["unsubscribe"])

        # You can tune these weights
        total_loss = 1.0 * click_loss + 1.0 * conv_loss + 0.5 * unsub_loss

        return {
            "total_loss": total_loss,
            "click_loss": click_loss,
            "conv_loss": conv_loss,
            "unsub_loss": unsub_loss,
        }


def expected_value_score(
    conv_prob: torch.Tensor,
    unsub_prob: torch.Tensor,
    margin: torch.Tensor,
    send_cost: torch.Tensor,
    lambda_unsub: float,
    lambda_cost: float,
) -> torch.Tensor:
    """
    Example ranking score:
      EV = P(conv) * margin - lambda1 * P(unsub) - lambda2 * cost
    """
    return conv_prob * margin - lambda_unsub * unsub_prob - lambda_cost * send_cost


# ============================================================
# Training / Eval
# ============================================================

def move_batch_to_device(batch: Dict[str, torch.Tensor], device: str) -> Dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in batch.items()}


def train_one_epoch(model, loader, optimizer, criterion, cfg: Config):
    model.train()
    running = 0.0

    for batch in loader:
        batch = move_batch_to_device(batch, cfg.device)

        outputs = model(batch)
        losses = criterion(outputs, batch)

        optimizer.zero_grad()
        losses["total_loss"].backward()
        optimizer.step()

        running += losses["total_loss"].item()

    return running / max(len(loader), 1)


@torch.no_grad()
def evaluate(model, loader, criterion, cfg: Config):
    model.eval()
    running = 0.0

    all_scores = []
    all_click_probs = []
    all_conv_probs = []
    all_unsub_probs = []

    for batch in loader:
        batch = move_batch_to_device(batch, cfg.device)
        outputs = model(batch)
        losses = criterion(outputs, batch)
        running += losses["total_loss"].item()

        score = expected_value_score(
            conv_prob=outputs["conv_prob"],
            unsub_prob=outputs["unsub_prob"],
            margin=batch["margin"],
            send_cost=batch["send_cost"],
            lambda_unsub=cfg.lambda_unsub,
            lambda_cost=cfg.lambda_cost,
        )

        all_scores.append(score.cpu())
        all_click_probs.append(outputs["click_prob"].cpu())
        all_conv_probs.append(outputs["conv_prob"].cpu())
        all_unsub_probs.append(outputs["unsub_prob"].cpu())

    metrics = {
        "loss": running / max(len(loader), 1),
        "mean_score": torch.cat(all_scores).mean().item(),
        "mean_click_prob": torch.cat(all_click_probs).mean().item(),
        "mean_conv_prob": torch.cat(all_conv_probs).mean().item(),
        "mean_unsub_prob": torch.cat(all_unsub_probs).mean().item(),
    }
    return metrics


# ============================================================
# Example inference helper
# ============================================================

@torch.no_grad()
def rank_candidates(model: nn.Module, candidate_batch: Dict[str, torch.Tensor], cfg: Config):
    model.eval()
    candidate_batch = move_batch_to_device(candidate_batch, cfg.device)

    outputs = model(candidate_batch)

    scores = expected_value_score(
        conv_prob=outputs["conv_prob"],
        unsub_prob=outputs["unsub_prob"],
        margin=candidate_batch["margin"],
        send_cost=candidate_batch["send_cost"],
        lambda_unsub=cfg.lambda_unsub,
        lambda_cost=cfg.lambda_cost,
    )

    return {
        "scores": scores.cpu(),
        "click_prob": outputs["click_prob"].cpu(),
        "conv_prob": outputs["conv_prob"].cpu(),
        "unsub_prob": outputs["unsub_prob"].cpu(),
    }


# ============================================================
# Main
# ============================================================

def main():
    train_rows = generate_fake_dataset(3000, CFG)
    valid_rows = generate_fake_dataset(800, CFG)

    train_ds = CampaignDataset(train_rows)
    valid_ds = CampaignDataset(valid_rows)

    train_loader = DataLoader(train_ds, batch_size=CFG.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=CFG.batch_size, shuffle=False)

    model = MultiTaskCampaignModel(CFG).to(CFG.device)
    criterion = MultiTaskLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=CFG.lr)

    for epoch in range(CFG.epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, CFG)
        valid_metrics = evaluate(model, valid_loader, criterion, CFG)

        print(
            f"Epoch {epoch+1}/{CFG.epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"valid_loss={valid_metrics['loss']:.4f} | "
            f"mean_score={valid_metrics['mean_score']:.4f}"
        )

    # Example: rank a small batch of candidates
    sample_batch = next(iter(valid_loader))
    pred = rank_candidates(model, sample_batch, CFG)
    print("\nSample ranking scores:", pred["scores"][:10])


if __name__ == "__main__":
    main()