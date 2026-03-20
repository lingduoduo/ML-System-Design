from __future__ import annotations

import random
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

try:
    # Allows PyTorch to use faster float32 matmul kernels when available.
    torch.set_float32_matmul_precision("high")
except AttributeError:
    # Older PyTorch versions do not expose this optimization knob.
    pass


def seed_everything(seed: int = 42) -> None:
    """Seed Python and PyTorch so runs are repeatable."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class Config:
    """Central place for data, model, and training hyperparameters."""

    num_action_types: int = 10
    min_history_len: int = 12
    max_history_len: int = 60
    profile_dim: int = 8
    graph_dim: int = 6

    train_size: int = 2500
    test_size: int = 500
    batch_size: int = 96
    epochs: int = 6
    lr: float = 1e-3
    weight_decay: float = 1e-2
    grad_clip: float = 1.0
    alpha: float = 1.0
    beta: float = 0.35
    max_exposure_weight: float = 4.0

    action_embed_dim: int = 48
    hidden_dim: int = 96
    num_heads: int = 4
    num_layers: int = 2
    ff_dim: int = 192
    dropout: float = 0.1

    num_workers: int = 0
    amp: bool = True
    compile_model: bool = False
    seed: int = 42


class BotDetectionDataset(Dataset):
    """Synthetic dataset that generates account behavior on the fly.

    The dataset stores only light-weight summary tensors up front, then
    re-creates per-account histories deterministically from `seed + idx`.
    This keeps memory usage low while making the task learnable.
    """

    def __init__(
        self,
        num_samples: int,
        num_action_types: int = 10,
        min_history_len: int = 12,
        max_history_len: int = 60,
        profile_dim: int = 8,
        graph_dim: int = 6,
        seed: int = 42,
    ):
        self.num_samples = num_samples
        self.num_action_types = num_action_types
        self.min_history_len = min_history_len
        self.max_history_len = max_history_len
        self.profile_dim = profile_dim
        self.graph_dim = graph_dim
        self.seed = seed

        gen = torch.Generator().manual_seed(seed)
        self.history_lens = torch.randint(
            min_history_len,
            max_history_len + 1,
            (num_samples,),
            generator=gen,
        )

        # Pre-generate compact account-level summaries once.
        self.profile_base = torch.rand(num_samples, profile_dim, generator=gen)
        self.graph_base = torch.rand(num_samples, graph_dim, generator=gen)
        self.exposures = torch.randint(50, 10001, (num_samples,), generator=gen).float()

        account_score = self._base_risk_score(self.profile_base, self.graph_base)
        noise = 0.15 * torch.randn(num_samples, generator=gen)
        risk_score = torch.sigmoid(account_score + noise)

        self.bot_label = (risk_score > 0.58).float()
        self.risk_score = risk_score.float()

    def _action_probabilities(self, is_bot: bool) -> torch.Tensor:
        """Return action probabilities that match `num_action_types`."""
        if is_bot:
            base_probs = [0.03, 0.24, 0.03, 0.21, 0.16, 0.18, 0.02, 0.08, 0.03, 0.02]
        else:
            base_probs = [0.10, 0.07, 0.16, 0.07, 0.10, 0.08, 0.13, 0.11, 0.08, 0.10]

        probs = torch.full((self.num_action_types,), 1.0 / self.num_action_types, dtype=torch.float32)
        limit = min(self.num_action_types, len(base_probs))
        probs[:limit] = torch.tensor(base_probs[:limit], dtype=torch.float32)
        probs = probs / probs.sum()
        return probs

    def __len__(self) -> int:
        return self.num_samples

    def _base_risk_score(self, profile_base: torch.Tensor, graph_base: torch.Tensor) -> torch.Tensor:
        # High posting intensity, low personalization, low reciprocity, and
        # unusually dense outbound activity are all common bot signals.
        posting_intensity = profile_base[:, 0]
        profile_completeness = profile_base[:, 1]
        device_diversity = profile_base[:, 2]
        session_regularity = profile_base[:, 3]
        text_uniqueness = profile_base[:, 4]

        out_degree_ratio = graph_base[:, 0]
        reciprocity = graph_base[:, 1]
        burstiness = graph_base[:, 2]
        follow_churn = graph_base[:, 3]

        return (
            2.1 * posting_intensity
            - 1.3 * profile_completeness
            - 1.0 * device_diversity
            + 1.8 * session_regularity
            - 1.4 * text_uniqueness
            + 1.6 * out_degree_ratio
            - 1.5 * reciprocity
            + 1.7 * burstiness
            + 1.0 * follow_churn
            - 1.2
        )

    def __getitem__(self, idx: int):
        gen = torch.Generator().manual_seed(self.seed + idx)
        history_len = int(self.history_lens[idx])

        profile_feats = self.profile_base[idx].clone()
        graph_feats = self.graph_base[idx].clone()
        bot_label = self.bot_label[idx]

        # Bot-like accounts over-index on repetitive follow / post / like loops.
        action_probs = self._action_probabilities(is_bot=bot_label.item() > 0.5)

        action_ids = torch.multinomial(action_probs, history_len, replacement=True, generator=gen) + 1
        hourly_gaps = torch.rand(history_len, generator=gen)
        if bot_label.item() > 0.5:
            hourly_gaps = hourly_gaps * 0.35
        else:
            hourly_gaps = 0.2 + hourly_gaps * 1.4

        stats = torch.stack(
            [
                hourly_gaps,
                torch.linspace(0.0, 1.0, history_len),
                (action_ids == 2).float(),
                (action_ids == 4).float(),
            ],
            dim=-1,
        )

        return {
            "action_ids": action_ids,
            "action_stats": stats,
            "profile_feats": profile_feats.float(),
            "graph_feats": graph_feats.float(),
            "bot_label": bot_label.float(),
            "risk_score": self.risk_score[idx].float(),
            "exposures": self.exposures[idx].float(),
        }


def collate_fn(batch):
    """Pad variable-length histories and stack account-level features."""
    action_ids = [sample["action_ids"] for sample in batch]
    action_stats = [sample["action_stats"] for sample in batch]

    padded_ids = pad_sequence(action_ids, batch_first=True, padding_value=0)
    padded_stats = pad_sequence(action_stats, batch_first=True, padding_value=0.0)
    action_mask = (padded_ids != 0).long()

    return {
        "action_ids": padded_ids,
        "action_stats": padded_stats,
        "action_mask": action_mask,
        "profile_feats": torch.stack([sample["profile_feats"] for sample in batch]),
        "graph_feats": torch.stack([sample["graph_feats"] for sample in batch]),
        "bot_label": torch.stack([sample["bot_label"] for sample in batch]),
        "risk_score": torch.stack([sample["risk_score"] for sample in batch]),
        "exposures": torch.stack([sample["exposures"] for sample in batch]),
    }


class BotDetectionModel(nn.Module):
    """Fuse action history with profile and graph features."""

    def __init__(
        self,
        num_action_types: int,
        profile_dim: int,
        graph_dim: int,
        action_embed_dim: int = 48,
        hidden_dim: int = 96,
        num_heads: int = 4,
        num_layers: int = 2,
        ff_dim: int = 192,
        max_history_len: int = 60,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.action_embedding = nn.Embedding(num_action_types + 1, action_embed_dim, padding_idx=0)
        self.action_proj = nn.Linear(action_embed_dim + 4, hidden_dim)
        self.max_history_len = max_history_len
        self.pos_embedding = nn.Parameter(torch.randn(1, max_history_len, hidden_dim) * 0.02)
        self.sequence_norm = nn.LayerNorm(hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.history_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        dense_input_dim = profile_dim + graph_dim
        self.account_mlp = nn.Sequential(
            nn.Linear(dense_input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        self.fusion_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        self.bot_head = nn.Linear(hidden_dim, 1)
        self.risk_head = nn.Linear(hidden_dim, 1)

    def forward(self, action_ids, action_stats, action_mask, profile_feats, graph_feats):
        history_len = action_ids.shape[1]
        if history_len > self.max_history_len:
            raise ValueError(
                f"Received history_len={history_len}, but the model only supports "
                f"max_history_len={self.max_history_len}. Increase Config.max_history_len."
            )

        action_x = self.action_embedding(action_ids)
        history_x = torch.cat([action_x, action_stats], dim=-1)
        history_x = self.action_proj(history_x)
        history_x = history_x + self.pos_embedding[:, :history_len, :]
        history_x = self.sequence_norm(history_x)

        padding_mask = action_mask == 0
        encoded = self.history_encoder(history_x, src_key_padding_mask=padding_mask)

        action_mask_f = action_mask.unsqueeze(-1).float()
        history_repr = (encoded * action_mask_f).sum(dim=1) / action_mask_f.sum(dim=1).clamp_min(1.0)

        account_repr = self.account_mlp(torch.cat([profile_feats, graph_feats], dim=-1))
        fused = self.fusion_mlp(torch.cat([history_repr, account_repr], dim=-1))

        return {
            "bot_logit": self.bot_head(fused).squeeze(-1),
            "risk_pred": self.risk_head(fused).squeeze(-1),
        }


def compute_loss(outputs, bot_label, risk_score, exposures, alpha=1.0, beta=0.35, c=4.0):
    """Combine weighted bot classification loss with risk-score regression."""
    bot_logit = outputs["bot_logit"]
    risk_pred = outputs["risk_pred"]

    bce_per_sample = F.binary_cross_entropy_with_logits(bot_logit, bot_label, reduction="none")
    weights = torch.clamp(torch.log1p(exposures), max=c)
    l_primary = (bce_per_sample * weights).mean()
    l_aux = F.mse_loss(torch.sigmoid(risk_pred), risk_score)
    total_loss = alpha * l_primary + beta * l_aux
    return total_loss, l_primary, l_aux


@torch.no_grad()
def batch_metrics(outputs, bot_label, risk_score):
    """Compute simple batch-level monitoring metrics."""
    bot_prob = torch.sigmoid(outputs["bot_logit"])
    bot_pred = (bot_prob >= 0.5).float()

    acc = (bot_pred == bot_label).float().mean().item()
    precision = ((bot_pred * bot_label).sum() / bot_pred.sum().clamp_min(1.0)).item()
    recall = ((bot_pred * bot_label).sum() / bot_label.sum().clamp_min(1.0)).item()
    mse = F.mse_loss(torch.sigmoid(outputs["risk_pred"]), risk_score).item()
    return acc, precision, recall, mse


def move_batch_to_device(batch, device):
    """Move every tensor in the batch dict onto the target device."""
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def build_loader(dataset: Dataset, batch_size: int, shuffle: bool, cfg: Config, device: torch.device):
    """Construct a DataLoader with GPU-friendly defaults when available."""
    use_cuda = device.type == "cuda"
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=cfg.num_workers,
        pin_memory=use_cuda,
        persistent_workers=cfg.num_workers > 0,
        collate_fn=collate_fn,
    )


def run_epoch(model, loader, device, cfg: Config, optimizer=None, scaler=None, scheduler=None):
    """Run one full pass over a dataloader for training or evaluation."""
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_primary = 0.0
    total_aux = 0.0
    total_acc = 0.0
    total_precision = 0.0
    total_recall = 0.0
    total_mse = 0.0
    n_batches = 0

    amp_enabled = cfg.amp and device.type == "cuda"

    for batch in loader:
        batch = move_batch_to_device(batch, device)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        grad_context = torch.enable_grad() if is_train else torch.no_grad()
        with grad_context:
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                outputs = model(
                    action_ids=batch["action_ids"],
                    action_stats=batch["action_stats"],
                    action_mask=batch["action_mask"],
                    profile_feats=batch["profile_feats"],
                    graph_feats=batch["graph_feats"],
                )
                loss, l_primary, l_aux = compute_loss(
                    outputs,
                    batch["bot_label"],
                    batch["risk_score"],
                    batch["exposures"],
                    alpha=cfg.alpha,
                    beta=cfg.beta,
                    c=cfg.max_exposure_weight,
                )

        if is_train:
            if scaler is not None and amp_enabled:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                optimizer.step()
            if scheduler is not None:
                scheduler.step()

        acc, precision, recall, mse = batch_metrics(outputs, batch["bot_label"], batch["risk_score"])
        total_loss += loss.item()
        total_primary += l_primary.item()
        total_aux += l_aux.item()
        total_acc += acc
        total_precision += precision
        total_recall += recall
        total_mse += mse
        n_batches += 1

    return {
        "loss": total_loss / n_batches,
        "primary_loss": total_primary / n_batches,
        "aux_loss": total_aux / n_batches,
        "acc": total_acc / n_batches,
        "precision": total_precision / n_batches,
        "recall": total_recall / n_batches,
        "risk_mse": total_mse / n_batches,
    }


def build_model(cfg: Config, device: torch.device) -> nn.Module:
    """Instantiate the model and optionally compile it on PyTorch 2.x."""
    model = BotDetectionModel(
        num_action_types=cfg.num_action_types,
        profile_dim=cfg.profile_dim,
        graph_dim=cfg.graph_dim,
        action_embed_dim=cfg.action_embed_dim,
        hidden_dim=cfg.hidden_dim,
        num_heads=cfg.num_heads,
        num_layers=cfg.num_layers,
        ff_dim=cfg.ff_dim,
        max_history_len=cfg.max_history_len,
        dropout=cfg.dropout,
    ).to(device)

    if cfg.compile_model and hasattr(torch, "compile"):
        model = torch.compile(model)
    return model


def train_model(cfg: Config | None = None):
    """Create data, train for several epochs, and collect history."""
    cfg = cfg or Config()
    seed_everything(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = BotDetectionDataset(
        num_samples=cfg.train_size,
        num_action_types=cfg.num_action_types,
        min_history_len=cfg.min_history_len,
        max_history_len=cfg.max_history_len,
        profile_dim=cfg.profile_dim,
        graph_dim=cfg.graph_dim,
        seed=cfg.seed,
    )
    test_dataset = BotDetectionDataset(
        num_samples=cfg.test_size,
        num_action_types=cfg.num_action_types,
        min_history_len=cfg.min_history_len,
        max_history_len=cfg.max_history_len,
        profile_dim=cfg.profile_dim,
        graph_dim=cfg.graph_dim,
        seed=cfg.seed + 10_000,
    )

    train_loader = build_loader(train_dataset, cfg.batch_size, shuffle=True, cfg=cfg, device=device)
    test_loader = build_loader(test_dataset, cfg.batch_size, shuffle=False, cfg=cfg, device=device)

    model = build_model(cfg, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    total_steps = cfg.epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(total_steps, 1))
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp and device.type == "cuda")

    history = []
    for epoch in range(1, cfg.epochs + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            device,
            cfg,
            optimizer=optimizer,
            scaler=scaler,
            scheduler=scheduler,
        )
        eval_metrics = run_epoch(model, test_loader, device, cfg)

        row = {
            "epoch": epoch,
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"test_{k}": v for k, v in eval_metrics.items()},
        }
        history.append(row)

        print(
            f"Epoch {epoch}/{cfg.epochs} | "
            f"train_loss={train_metrics['loss']:.4f} | train_acc={train_metrics['acc']:.4f} | "
            f"test_loss={eval_metrics['loss']:.4f} | test_acc={eval_metrics['acc']:.4f} | "
            f"test_precision={eval_metrics['precision']:.4f} | test_recall={eval_metrics['recall']:.4f}"
        )

    return model, history


def main():
    cfg = Config()
    print("Config:", asdict(cfg))
    model, history = train_model(cfg)
    print("Final epoch:", history[-1])
    return model, history


if __name__ == "__main__":
    main()
