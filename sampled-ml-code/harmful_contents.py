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

    vocab_size: int = 5000
    max_text_len: int = 32
    min_text_len: int = 8
    num_image_tokens: int = 8
    image_feat_dim: int = 128
    behavior_dim: int = 16

    train_size: int = 1000
    test_size: int = 200
    batch_size: int = 64
    epochs: int = 5
    lr: float = 1e-3
    weight_decay: float = 1e-2
    grad_clip: float = 1.0
    alpha: float = 1.0
    beta: float = 0.5
    max_view_weight: float = 5.0

    text_embed_dim: int = 128
    hidden_dim: int = 128
    num_heads: int = 4
    num_layers: int = 2
    ff_dim: int = 256
    dropout: float = 0.1

    num_workers: int = 0
    amp: bool = True
    compile_model: bool = False
    seed: int = 42


class MultimodalModerationDataset(Dataset):
    """Synthetic dataset that generates samples on the fly.

    This keeps memory usage small while remaining deterministic for a fixed
    seed and sample index.
    """

    def __init__(
        self,
        num_samples: int,
        vocab_size: int = 5000,
        min_text_len: int = 8,
        max_text_len: int = 32,
        num_image_tokens: int = 8,
        image_feat_dim: int = 128,
        behavior_dim: int = 16,
        seed: int = 42,
    ):
        self.num_samples = num_samples
        self.vocab_size = vocab_size
        self.min_text_len = min_text_len
        self.max_text_len = max_text_len
        self.num_image_tokens = num_image_tokens
        self.image_feat_dim = image_feat_dim
        self.behavior_dim = behavior_dim
        self.seed = seed

        # Pre-generate light-weight labels and lengths once.
        gen = torch.Generator().manual_seed(seed)
        self.text_lens = torch.randint(
            min_text_len,
            max_text_len + 1,
            (num_samples,),
            generator=gen,
        )
        self.harmful = torch.randint(0, 2, (num_samples,), generator=gen)
        self.report_rate = torch.rand(num_samples, generator=gen)
        self.views = torch.randint(1, 10001, (num_samples,), generator=gen)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Re-seeding by index makes each sample deterministic without storing
        # all token and feature tensors in memory.
        gen = torch.Generator().manual_seed(self.seed + idx)
        text_len = int(self.text_lens[idx])
        text_ids = torch.randint(1, self.vocab_size, (text_len,), generator=gen)
        image_feats = torch.randn(self.num_image_tokens, self.image_feat_dim, generator=gen)
        behavior_feats = torch.randn(self.behavior_dim, generator=gen)
        return {
            "text_ids": text_ids,
            "image_feats": image_feats,
            "behavior_feats": behavior_feats,
            "harmful": self.harmful[idx].float(),
            "report_rate": self.report_rate[idx].float(),
            "views": self.views[idx].float(),
        }


def collate_fn(batch):
    """Pad variable-length text and stack the fixed-width modalities."""
    text_ids = [sample["text_ids"] for sample in batch]
    image_feats = torch.stack([sample["image_feats"] for sample in batch])
    behavior_feats = torch.stack([sample["behavior_feats"] for sample in batch])
    harmful = torch.stack([sample["harmful"] for sample in batch])
    report_rate = torch.stack([sample["report_rate"] for sample in batch])
    views = torch.stack([sample["views"] for sample in batch])

    # Padding keeps the batch rectangular for the transformer.
    padded_text = pad_sequence(text_ids, batch_first=True, padding_value=0)
    # The mask tells attention which text positions are real tokens.
    text_mask = (padded_text != 0).long()

    return {
        "text_ids": padded_text,
        "text_mask": text_mask,
        "image_feats": image_feats,
        "behavior_feats": behavior_feats,
        "harmful": harmful,
        "report_rate": report_rate,
        "views": views,
    }


class MultimodalClassifier(nn.Module):
    """Fuse text, image, and behavior signals for two moderation tasks."""

    def __init__(
        self,
        vocab_size: int,
        text_embed_dim: int = 128,
        image_feat_dim: int = 128,
        behavior_dim: int = 16,
        hidden_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        ff_dim: int = 256,
        max_text_len: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Text tokens start as learned embeddings, then get projected into the
        # same hidden space used by the multimodal transformer.
        self.text_embedding = nn.Embedding(vocab_size, text_embed_dim, padding_idx=0)
        self.text_proj = nn.Linear(text_embed_dim, hidden_dim)
        self.text_pos_embedding = nn.Parameter(torch.randn(1, max_text_len, hidden_dim) * 0.02)

        # Image features are assumed to already be extracted patch/token embeddings.
        self.image_proj = nn.Linear(image_feat_dim, hidden_dim)
        self.modality_embedding = nn.Embedding(2, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # The structured behavior branch stays separate until late fusion.
        self.behavior_mlp = nn.Sequential(
            nn.Linear(behavior_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        # Final fusion combines the transformer summary with behavior features.
        self.fusion_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )

        self.primary_head = nn.Linear(hidden_dim, 1)
        self.report_head = nn.Linear(hidden_dim, 1)

    def forward(self, text_ids, text_mask, image_feats, behavior_feats):
        batch_size, text_len = text_ids.shape
        _, image_tokens, _ = image_feats.shape

        # Add both position and modality identity so the transformer can tell
        # text tokens from image tokens after concatenation.
        text_x = self.text_proj(self.text_embedding(text_ids))
        text_x = text_x + self.text_pos_embedding[:, :text_len, :]
        text_mod = self.modality_embedding(
            torch.zeros(batch_size, text_len, dtype=torch.long, device=text_ids.device)
        )
        text_x = text_x + text_mod

        image_x = self.image_proj(image_feats)
        image_mod = self.modality_embedding(
            torch.ones(batch_size, image_tokens, dtype=torch.long, device=text_ids.device)
        )
        image_x = image_x + image_mod

        # Build one joint multimodal sequence for cross-modal attention.
        joint_x = self.input_norm(torch.cat([text_x, image_x], dim=1))
        image_mask = torch.ones(batch_size, image_tokens, dtype=text_mask.dtype, device=text_mask.device)
        joint_mask = torch.cat([text_mask, image_mask], dim=1)
        padding_mask = joint_mask == 0

        encoded = self.transformer(joint_x, src_key_padding_mask=padding_mask)

        # Mean-pool only over valid positions, ignoring padded text tokens.
        joint_mask_f = joint_mask.unsqueeze(-1).float()
        multimodal_repr = (encoded * joint_mask_f).sum(dim=1) / joint_mask_f.sum(dim=1).clamp_min(1.0)

        behavior_repr = self.behavior_mlp(behavior_feats)
        fused = self.fusion_mlp(torch.cat([multimodal_repr, behavior_repr], dim=-1))

        # We predict both a binary harm logit and a continuous report-rate target.
        return {
            "harmful_logit": self.primary_head(fused).squeeze(-1),
            "report_pred": self.report_head(fused).squeeze(-1),
        }


def compute_loss(outputs, harmful, report_rate, views, alpha=1.0, beta=0.5, c=5.0):
    """Combine weighted classification loss with report-rate regression."""
    harmful_logit = outputs["harmful_logit"]
    report_pred = outputs["report_pred"]

    bce_per_sample = F.binary_cross_entropy_with_logits(harmful_logit, harmful, reduction="none")
    # Heavily viewed items get more weight, but the cap avoids extreme values.
    weights = torch.clamp(torch.log1p(views), max=c)
    l_primary = (bce_per_sample * weights).mean()
    l_reports = F.mse_loss(report_pred, report_rate)
    total_loss = alpha * l_primary + beta * l_reports
    return total_loss, l_primary, l_reports


@torch.no_grad()
def batch_metrics(outputs, harmful, report_rate):
    """Compute simple batch-level monitoring metrics."""
    harmful_prob = torch.sigmoid(outputs["harmful_logit"])
    harmful_pred = (harmful_prob >= 0.5).float()
    acc = (harmful_pred == harmful).float().mean().item()
    mse = F.mse_loss(outputs["report_pred"], report_rate).item()
    return acc, mse


def move_batch_to_device(batch, device):
    """Move every tensor in the batch dict onto the target device."""
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def build_loader(dataset: Dataset, batch_size: int, shuffle: bool, cfg: Config, device: torch.device):
    """Construct a DataLoader with GPU-friendly defaults when available."""
    # `pin_memory` helps GPU training by speeding up host-to-device copies.
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
    total_reports = 0.0
    total_acc = 0.0
    total_mse = 0.0
    n_batches = 0

    # AMP only helps on CUDA devices, so we guard it here.
    amp_enabled = cfg.amp and device.type == "cuda"

    for batch in loader:
        batch = move_batch_to_device(batch, device)

        if is_train:
            # `set_to_none=True` is slightly faster and reduces memory writes.
            optimizer.zero_grad(set_to_none=True)

        grad_context = torch.enable_grad() if is_train else torch.no_grad()
        with grad_context:
            # Mixed precision speeds up many GPU workloads with minimal code changes.
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                outputs = model(
                    text_ids=batch["text_ids"],
                    text_mask=batch["text_mask"],
                    image_feats=batch["image_feats"],
                    behavior_feats=batch["behavior_feats"],
                )
                loss, l_primary, l_reports = compute_loss(
                    outputs,
                    batch["harmful"],
                    batch["report_rate"],
                    batch["views"],
                    alpha=cfg.alpha,
                    beta=cfg.beta,
                    c=cfg.max_view_weight,
                )

        if is_train:
            if scaler is not None and amp_enabled:
                # Unscale before clipping so gradient norms are measured correctly.
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
                # Step every batch because the cosine schedule is configured in steps.
                scheduler.step()

        acc, mse = batch_metrics(outputs, batch["harmful"], batch["report_rate"])
        total_loss += loss.item()
        total_primary += l_primary.item()
        total_reports += l_reports.item()
        total_acc += acc
        total_mse += mse
        n_batches += 1

    return {
        "loss": total_loss / n_batches,
        "primary_loss": total_primary / n_batches,
        "report_loss": total_reports / n_batches,
        "acc": total_acc / n_batches,
        "report_mse": total_mse / n_batches,
    }


def build_model(cfg: Config, device: torch.device) -> nn.Module:
    """Instantiate the model and optionally compile it on PyTorch 2.x."""
    model = MultimodalClassifier(
        vocab_size=cfg.vocab_size,
        text_embed_dim=cfg.text_embed_dim,
        image_feat_dim=cfg.image_feat_dim,
        behavior_dim=cfg.behavior_dim,
        hidden_dim=cfg.hidden_dim,
        num_heads=cfg.num_heads,
        num_layers=cfg.num_layers,
        ff_dim=cfg.ff_dim,
        max_text_len=cfg.max_text_len,
        dropout=cfg.dropout,
    ).to(device)

    if cfg.compile_model and hasattr(torch, "compile"):
        # Compilation can improve throughput after an initial warmup cost.
        model = torch.compile(model)
    return model


def train_model(cfg: Config | None = None):
    """Create data, train for several epochs, and collect history."""
    cfg = cfg or Config()
    seed_everything(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = MultimodalModerationDataset(
        num_samples=cfg.train_size,
        vocab_size=cfg.vocab_size,
        min_text_len=cfg.min_text_len,
        max_text_len=cfg.max_text_len,
        num_image_tokens=cfg.num_image_tokens,
        image_feat_dim=cfg.image_feat_dim,
        behavior_dim=cfg.behavior_dim,
        seed=cfg.seed,
    )
    test_dataset = MultimodalModerationDataset(
        num_samples=cfg.test_size,
        vocab_size=cfg.vocab_size,
        min_text_len=cfg.min_text_len,
        max_text_len=cfg.max_text_len,
        num_image_tokens=cfg.num_image_tokens,
        image_feat_dim=cfg.image_feat_dim,
        behavior_dim=cfg.behavior_dim,
        seed=cfg.seed + 10_000,
    )

    # Training shuffles; evaluation does not.
    train_loader = build_loader(train_dataset, cfg.batch_size, shuffle=True, cfg=cfg, device=device)
    test_loader = build_loader(test_dataset, cfg.batch_size, shuffle=False, cfg=cfg, device=device)

    model = build_model(cfg, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    # The scheduler runs once per optimizer step, so T_max is total train steps.
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

        # Flatten train/test metrics into one record per epoch for easy display.
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
            f"test_report_mse={eval_metrics['report_mse']:.4f}"
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
