from __future__ import annotations

import random
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

try:
    torch.set_float32_matmul_precision("high")
except AttributeError:
    pass


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class Config:
    num_users: int = 6000
    num_videos: int = 1800
    num_topics: int = 24
    num_creators: int = 300
    history_min_len: int = 12
    history_max_len: int = 40
    user_dense_dim: int = 10
    video_dense_dim: int = 12

    train_size: int = 3500
    test_size: int = 800
    slate_size: int = 6
    batch_size: int = 64
    epochs: int = 5
    lr: float = 1e-3
    weight_decay: float = 1e-2
    grad_clip: float = 1.0
    alpha: float = 1.0
    beta: float = 0.25

    video_embed_dim: int = 48
    topic_embed_dim: int = 12
    creator_embed_dim: int = 12
    hidden_dim: int = 96
    num_heads: int = 4
    num_layers: int = 2
    ff_dim: int = 192
    dropout: float = 0.1

    num_workers: int = 0
    amp: bool = True
    compile_model: bool = False
    seed: int = 42


class VideoCatalog:
    def __init__(self, cfg: Config, seed: int):
        gen = torch.Generator().manual_seed(seed)
        self.video_dense = torch.randn(cfg.num_videos, cfg.video_dense_dim, generator=gen)
        self.video_topic = torch.randint(0, cfg.num_topics, (cfg.num_videos,), generator=gen)
        self.creator_id = torch.randint(0, cfg.num_creators, (cfg.num_videos,), generator=gen)
        self.quality = torch.rand(cfg.num_videos, generator=gen)
        self.freshness = torch.rand(cfg.num_videos, generator=gen)
        self.length_bucket = torch.randint(0, 5, (cfg.num_videos,), generator=gen)

    def get(self, video_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "video_ids": video_ids,
            "video_dense": self.video_dense[video_ids],
            "video_topic": self.video_topic[video_ids],
            "creator_id": self.creator_id[video_ids],
            "quality": self.quality[video_ids],
            "freshness": self.freshness[video_ids],
            "length_bucket": self.length_bucket[video_ids],
        }


def catalog_from_retrieval_corpus(corpus, cfg: Config) -> VideoCatalog:
    """Reuse the retriever's sampled catalog so both stages rank the same videos."""
    catalog = VideoCatalog.__new__(VideoCatalog)
    catalog.video_dense = corpus.video_dense
    catalog.video_topic = corpus.video_topics
    catalog.creator_id = corpus.video_creators
    catalog.quality = corpus.video_quality
    catalog.freshness = corpus.video_freshness

    quality_bucket = torch.clamp((catalog.quality * 5).long(), max=4)
    freshness_bucket = torch.clamp((catalog.freshness * 5).long(), max=4)
    catalog.length_bucket = ((quality_bucket + freshness_bucket) // 2).long()
    return catalog


def build_reranker_batch_from_retrieval(
    retrieval_batch: dict[str, torch.Tensor],
    retrieved_video_ids: torch.Tensor,
    catalog: VideoCatalog,
    cfg: Config,
    seed: int = 42,
) -> dict[str, torch.Tensor]:
    """Turn top-k retrieval results into a reranker slate with sampled histories."""
    batch_size, slate_size = retrieved_video_ids.shape
    gen = torch.Generator().manual_seed(seed)

    positive_video_id = retrieval_batch["positive_video_id"].long()
    slate_ids = retrieved_video_ids.long().clone()

    missing_positive = ~(slate_ids == positive_video_id.unsqueeze(1)).any(dim=1)
    if missing_positive.any():
        slate_ids[missing_positive, -1] = positive_video_id[missing_positive]

    click_label = (slate_ids == positive_video_id.unsqueeze(1)).float()

    history_lengths = torch.randint(
        cfg.history_min_len,
        cfg.history_max_len + 1,
        (batch_size,),
        generator=gen,
    )
    history_rows = []
    for row_idx in range(batch_size):
        history_len = int(history_lengths[row_idx])
        history_ids = torch.randint(0, cfg.num_videos, (history_len,), generator=gen)
        positive_mask = torch.rand(history_len, generator=gen) < 0.6
        history_ids = torch.where(
            positive_mask,
            torch.full_like(history_ids, positive_video_id[row_idx]),
            history_ids,
        )
        history_rows.append(history_ids)

    padded_history = pad_sequence(history_rows, batch_first=True, padding_value=-1)
    history_mask = (padded_history != -1).long()
    slate = catalog.get(slate_ids)

    watch_target = (
        0.30
        + 0.30 * click_label
        + 0.15 * (slate["video_topic"] == retrieval_batch["pref_topic"].unsqueeze(1)).float()
        + 0.10 * slate["quality"]
        + 0.10 * slate["freshness"] * retrieval_batch["novelty_pref"].unsqueeze(1)
    ).clamp(max=1.0)

    return {
        "user_id": retrieval_batch["user_id"].long(),
        "user_dense": retrieval_batch["user_dense"].float(),
        "pref_topic": retrieval_batch["pref_topic"].long(),
        "pref_creator": retrieval_batch["pref_creator"].long(),
        "session_depth": torch.randint(1, 10, (batch_size,), generator=gen).float(),
        "novelty_pref": retrieval_batch["novelty_pref"].float(),
        "history_ids": padded_history.long(),
        "history_mask": history_mask.long(),
        "slate_ids": slate_ids.long(),
        "slate_dense": slate["video_dense"].float(),
        "slate_topic": slate["video_topic"].long(),
        "slate_creator": slate["creator_id"].long(),
        "slate_quality": slate["quality"].float(),
        "slate_freshness": slate["freshness"].float(),
        "slate_length_bucket": slate["length_bucket"].long(),
        "click_label": click_label.float(),
        "watch_target": watch_target.float(),
    }


class RankingDataset(Dataset):
    """Produces small slates for reranking with one positive per slate."""

    def __init__(self, num_samples: int, cfg: Config, catalog: VideoCatalog, seed: int):
        self.num_samples = num_samples
        self.cfg = cfg
        self.catalog = catalog
        self.seed = seed

        gen = torch.Generator().manual_seed(seed)
        self.user_id = torch.randint(0, cfg.num_users, (num_samples,), generator=gen)
        self.user_dense = torch.randn(num_samples, cfg.user_dense_dim, generator=gen)
        self.pref_topic = torch.randint(0, cfg.num_topics, (num_samples,), generator=gen)
        self.pref_creator = torch.randint(0, cfg.num_creators, (num_samples,), generator=gen)
        self.session_depth = torch.randint(1, 10, (num_samples,), generator=gen).float()
        self.novelty_pref = torch.rand(num_samples, generator=gen)
        self.history_len = torch.randint(
            cfg.history_min_len,
            cfg.history_max_len + 1,
            (num_samples,),
            generator=gen,
        )

        topic_match = (self.pref_topic.unsqueeze(1) == catalog.video_topic.unsqueeze(0)).float()
        creator_match = (self.pref_creator.unsqueeze(1) == catalog.creator_id.unsqueeze(0)).float()
        affinity = (
            1.7 * topic_match
            + 1.0 * creator_match
            + 0.6 * catalog.quality.unsqueeze(0)
            + 0.3 * catalog.freshness.unsqueeze(0) * self.novelty_pref.unsqueeze(1)
        )
        self.best_video = (affinity + 0.1 * torch.randn(affinity.shape, generator=gen)).argmax(dim=1)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        gen = torch.Generator().manual_seed(self.seed + idx)
        history_len = int(self.history_len[idx])

        history_ids = torch.randint(0, self.cfg.num_videos, (history_len,), generator=gen)
        mask = torch.rand(history_len, generator=gen) < 0.55
        history_ids = torch.where(mask, torch.full_like(history_ids, self.best_video[idx]), history_ids)

        positive_video = self.best_video[idx]
        negatives = torch.randint(0, self.cfg.num_videos, (self.cfg.slate_size - 1,), generator=gen)
        slate_ids = torch.cat([positive_video.unsqueeze(0), negatives], dim=0)
        slate_ids = slate_ids[torch.randperm(self.cfg.slate_size, generator=gen)]
        label = (slate_ids == positive_video).float()

        slate = self.catalog.get(slate_ids)

        watch_target = (
            0.35
            + 0.25 * label
            + 0.15 * (slate["video_topic"] == self.pref_topic[idx]).float()
            + 0.10 * slate["quality"]
        ).clamp(max=1.0)

        return {
            "user_id": self.user_id[idx],
            "user_dense": self.user_dense[idx],
            "pref_topic": self.pref_topic[idx],
            "pref_creator": self.pref_creator[idx],
            "session_depth": self.session_depth[idx],
            "novelty_pref": self.novelty_pref[idx],
            "history_ids": history_ids,
            "slate_ids": slate_ids,
            "slate_dense": slate["video_dense"],
            "slate_topic": slate["video_topic"],
            "slate_creator": slate["creator_id"],
            "slate_quality": slate["quality"],
            "slate_freshness": slate["freshness"],
            "slate_length_bucket": slate["length_bucket"],
            "click_label": label,
            "watch_target": watch_target,
        }


def collate_fn(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    history_ids = [sample["history_ids"] for sample in batch]
    padded_history = pad_sequence(history_ids, batch_first=True, padding_value=-1)
    history_mask = (padded_history != -1).long()

    return {
        "user_id": torch.stack([sample["user_id"] for sample in batch]),
        "user_dense": torch.stack([sample["user_dense"] for sample in batch]),
        "pref_topic": torch.stack([sample["pref_topic"] for sample in batch]),
        "pref_creator": torch.stack([sample["pref_creator"] for sample in batch]),
        "session_depth": torch.stack([sample["session_depth"] for sample in batch]),
        "novelty_pref": torch.stack([sample["novelty_pref"] for sample in batch]),
        "history_ids": padded_history,
        "history_mask": history_mask,
        "slate_ids": torch.stack([sample["slate_ids"] for sample in batch]),
        "slate_dense": torch.stack([sample["slate_dense"] for sample in batch]),
        "slate_topic": torch.stack([sample["slate_topic"] for sample in batch]),
        "slate_creator": torch.stack([sample["slate_creator"] for sample in batch]),
        "slate_quality": torch.stack([sample["slate_quality"] for sample in batch]),
        "slate_freshness": torch.stack([sample["slate_freshness"] for sample in batch]),
        "slate_length_bucket": torch.stack([sample["slate_length_bucket"] for sample in batch]),
        "click_label": torch.stack([sample["click_label"] for sample in batch]),
        "watch_target": torch.stack([sample["watch_target"] for sample in batch]),
    }


class DeepRankerModel(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.video_embedding = nn.Embedding(cfg.num_videos + 1, cfg.video_embed_dim, padding_idx=0)
        self.topic_embedding = nn.Embedding(cfg.num_topics, cfg.topic_embed_dim)
        self.creator_embedding = nn.Embedding(cfg.num_creators, cfg.creator_embed_dim)
        self.length_embedding = nn.Embedding(5, 6)
        self.user_embedding = nn.Embedding(cfg.num_users, 24)

        history_input_dim = cfg.video_embed_dim + cfg.topic_embed_dim + cfg.creator_embed_dim
        self.history_proj = nn.Linear(history_input_dim, cfg.hidden_dim)
        self.history_pos_embedding = nn.Parameter(torch.randn(1, cfg.history_max_len, cfg.hidden_dim) * 0.02)
        self.history_norm = nn.LayerNorm(cfg.hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.hidden_dim,
            nhead=cfg.num_heads,
            dim_feedforward=cfg.ff_dim,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.history_encoder = nn.TransformerEncoder(encoder_layer, num_layers=cfg.num_layers)

        context_dim = 24 + cfg.user_dense_dim + cfg.topic_embed_dim + cfg.creator_embed_dim + 2
        self.context_mlp = nn.Sequential(
            nn.Linear(context_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.LayerNorm(cfg.hidden_dim),
        )

        candidate_dim = cfg.video_embed_dim + cfg.video_dense_dim + cfg.topic_embed_dim + cfg.creator_embed_dim + 6 + 2
        self.candidate_mlp = nn.Sequential(
            nn.Linear(candidate_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.LayerNorm(cfg.hidden_dim),
        )

        self.fusion_mlp = nn.Sequential(
            nn.Linear(cfg.hidden_dim * 3, cfg.hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.LayerNorm(cfg.hidden_dim),
        )
        self.click_head = nn.Linear(cfg.hidden_dim, 1)
        self.watch_head = nn.Linear(cfg.hidden_dim, 1)

    def encode_history(self, history_ids: torch.Tensor, history_mask: torch.Tensor) -> torch.Tensor:
        safe_history_ids = history_ids.clamp_min(0)
        history_video = self.video_embedding(safe_history_ids + 1)
        topic_ids = safe_history_ids.remainder(self.cfg.num_topics)
        creator_ids = safe_history_ids.remainder(self.cfg.num_creators)
        history_x = torch.cat(
            [
                history_video,
                self.topic_embedding(topic_ids),
                self.creator_embedding(creator_ids),
            ],
            dim=-1,
        )
        history_x = self.history_proj(history_x)
        history_x = history_x + self.history_pos_embedding[:, :history_x.shape[1], :]
        history_x = self.history_norm(history_x)

        encoded = self.history_encoder(history_x, src_key_padding_mask=history_mask == 0)
        history_mask_f = history_mask.unsqueeze(-1).float()
        return (encoded * history_mask_f).sum(dim=1) / history_mask_f.sum(dim=1).clamp_min(1.0)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        batch_size, slate_size = batch["slate_ids"].shape

        history_repr = self.encode_history(batch["history_ids"], batch["history_mask"])
        context_repr = self.context_mlp(
            torch.cat(
                [
                    self.user_embedding(batch["user_id"]),
                    batch["user_dense"],
                    self.topic_embedding(batch["pref_topic"]),
                    self.creator_embedding(batch["pref_creator"]),
                    batch["session_depth"].unsqueeze(-1) / 10.0,
                    batch["novelty_pref"].unsqueeze(-1),
                ],
                dim=-1,
            )
        )

        candidate_repr = self.candidate_mlp(
            torch.cat(
                [
                    # Shift by one so embedding index 0 remains reserved for padding.
                    self.video_embedding(batch["slate_ids"] + 1),
                    batch["slate_dense"],
                    self.topic_embedding(batch["slate_topic"]),
                    self.creator_embedding(batch["slate_creator"]),
                    self.length_embedding(batch["slate_length_bucket"]),
                    batch["slate_quality"].unsqueeze(-1),
                    batch["slate_freshness"].unsqueeze(-1),
                ],
                dim=-1,
            )
        )

        context_tiled = context_repr.unsqueeze(1).expand(batch_size, slate_size, -1)
        history_tiled = history_repr.unsqueeze(1).expand(batch_size, slate_size, -1)
        fused = self.fusion_mlp(torch.cat([candidate_repr, context_tiled, history_tiled], dim=-1))

        return {
            "click_logit": self.click_head(fused).squeeze(-1),
            "watch_pred": self.watch_head(fused).squeeze(-1),
        }


def compute_loss(
    outputs: dict[str, torch.Tensor],
    click_label: torch.Tensor,
    watch_target: torch.Tensor,
    alpha: float = 1.0,
    beta: float = 0.25,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    click_loss = F.binary_cross_entropy_with_logits(outputs["click_logit"], click_label)
    watch_loss = F.mse_loss(torch.sigmoid(outputs["watch_pred"]), watch_target)
    total = alpha * click_loss + beta * watch_loss
    return total, click_loss, watch_loss


@torch.no_grad()
def ranking_metrics(outputs: dict[str, torch.Tensor], click_label: torch.Tensor) -> tuple[float, float]:
    probs = torch.sigmoid(outputs["click_logit"])
    top_choice = probs.argmax(dim=1)
    hit_rate = click_label.gather(1, top_choice.unsqueeze(1)).mean().item()

    ranks = probs.argsort(dim=1, descending=True)
    positive_positions = (click_label.gather(1, ranks) > 0).float().argmax(dim=1) + 1
    mrr = (1.0 / positive_positions.float()).mean().item()
    return hit_rate, mrr


def move_batch_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def build_loader(dataset: Dataset, batch_size: int, shuffle: bool, cfg: Config, device: torch.device) -> DataLoader:
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


def build_model(cfg: Config, device: torch.device) -> nn.Module:
    model = DeepRankerModel(cfg).to(device)
    if cfg.compile_model and hasattr(torch, "compile"):
        model = torch.compile(model)
    return model


def run_epoch(
    model: DeepRankerModel,
    loader: DataLoader,
    device: torch.device,
    cfg: Config,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.cuda.amp.GradScaler | None = None,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    amp_enabled = cfg.amp and device.type == "cuda"

    total_loss = 0.0
    total_click_loss = 0.0
    total_watch_loss = 0.0
    total_hit_rate = 0.0
    total_mrr = 0.0
    n_batches = 0

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        if is_train:
            optimizer.zero_grad(set_to_none=True)

        grad_context = torch.enable_grad() if is_train else torch.no_grad()
        with grad_context:
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                outputs = model(batch)
                loss, click_loss, watch_loss = compute_loss(
                    outputs,
                    batch["click_label"],
                    batch["watch_target"],
                    alpha=cfg.alpha,
                    beta=cfg.beta,
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

        hit_rate, mrr = ranking_metrics(outputs, batch["click_label"])
        total_loss += loss.item()
        total_click_loss += click_loss.item()
        total_watch_loss += watch_loss.item()
        total_hit_rate += hit_rate
        total_mrr += mrr
        n_batches += 1

    return {
        "loss": total_loss / max(n_batches, 1),
        "click_loss": total_click_loss / max(n_batches, 1),
        "watch_loss": total_watch_loss / max(n_batches, 1),
        "hit_rate": total_hit_rate / max(n_batches, 1),
        "mrr": total_mrr / max(n_batches, 1),
    }


@torch.no_grad()
def rerank_slate(
    model: DeepRankerModel,
    batch: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    model.eval()
    batch = move_batch_to_device(batch, device)
    outputs = model(batch)
    scores = torch.sigmoid(outputs["click_logit"])
    ranked = scores.argsort(dim=1, descending=True)
    return {
        "ranked_indices": ranked.cpu(),
        "ranked_video_ids": batch["slate_ids"].gather(1, ranked).cpu(),
        "scores": scores.cpu(),
    }


def train_model(cfg: Config | None = None):
    cfg = cfg or Config()
    seed_everything(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    catalog = VideoCatalog(cfg, seed=cfg.seed)
    train_dataset = RankingDataset(cfg.train_size, cfg, catalog, seed=cfg.seed + 1)
    test_dataset = RankingDataset(cfg.test_size, cfg, catalog, seed=cfg.seed + 10_000)

    train_loader = build_loader(train_dataset, cfg.batch_size, shuffle=True, cfg=cfg, device=device)
    test_loader = build_loader(test_dataset, cfg.batch_size, shuffle=False, cfg=cfg, device=device)

    model = build_model(cfg, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp and device.type == "cuda")

    history = []
    for epoch in range(1, cfg.epochs + 1):
        train_metrics = run_epoch(model, train_loader, device, cfg, optimizer=optimizer, scaler=scaler)
        eval_metrics = run_epoch(model, test_loader, device, cfg)
        row = {
            "epoch": epoch,
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"test_{k}": v for k, v in eval_metrics.items()},
        }
        history.append(row)
        print(
            f"Epoch {epoch}/{cfg.epochs} | "
            f"train_loss={train_metrics['loss']:.4f} | train_hit_rate={train_metrics['hit_rate']:.4f} | "
            f"test_loss={eval_metrics['loss']:.4f} | test_hit_rate={eval_metrics['hit_rate']:.4f} | "
            f"test_mrr={eval_metrics['mrr']:.4f}"
        )

    return model, catalog, history


def main():
    cfg = Config()
    print("Config:", asdict(cfg))
    model, catalog, history = train_model(cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sample_dataset = RankingDataset(num_samples=3, cfg=cfg, catalog=catalog, seed=cfg.seed + 123)
    sample_batch = next(iter(build_loader(sample_dataset, batch_size=3, shuffle=False, cfg=cfg, device=device)))
    pred = rerank_slate(model, sample_batch, device)
    print("Sample reranked video ids:", pred["ranked_video_ids"][0].tolist())
    print("Final epoch:", history[-1])
    return model, catalog, history


if __name__ == "__main__":
    main()
