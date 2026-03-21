from __future__ import annotations

import random
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
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
    num_videos: int = 1500
    user_dense_dim: int = 10
    video_dense_dim: int = 12
    num_topics: int = 24
    num_creators: int = 300

    train_size: int = 5000
    test_size: int = 1000
    batch_size: int = 128
    epochs: int = 5
    lr: float = 2e-3
    weight_decay: float = 1e-3
    grad_clip: float = 1.0
    temperature: float = 0.08

    user_id_embed_dim: int = 32
    video_id_embed_dim: int = 32
    hidden_dim: int = 96
    embedding_dim: int = 64
    dropout: float = 0.1

    num_workers: int = 0
    amp: bool = True
    compile_model: bool = False
    retrieval_k: int = 10
    seed: int = 42


class VideoCorpus:
    """Stores a compact catalog used by both training data and retrieval."""

    def __init__(self, cfg: Config, seed: int):
        gen = torch.Generator().manual_seed(seed)
        self.video_dense = torch.randn(cfg.num_videos, cfg.video_dense_dim, generator=gen)
        self.video_topics = torch.randint(0, cfg.num_topics, (cfg.num_videos,), generator=gen)
        self.video_creators = torch.randint(0, cfg.num_creators, (cfg.num_videos,), generator=gen)
        self.video_quality = torch.rand(cfg.num_videos, generator=gen)
        self.video_freshness = torch.rand(cfg.num_videos, generator=gen)

    def get_batch(self, video_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "video_id": video_ids,
            "video_dense": self.video_dense[video_ids],
            "video_topic": self.video_topics[video_ids],
            "creator_id": self.video_creators[video_ids],
            "video_quality": self.video_quality[video_ids],
            "video_freshness": self.video_freshness[video_ids],
        }


class RetrievalDataset(Dataset):
    """Generates one positive user-video interaction per sample on the fly."""

    def __init__(self, num_samples: int, cfg: Config, corpus: VideoCorpus, seed: int):
        self.num_samples = num_samples
        self.cfg = cfg
        self.corpus = corpus
        self.seed = seed

        gen = torch.Generator().manual_seed(seed)
        self.user_ids = torch.randint(0, cfg.num_users, (num_samples,), generator=gen)
        self.user_dense_base = torch.randn(num_samples, cfg.user_dense_dim, generator=gen)
        self.pref_topics = torch.randint(0, cfg.num_topics, (num_samples,), generator=gen)
        self.pref_creators = torch.randint(0, cfg.num_creators, (num_samples,), generator=gen)
        self.watch_budget = torch.rand(num_samples, generator=gen)
        self.novelty_pref = torch.rand(num_samples, generator=gen)

        topic_match = (self.pref_topics.unsqueeze(1) == corpus.video_topics.unsqueeze(0)).float()
        creator_match = (self.pref_creators.unsqueeze(1) == corpus.video_creators.unsqueeze(0)).float()
        base_score = (
            1.8 * topic_match
            + 1.2 * creator_match
            + 0.7 * corpus.video_quality.unsqueeze(0)
            + 0.5 * corpus.video_freshness.unsqueeze(0) * self.novelty_pref.unsqueeze(1)
        )
        score_noise = 0.15 * torch.randn(base_score.shape, generator=gen)
        self.positive_video_ids = (base_score + score_noise).argmax(dim=1)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        pos_video_id = self.positive_video_ids[idx]
        pos_video = self.corpus.get_batch(pos_video_id.unsqueeze(0))
        return {
            "user_id": self.user_ids[idx],
            "user_dense": self.user_dense_base[idx],
            "pref_topic": self.pref_topics[idx],
            "pref_creator": self.pref_creators[idx],
            "watch_budget": self.watch_budget[idx],
            "novelty_pref": self.novelty_pref[idx],
            "positive_video_id": pos_video_id,
            "positive_video_dense": pos_video["video_dense"].squeeze(0),
            "positive_topic": pos_video["video_topic"].squeeze(0),
            "positive_creator": pos_video["creator_id"].squeeze(0),
            "positive_quality": pos_video["video_quality"].squeeze(0),
            "positive_freshness": pos_video["video_freshness"].squeeze(0),
        }


class TwoTowerRetrievalModel(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg

        self.user_embedding = nn.Embedding(cfg.num_users, cfg.user_id_embed_dim)
        self.pref_topic_embedding = nn.Embedding(cfg.num_topics, 12)
        self.pref_creator_embedding = nn.Embedding(cfg.num_creators, 12)

        self.video_embedding = nn.Embedding(cfg.num_videos, cfg.video_id_embed_dim)
        self.video_topic_embedding = nn.Embedding(cfg.num_topics, 12)
        self.creator_embedding = nn.Embedding(cfg.num_creators, 12)

        user_input_dim = cfg.user_id_embed_dim + cfg.user_dense_dim + 12 + 12 + 2
        video_input_dim = cfg.video_id_embed_dim + cfg.video_dense_dim + 12 + 12 + 2

        self.user_tower = nn.Sequential(
            nn.Linear(user_input_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, cfg.embedding_dim),
        )
        self.video_tower = nn.Sequential(
            nn.Linear(video_input_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, cfg.embedding_dim),
        )

    def encode_user(
        self,
        user_id: torch.Tensor,
        user_dense: torch.Tensor,
        pref_topic: torch.Tensor,
        pref_creator: torch.Tensor,
        watch_budget: torch.Tensor,
        novelty_pref: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat(
            [
                self.user_embedding(user_id),
                user_dense,
                self.pref_topic_embedding(pref_topic),
                self.pref_creator_embedding(pref_creator),
                watch_budget.unsqueeze(-1),
                novelty_pref.unsqueeze(-1),
            ],
            dim=-1,
        )
        return F.normalize(self.user_tower(x), dim=-1)

    def encode_video(
        self,
        video_id: torch.Tensor,
        video_dense: torch.Tensor,
        video_topic: torch.Tensor,
        creator_id: torch.Tensor,
        video_quality: torch.Tensor,
        video_freshness: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat(
            [
                self.video_embedding(video_id),
                video_dense,
                self.video_topic_embedding(video_topic),
                self.creator_embedding(creator_id),
                video_quality.unsqueeze(-1),
                video_freshness.unsqueeze(-1),
            ],
            dim=-1,
        )
        return F.normalize(self.video_tower(x), dim=-1)

    def forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        user_vec = self.encode_user(
            user_id=batch["user_id"],
            user_dense=batch["user_dense"],
            pref_topic=batch["pref_topic"],
            pref_creator=batch["pref_creator"],
            watch_budget=batch["watch_budget"],
            novelty_pref=batch["novelty_pref"],
        )
        video_vec = self.encode_video(
            video_id=batch["positive_video_id"],
            video_dense=batch["positive_video_dense"],
            video_topic=batch["positive_topic"],
            creator_id=batch["positive_creator"],
            video_quality=batch["positive_quality"],
            video_freshness=batch["positive_freshness"],
        )
        return user_vec, video_vec


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
    )


def build_model(cfg: Config, device: torch.device) -> nn.Module:
    model = TwoTowerRetrievalModel(cfg).to(device)
    if cfg.compile_model and hasattr(torch, "compile"):
        model = torch.compile(model)
    return model


def precompute_video_embeddings(
    model: TwoTowerRetrievalModel,
    corpus: VideoCorpus,
    cfg: Config,
    device: torch.device,
) -> torch.Tensor:
    model.eval()
    batch_size = max(cfg.batch_size * 4, 256)
    chunks = []
    with torch.no_grad():
        for start in range(0, cfg.num_videos, batch_size):
            video_ids = torch.arange(start, min(start + batch_size, cfg.num_videos))
            video_batch = corpus.get_batch(video_ids)
            video_batch = move_batch_to_device(video_batch, device)
            chunks.append(
                model.encode_video(
                    video_id=video_batch["video_id"],
                    video_dense=video_batch["video_dense"],
                    video_topic=video_batch["video_topic"],
                    creator_id=video_batch["creator_id"],
                    video_quality=video_batch["video_quality"],
                    video_freshness=video_batch["video_freshness"],
                )
            )
    return torch.cat(chunks, dim=0)


def retrieval_metrics(similarity: torch.Tensor, positive_indices: torch.Tensor, k: int) -> tuple[float, float]:
    topk = similarity.topk(k=min(k, similarity.shape[1]), dim=1).indices
    hits = (topk == positive_indices.unsqueeze(1)).any(dim=1).float().mean().item()

    sorted_indices = similarity.argsort(dim=1, descending=True)
    positive_ranks = (sorted_indices == positive_indices.unsqueeze(1)).float().argmax(dim=1) + 1
    mrr = (1.0 / positive_ranks.float()).mean().item()
    return hits, mrr


def run_epoch(
    model: TwoTowerRetrievalModel,
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
    total_acc = 0.0
    n_batches = 0

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        if is_train:
            optimizer.zero_grad(set_to_none=True)

        grad_context = torch.enable_grad() if is_train else torch.no_grad()
        with grad_context:
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                user_vec, video_vec = model(batch)
                logits = user_vec @ video_vec.T / cfg.temperature
                labels = torch.arange(logits.shape[0], device=device)
                loss = F.cross_entropy(logits, labels)

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

        acc = (logits.argmax(dim=1) == labels).float().mean().item()
        total_loss += loss.item()
        total_acc += acc
        n_batches += 1

    return {
        "loss": total_loss / max(n_batches, 1),
        "batch_acc": total_acc / max(n_batches, 1),
    }


@torch.no_grad()
def evaluate_retrieval(
    model: TwoTowerRetrievalModel,
    loader: DataLoader,
    corpus: VideoCorpus,
    cfg: Config,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    video_matrix = precompute_video_embeddings(model, corpus, cfg, device)

    all_hits = []
    all_mrr = []
    for batch in loader:
        batch = move_batch_to_device(batch, device)
        user_vec = model.encode_user(
            user_id=batch["user_id"],
            user_dense=batch["user_dense"],
            pref_topic=batch["pref_topic"],
            pref_creator=batch["pref_creator"],
            watch_budget=batch["watch_budget"],
            novelty_pref=batch["novelty_pref"],
        )
        similarity = user_vec @ video_matrix.T
        hits, mrr = retrieval_metrics(similarity, batch["positive_video_id"], cfg.retrieval_k)
        all_hits.append(hits)
        all_mrr.append(mrr)

    return {
        f"recall@{cfg.retrieval_k}": sum(all_hits) / max(len(all_hits), 1),
        "mrr": sum(all_mrr) / max(len(all_mrr), 1),
    }


@torch.no_grad()
def retrieve_topk(
    model: TwoTowerRetrievalModel,
    batch: dict[str, torch.Tensor],
    corpus: VideoCorpus,
    cfg: Config,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    model.eval()
    batch = move_batch_to_device(batch, device)
    user_vec = model.encode_user(
        user_id=batch["user_id"],
        user_dense=batch["user_dense"],
        pref_topic=batch["pref_topic"],
        pref_creator=batch["pref_creator"],
        watch_budget=batch["watch_budget"],
        novelty_pref=batch["novelty_pref"],
    )
    video_matrix = precompute_video_embeddings(model, corpus, cfg, device)
    scores, indices = (user_vec @ video_matrix.T).topk(k=min(cfg.retrieval_k, cfg.num_videos), dim=1)
    return {
        "topk_video_ids": indices.cpu(),
        "topk_scores": scores.cpu(),
    }


def train_model(cfg: Config | None = None):
    cfg = cfg or Config()
    seed_everything(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    corpus = VideoCorpus(cfg, seed=cfg.seed)
    train_dataset = RetrievalDataset(cfg.train_size, cfg, corpus, seed=cfg.seed + 1)
    test_dataset = RetrievalDataset(cfg.test_size, cfg, corpus, seed=cfg.seed + 10_000)

    train_loader = build_loader(train_dataset, cfg.batch_size, shuffle=True, cfg=cfg, device=device)
    test_loader = build_loader(test_dataset, cfg.batch_size, shuffle=False, cfg=cfg, device=device)

    model = build_model(cfg, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp and device.type == "cuda")

    history = []
    for epoch in range(1, cfg.epochs + 1):
        train_metrics = run_epoch(model, train_loader, device, cfg, optimizer=optimizer, scaler=scaler)
        eval_metrics = evaluate_retrieval(model, test_loader, corpus, cfg, device)
        row = {
            "epoch": epoch,
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **eval_metrics,
        }
        history.append(row)
        print(
            f"Epoch {epoch}/{cfg.epochs} | "
            f"train_loss={train_metrics['loss']:.4f} | "
            f"train_batch_acc={train_metrics['batch_acc']:.4f} | "
            f"recall@{cfg.retrieval_k}={eval_metrics[f'recall@{cfg.retrieval_k}']:.4f} | "
            f"mrr={eval_metrics['mrr']:.4f}"
        )

    return model, corpus, history


def main():
    cfg = Config()
    print("Config:", asdict(cfg))
    model, corpus, history = train_model(cfg)

    sample_dataset = RetrievalDataset(num_samples=4, cfg=cfg, corpus=corpus, seed=cfg.seed + 123)
    sample_batch = next(iter(build_loader(sample_dataset, batch_size=4, shuffle=False, cfg=cfg, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))))
    pred = retrieve_topk(model, sample_batch, corpus, cfg, torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    print("Sample retrieved video ids:", pred["topk_video_ids"][0].tolist())
    print("Final epoch:", history[-1])
    return model, corpus, history


if __name__ == "__main__":
    main()
