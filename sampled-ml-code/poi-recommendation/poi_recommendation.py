import random
from dataclasses import dataclass
from typing import Dict, List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split


# -----------------------------
# Config
# -----------------------------
@dataclass
class ModelConfig:
    # categorical cardinalities
    num_rider_ids: int = 100000
    num_driver_ids: int = 50000
    num_pickup_zones: int = 5000
    num_dropoff_zones: int = 5000
    num_vehicle_types: int = 20
    num_hours: int = 24
    num_weekdays: int = 7

    # embedding dimensions
    id_emb_dim: int = 32
    zone_emb_dim: int = 16
    vehicle_emb_dim: int = 8
    time_emb_dim: int = 8

    # dense input dims
    rider_dense_dim: int = 6
    driver_dense_dim: int = 7
    trip_dense_dim: int = 4
    context_dense_dim: int = 4
    market_dense_dim: int = 4

    # hidden dims
    encoder_hidden_dim: int = 64
    interaction_hidden_dim: int = 128
    tower_hidden_dim: int = 64

    dropout: float = 0.1


# -----------------------------
# Example Dataset
# -----------------------------
class RideMatchDataset(Dataset):
    """
    Each sample corresponds to one (rider_request, candidate_driver) pair.
    """

    CATEGORICAL_FEATURES = (
        "rider_id",
        "driver_id",
        "pickup_zone",
        "dropoff_zone",
        "vehicle_type",
        "hour",
        "weekday",
    )
    DENSE_FEATURES = (
        "rider_dense",
        "driver_dense",
        "trip_dense",
        "context_dense",
        "market_dense",
    )
    LABEL_FEATURES = (
        "accept_label",
        "cancel_label",
        "eta_label",
    )

    def __init__(self, samples: Sequence[Dict]):
        # Materialize tensors once up front instead of rebuilding them for every sample access.
        self.samples = {
            key: self._build_tensor(samples, key)
            for key in (
                *self.CATEGORICAL_FEATURES,
                *self.DENSE_FEATURES,
                *self.LABEL_FEATURES,
            )
        }

    def __len__(self):
        return self.samples["rider_id"].shape[0]

    def __getitem__(self, idx: int):
        return {key: value[idx] for key, value in self.samples.items()}

    @staticmethod
    def _build_tensor(samples: Sequence[Dict], key: str) -> torch.Tensor:
        dtype = torch.long if key in RideMatchDataset.CATEGORICAL_FEATURES else torch.float32
        values = [sample[key] for sample in samples]
        return torch.tensor(values, dtype=dtype)


# -----------------------------
# MLP Block
# -----------------------------
class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int], dropout: float = 0.1):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.extend([
                nn.Linear(prev, h),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ])
            prev = h
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# -----------------------------
# Rider Encoder
# -----------------------------
class RiderEncoder(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.rider_emb = nn.Embedding(cfg.num_rider_ids, cfg.id_emb_dim)
        self.pickup_emb = nn.Embedding(cfg.num_pickup_zones, cfg.zone_emb_dim)
        self.dropoff_emb = nn.Embedding(cfg.num_dropoff_zones, cfg.zone_emb_dim)
        self.hour_emb = nn.Embedding(cfg.num_hours, cfg.time_emb_dim)
        self.weekday_emb = nn.Embedding(cfg.num_weekdays, cfg.time_emb_dim)

        input_dim = (
            cfg.id_emb_dim
            + cfg.zone_emb_dim
            + cfg.zone_emb_dim
            + cfg.time_emb_dim
            + cfg.time_emb_dim
            + cfg.rider_dense_dim
            + cfg.trip_dense_dim
            + cfg.context_dense_dim
        )

        self.mlp = MLP(
            input_dim=input_dim,
            hidden_dims=[cfg.encoder_hidden_dim, cfg.encoder_hidden_dim],
            dropout=cfg.dropout,
        )
        self.output_dim = cfg.encoder_hidden_dim

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        x = torch.cat(
            (
                self.rider_emb(batch["rider_id"]),
                self.pickup_emb(batch["pickup_zone"]),
                self.dropoff_emb(batch["dropoff_zone"]),
                self.hour_emb(batch["hour"]),
                self.weekday_emb(batch["weekday"]),
                batch["rider_dense"],
                batch["trip_dense"],
                batch["context_dense"],
            ),
            dim=-1,
        )
        return self.mlp(x)


# -----------------------------
# Driver Encoder
# -----------------------------
class DriverEncoder(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.driver_emb = nn.Embedding(cfg.num_driver_ids, cfg.id_emb_dim)
        self.vehicle_emb = nn.Embedding(cfg.num_vehicle_types, cfg.vehicle_emb_dim)
        self.pickup_emb = nn.Embedding(cfg.num_pickup_zones, cfg.zone_emb_dim)

        input_dim = (
            cfg.id_emb_dim
            + cfg.vehicle_emb_dim
            + cfg.zone_emb_dim
            + cfg.driver_dense_dim
            + cfg.market_dense_dim
        )

        self.mlp = MLP(
            input_dim=input_dim,
            hidden_dims=[cfg.encoder_hidden_dim, cfg.encoder_hidden_dim],
            dropout=cfg.dropout,
        )
        self.output_dim = cfg.encoder_hidden_dim

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        x = torch.cat(
            (
                self.driver_emb(batch["driver_id"]),
                self.vehicle_emb(batch["vehicle_type"]),
                self.pickup_emb(batch["pickup_zone"]),
                batch["driver_dense"],
                batch["market_dense"],
            ),
            dim=-1,
        )
        return self.mlp(x)


# -----------------------------
# Interaction + Multi-Task Heads
# -----------------------------
class RideMatchingModel(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.rider_encoder = RiderEncoder(cfg)
        self.driver_encoder = DriverEncoder(cfg)

        # interaction features:
        # rider_vec, driver_vec, abs diff, elementwise product, dense context
        encoder_dim = self.rider_encoder.output_dim
        interaction_input_dim = (
            encoder_dim * 4
            + cfg.trip_dense_dim
            + cfg.context_dense_dim
            + cfg.market_dense_dim
        )

        self.interaction_mlp = MLP(
            input_dim=interaction_input_dim,
            hidden_dims=[cfg.interaction_hidden_dim, cfg.tower_hidden_dim],
            dropout=cfg.dropout,
        )

        # multitask heads
        self.accept_head = nn.Linear(cfg.tower_hidden_dim, 1)
        self.cancel_head = nn.Linear(cfg.tower_hidden_dim, 1)
        self.eta_head = nn.Linear(cfg.tower_hidden_dim, 1)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        rider_vec = self.rider_encoder(batch)    # [B, H]
        driver_vec = self.driver_encoder(batch)  # [B, H]

        pair_features = torch.cat(
            (
                rider_vec,
                driver_vec,
                torch.abs(rider_vec - driver_vec),
                rider_vec * driver_vec,
                batch["trip_dense"],
                batch["context_dense"],
                batch["market_dense"],
            ),
            dim=-1,
        )

        hidden = self.interaction_mlp(pair_features)

        accept_logit = self.accept_head(hidden).squeeze(-1)
        cancel_logit = self.cancel_head(hidden).squeeze(-1)
        eta_pred = self.eta_head(hidden).squeeze(-1)

        return {
            "accept_logit": accept_logit,
            "cancel_logit": cancel_logit,
            "eta_pred": eta_pred,
            "rider_vec": rider_vec,
            "driver_vec": driver_vec,
        }

    @staticmethod
    def compute_loss(
        outputs: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
        w_accept: float = 1.0,
        w_cancel: float = 1.0,
        w_eta: float = 0.1,
    ):
        accept_loss = F.binary_cross_entropy_with_logits(
            outputs["accept_logit"],
            batch["accept_label"],
        )

        cancel_loss = F.binary_cross_entropy_with_logits(
            outputs["cancel_logit"],
            batch["cancel_label"],
        )

        eta_loss = F.smooth_l1_loss(
            outputs["eta_pred"],
            batch["eta_label"],
        )

        total_loss = (w_accept * accept_loss) + (w_cancel * cancel_loss) + (w_eta * eta_loss)

        return total_loss, {
            "accept_loss": accept_loss.item(),
            "cancel_loss": cancel_loss.item(),
            "eta_loss": eta_loss.item(),
            "total_loss": total_loss.item(),
        }


def build_fake_samples(n: int = 2000):
    samples = []
    for _ in range(n):
        estimated_eta = random.uniform(2, 20)
        distance_km = random.uniform(0.5, 15.0)
        surge = random.uniform(1.0, 3.0)
        idle_drivers = random.randint(0, 20)

        accept_prob = max(0.05, min(0.95, 0.8 - 0.03 * estimated_eta + 0.01 * idle_drivers))
        cancel_prob = max(0.01, min(0.8, 0.05 + 0.02 * estimated_eta + 0.03 * (surge - 1.0)))

        accept = 1 if random.random() < accept_prob else 0
        cancel = 1 if random.random() < cancel_prob else 0

        sample = {
            "rider_id": random.randint(0, 99999),
            "driver_id": random.randint(0, 49999),
            "pickup_zone": random.randint(0, 4999),
            "dropoff_zone": random.randint(0, 4999),
            "vehicle_type": random.randint(0, 19),
            "hour": random.randint(0, 23),
            "weekday": random.randint(0, 6),

            # example dense features
            "rider_dense": [
                random.uniform(0, 1),   # rider cancellation rate
                random.uniform(0, 1),   # rider completion rate
                random.uniform(0, 10),  # rider avg trips/week
                random.uniform(0, 1),   # rider premium score
                random.uniform(0, 1),   # late-night preference
                random.uniform(0, 1),   # airport trip tendency
            ],
            "driver_dense": [
                random.uniform(0, 1),   # driver acceptance rate
                random.uniform(0, 1),   # driver cancellation rate
                random.uniform(0, 12),  # driver rating proxy
                random.uniform(0, 1),   # reliability
                random.uniform(0, 1),   # fatigue proxy
                random.uniform(0, 1),   # area familiarity
                random.uniform(0, 1),   # recency activity
            ],
            "trip_dense": [
                distance_km,            # estimated distance
                estimated_eta,          # current ETA
                random.uniform(5, 80),  # fare
                random.uniform(0, 1),   # airport flag / trip complexity
            ],
            "context_dense": [
                1.0 if random.random() < 0.3 else 0.0,  # weekend
                1.0 if random.random() < 0.1 else 0.0,  # holiday
                1.0 if random.random() < 0.2 else 0.0,  # rush hour
                random.uniform(0, 1),                   # weather severity
            ],
            "market_dense": [
                float(idle_drivers),            # nearby idle drivers
                random.uniform(0, 30),          # nearby requests
                random.uniform(0.5, 3.0),       # supply-demand ratio
                surge,                          # surge multiplier
            ],

            "accept_label": float(accept),
            "cancel_label": float(cancel),
            "eta_label": float(estimated_eta + random.uniform(-1.0, 1.0)),
        }
        samples.append(sample)
    return samples


def train_one_epoch(
    model: RideMatchingModel,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Dict[str, float]:
    model.train()
    total_examples = 0
    total_loss = 0.0
    total_accept_loss = 0.0
    total_cancel_loss = 0.0
    total_eta_loss = 0.0

    for batch in dataloader:
        batch = {
            key: value.to(device)
            for key, value in batch.items()
        }

        optimizer.zero_grad(set_to_none=True)
        outputs = model(batch)
        loss, metrics = model.compute_loss(outputs, batch)
        loss.backward()
        optimizer.step()

        batch_size = batch["rider_id"].shape[0]
        total_examples += batch_size
        total_loss += metrics["total_loss"] * batch_size
        total_accept_loss += metrics["accept_loss"] * batch_size
        total_cancel_loss += metrics["cancel_loss"] * batch_size
        total_eta_loss += metrics["eta_loss"] * batch_size

    return {
        "loss": total_loss / total_examples,
        "accept_loss": total_accept_loss / total_examples,
        "cancel_loss": total_cancel_loss / total_examples,
        "eta_loss": total_eta_loss / total_examples,
    }


@torch.no_grad()
def evaluate(
    model: RideMatchingModel,
    dataloader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    total_examples = 0
    total_loss = 0.0
    total_accept_loss = 0.0
    total_cancel_loss = 0.0
    total_eta_loss = 0.0
    total_eta_abs_error = 0.0
    total_accept_correct = 0
    total_cancel_correct = 0

    for batch in dataloader:
        batch = {
            key: value.to(device)
            for key, value in batch.items()
        }

        outputs = model(batch)
        _, metrics = model.compute_loss(outputs, batch)

        accept_pred = (torch.sigmoid(outputs["accept_logit"]) >= 0.5).float()
        cancel_pred = (torch.sigmoid(outputs["cancel_logit"]) >= 0.5).float()
        eta_abs_error = torch.abs(outputs["eta_pred"] - batch["eta_label"])

        batch_size = batch["rider_id"].shape[0]
        total_examples += batch_size
        total_loss += metrics["total_loss"] * batch_size
        total_accept_loss += metrics["accept_loss"] * batch_size
        total_cancel_loss += metrics["cancel_loss"] * batch_size
        total_eta_loss += metrics["eta_loss"] * batch_size
        total_eta_abs_error += eta_abs_error.sum().item()
        total_accept_correct += (accept_pred == batch["accept_label"]).sum().item()
        total_cancel_correct += (cancel_pred == batch["cancel_label"]).sum().item()

    return {
        "loss": total_loss / total_examples,
        "accept_loss": total_accept_loss / total_examples,
        "cancel_loss": total_cancel_loss / total_examples,
        "eta_loss": total_eta_loss / total_examples,
        "eta_mae": total_eta_abs_error / total_examples,
        "accept_accuracy": total_accept_correct / total_examples,
        "cancel_accuracy": total_cancel_correct / total_examples,
    }


def create_dataloaders(
    samples: Sequence[Dict],
    batch_size: int = 128,
    train_ratio: float = 0.8,
) -> tuple[DataLoader, DataLoader]:
    dataset = RideMatchDataset(samples)
    train_size = int(len(dataset) * train_ratio)
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def run_training_demo(
    num_samples: int = 2000,
    batch_size: int = 128,
    num_epochs: int = 3,
    learning_rate: float = 1e-3,
) -> List[Dict[str, float]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = ModelConfig()
    model = RideMatchingModel(cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_loader, val_loader = create_dataloaders(
        build_fake_samples(num_samples),
        batch_size=batch_size,
    )

    history = []
    for epoch in range(1, num_epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, device)
        val_metrics = evaluate(model, val_loader, device)

        metrics = {
            "epoch": float(epoch),
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"val_{key}": value for key, value in val_metrics.items()},
        }
        history.append(metrics)

        print(
            f"Epoch {epoch}/{num_epochs} | "
            f"train_loss={train_metrics['loss']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"val_accept_acc={val_metrics['accept_accuracy']:.4f} | "
            f"val_cancel_acc={val_metrics['cancel_accuracy']:.4f} | "
            f"val_eta_mae={val_metrics['eta_mae']:.4f}"
        )

    return history


if __name__ == "__main__":
    run_training_demo()
