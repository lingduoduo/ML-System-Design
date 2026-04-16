import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict
import numpy as np


class PointwiseLTRModel(nn.Module):
    """Point-wise Learning-to-Rank model treating ranking as a regression problem."""

    def __init__(self, feature_dim: int = 32, hidden_dim: int = 128):
        super().__init__()
        self.feature_dim = feature_dim
        self.fc1 = nn.Linear(feature_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, 1)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        hidden = self.dropout(self.relu(self.fc1(features)))
        hidden = self.dropout(self.relu(self.fc2(hidden)))
        score = torch.sigmoid(self.fc3(hidden))
        return score


class PairwiseLTRModel(nn.Module):
    """Pair-wise Learning-to-Rank model using pairwise preference classification."""

    def __init__(self, feature_dim: int = 32, hidden_dim: int = 128):
        super().__init__()
        self.feature_dim = feature_dim
        self.fc1 = nn.Linear(feature_dim * 2, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, 1)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, doc1_features: torch.Tensor, doc2_features: torch.Tensor) -> torch.Tensor:
        concat_features = torch.cat([doc1_features, doc2_features], dim=1)
        hidden = self.dropout(self.relu(self.fc1(concat_features)))
        hidden = self.dropout(self.relu(self.fc2(hidden)))
        preference = torch.sigmoid(self.fc3(hidden))
        return preference


class RankingFeatureExtractor:
    """Extract ranking features from documents and queries."""

    def extract_features(
        self, query: List[str], doc: Dict[str, object], bm25_score: float = 0.0
    ) -> np.ndarray:
        """
        Extract features for a query-document pair.
        Features: [bm25_score, query_len, doc_len, term_overlap, idf_sum]
        """
        query_set = set(query)
        doc_text = " ".join(
            [str(doc.get(field, "")).lower() for field in ["name", "description", "category"]]
        )
        doc_tokens = doc_text.split()
        doc_set = set(doc_tokens)

        term_overlap = len(query_set & doc_set) / (len(query_set) + 1e-8)
        query_len = len(query)
        doc_len = len(doc_tokens)

        features = [
            bm25_score,
            query_len,
            doc_len,
            term_overlap,
            len(query_set & doc_set),
        ]
        return np.array(features, dtype=np.float32)

    def extract_batch_features(
        self, queries: List[List[str]], docs: List[Dict[str, object]], scores: List[float] = None
    ) -> np.ndarray:
        if scores is None:
            scores = [0.0] * len(docs)

        batch_features = []
        for query, doc, score in zip(queries, docs, scores):
            features = self.extract_features(query, doc, score)
            batch_features.append(features)
        return np.array(batch_features, dtype=np.float32)


class PointwiseLTRRanker:
    """Point-wise LTR ranker for document ranking."""

    def __init__(self, feature_dim: int = 5, hidden_dim: int = 128):
        self.model = PointwiseLTRModel(feature_dim, hidden_dim)
        self.feature_extractor = RankingFeatureExtractor()
        self.is_trained = False

    def train(
        self,
        train_data: List[Tuple[List[str], List[Dict[str, object]], List[int]]],
        epochs: int = 10,
        lr: float = 0.001,
    ) -> None:
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        self.model.train()
        for epoch in range(epochs):
            total_loss = 0.0

            for query, docs, relevance_labels in train_data:
                for doc, label in zip(docs, relevance_labels):
                    features = self.feature_extractor.extract_features(query, doc)
                    features_tensor = torch.tensor([features]).to(self.model.device)
                    label_tensor = torch.tensor([[float(label) / 5.0]]).to(self.model.device)

                    optimizer.zero_grad()
                    pred = self.model(features_tensor)
                    loss = criterion(pred, label_tensor)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()

            if (epoch + 1) % 2 == 0:
                avg_loss = total_loss / max(sum(len(docs) for _, docs, _ in train_data), 1)
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

        self.is_trained = True

    def rank(self, query: List[str], docs: List[Dict[str, object]], top_k: int = 5) -> List[Dict[str, object]]:
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call train first.")

        self.model.eval()
        scores = []

        with torch.no_grad():
            for doc in docs:
                features = self.feature_extractor.extract_features(query, doc)
                features_tensor = torch.tensor([features]).to(self.model.device)
                score = self.model(features_tensor).item()
                scores.append(score)

        ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        result = []
        for doc, score in ranked[:top_k]:
            doc_copy = doc.copy()
            doc_copy["pointwise_ltr_score"] = score
            result.append(doc_copy)

        return result


class PairwiseLTRRanker:
    """Pair-wise LTR ranker using pairwise preference classification."""

    def __init__(self, feature_dim: int = 5, hidden_dim: int = 128):
        self.model = PairwiseLTRModel(feature_dim, hidden_dim)
        self.feature_extractor = RankingFeatureExtractor()
        self.is_trained = False

    def train(
        self,
        train_data: List[Tuple[List[str], List[Dict[str, object]], List[int]]],
        epochs: int = 10,
        lr: float = 0.001,
    ) -> None:
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.BCELoss()

        self.model.train()
        for epoch in range(epochs):
            total_loss = 0.0
            pair_count = 0

            for query, docs, relevance_labels in train_data:
                # Precompute features once per doc to avoid O(n²) redundant extractions
                all_doc_features = [
                    self.feature_extractor.extract_features(query, doc) for doc in docs
                ]
                for i in range(len(docs)):
                    for j in range(i + 1, len(docs)):
                        label1, label2 = relevance_labels[i], relevance_labels[j]

                        features1 = all_doc_features[i]
                        features2 = all_doc_features[j]

                        features1_tensor = torch.tensor([features1]).to(self.model.device)
                        features2_tensor = torch.tensor([features2]).to(self.model.device)

                        if label1 > label2:
                            pair_label = torch.tensor([[1.0]]).to(self.model.device)
                        elif label1 < label2:
                            pair_label = torch.tensor([[0.0]]).to(self.model.device)
                        else:
                            continue

                        optimizer.zero_grad()
                        pred = self.model(features1_tensor, features2_tensor)
                        loss = criterion(pred, pair_label)
                        loss.backward()
                        optimizer.step()
                        total_loss += loss.item()
                        pair_count += 1

            if (epoch + 1) % 2 == 0:
                avg_loss = total_loss / max(pair_count, 1)
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

        self.is_trained = True

    def rank(self, query: List[str], docs: List[Dict[str, object]], top_k: int = 5) -> List[Dict[str, object]]:
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call train first.")

        self.model.eval()
        n = len(docs)
        scores = [0.0] * n

        # Precompute features once per doc (O(n) instead of O(n²) extractions)
        all_doc_features = [self.feature_extractor.extract_features(query, doc) for doc in docs]
        features_tensor = torch.tensor(all_doc_features, dtype=torch.float32).to(self.model.device)

        with torch.no_grad():
            for i in range(n):
                for j in range(i + 1, n):
                    # Use antisymmetry: pref(j,i) ≈ 1 - pref(i,j), halving forward passes
                    pref_ij = self.model(features_tensor[i:i+1], features_tensor[j:j+1]).item()
                    scores[i] += pref_ij
                    scores[j] += 1.0 - pref_ij

        ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        result = []
        for doc, score in ranked[:top_k]:
            doc_copy = doc.copy()
            doc_copy["pairwise_ltr_score"] = score
            result.append(doc_copy)

        return result


if __name__ == "__main__":
    pois = [
        {"id": 1, "name": "Tower of London", "category": "Historic Landmark", "description": "ancient fortress and UNESCO World Heritage Site"},
        {"id": 2, "name": "British Museum", "category": "Museum", "description": "world-class collection of art and artefacts"},
        {"id": 3, "name": "Tate Modern", "category": "Gallery", "description": "contemporary art gallery on the South Bank"},
        {"id": 4, "name": "Hyde Park", "category": "Park", "description": "iconic royal park with open-air concerts"},
    ]

    training_data = [
        (["museum", "london"], pois[:3], [3, 4, 1]),
        (["gallery", "art"], pois, [2, 1, 5, 2]),
    ]

    print("=== Point-wise LTR ===")
    pointwise_ranker = PointwiseLTRRanker()
    pointwise_ranker.train(training_data, epochs=5)
    results = pointwise_ranker.rank(["museum", "london"], pois, top_k=2)
    for doc in results:
        print(f"  {doc['name']}: {doc.get('pointwise_ltr_score', 0):.4f}")

    print("\n=== Pair-wise LTR ===")
    pairwise_ranker = PairwiseLTRRanker()
    pairwise_ranker.train(training_data, epochs=5)
    results = pairwise_ranker.rank(["museum", "london"], pois, top_k=2)
    for doc in results:
        print(f"  {doc['name']}: {doc.get('pairwise_ltr_score', 0):.4f}")
