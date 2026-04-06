from __future__ import annotations

from typing import Dict, List, Tuple
import numpy as np

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
    TORCH_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - environment-specific import failure
    torch = None
    nn = None
    TORCH_AVAILABLE = False
    TORCH_IMPORT_ERROR = exc


BaseModule = nn.Module if TORCH_AVAILABLE else object


class CrossEncoderModel(BaseModule):
    """Cross-encoder model that jointly encodes query and document."""

    def __init__(self, input_dim: int = 5, hidden_dim: int = 256):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for CrossEncoderModel") from TORCH_IMPORT_ERROR
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, 1)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: concatenated query and document features
        Returns:
            relevance score [0, 1]
        """
        hidden = self.dropout(self.relu(self.fc1(features)))
        hidden = self.dropout(self.relu(self.fc2(hidden)))
        score = torch.sigmoid(self.fc3(hidden))
        return score


class DenseSemanticModel(BaseModule):
    """Deep semantic model with separate query and document encoders."""

    def __init__(self, input_dim: int = 32, embedding_dim: int = 128, hidden_dim: int = 256):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for DenseSemanticModel") from TORCH_IMPORT_ERROR
        super().__init__()
        self.embedding_dim = embedding_dim

        # Query encoder
        self.query_fc1 = nn.Linear(input_dim, hidden_dim)
        self.query_relu = nn.ReLU()
        self.query_dropout = nn.Dropout(0.3)
        self.query_fc2 = nn.Linear(hidden_dim, embedding_dim)

        # Document encoder
        self.doc_fc1 = nn.Linear(input_dim, hidden_dim)
        self.doc_relu = nn.ReLU()
        self.doc_dropout = nn.Dropout(0.3)
        self.doc_fc2 = nn.Linear(hidden_dim, embedding_dim)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def encode_query(self, query_features: torch.Tensor) -> torch.Tensor:
        hidden = self.query_dropout(self.query_relu(self.query_fc1(query_features)))
        embedding = self.query_fc2(hidden)
        embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
        return embedding

    def encode_document(self, doc_features: torch.Tensor) -> torch.Tensor:
        hidden = self.doc_dropout(self.doc_relu(self.doc_fc1(doc_features)))
        embedding = self.doc_fc2(hidden)
        embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
        return embedding

    def forward(self, query_features: torch.Tensor, doc_features: torch.Tensor) -> torch.Tensor:
        query_emb = self.encode_query(query_features)
        doc_emb = self.encode_document(doc_features)
        similarity = torch.sum(query_emb * doc_emb, dim=1, keepdim=True)
        return similarity


class RerankerFeatureExtractor:
    """Extract features for reranking."""

    def extract_features(self, query: List[str], doc: Dict[str, object]) -> np.ndarray:
        """Extract semantic and relevance features."""
        query_set = set(query)
        doc_text = " ".join(
            [str(doc.get(field, "")).lower() for field in ["name", "description", "category"]]
        )
        doc_tokens = doc_text.split()
        doc_set = set(doc_tokens)

        term_overlap = len(query_set & doc_set) / (len(query_set) + 1e-8)
        query_len = len(query) / 10.0
        doc_len = len(doc_tokens) / 50.0
        shared_terms = len(query_set & doc_set) / 10.0

        features = [
            term_overlap,
            query_len,
            doc_len,
            shared_terms,
            len(query),
        ]
        return np.array(features, dtype=np.float32)


class CrossEncoderReranker:
    """Cross-encoder based reranker."""

    def __init__(self, input_dim: int = 5, hidden_dim: int = 256):
        self.model = CrossEncoderModel(input_dim, hidden_dim)
        self.feature_extractor = RerankerFeatureExtractor()
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
            count = 0

            for query, docs, relevance_labels in train_data:
                for doc, label in zip(docs, relevance_labels):
                    features = self.feature_extractor.extract_features(query, doc)
                    normalized_label = min(label / 5.0, 1.0)

                    features_tensor = torch.tensor(features, dtype=torch.float32, device=self.model.device).unsqueeze(0)
                    label_tensor = torch.tensor([[normalized_label]], dtype=torch.float32, device=self.model.device)

                    optimizer.zero_grad()
                    score = self.model(features_tensor)
                    loss = criterion(score, label_tensor)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                    count += 1

            if (epoch + 1) % 2 == 0:
                avg_loss = total_loss / max(count, 1)
                print(f"Cross-Encoder Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

        self.is_trained = True

    def rerank(self, query: List[str], docs: List[Dict[str, object]], top_k: int = 5) -> List[Dict[str, object]]:
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call train first.")

        self.model.eval()
        scores = []

        with torch.no_grad():
            for doc in docs:
                features = self.feature_extractor.extract_features(query, doc)
                features_tensor = torch.tensor(features, dtype=torch.float32, device=self.model.device).unsqueeze(0)
                score = self.model(features_tensor).item()
                scores.append(score)

        ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        result = []
        for doc, score in ranked[:top_k]:
            doc_copy = doc.copy()
            doc_copy["cross_encoder_score"] = score
            result.append(doc_copy)

        return result


class DenseSemanticReranker:
    """Dense semantic model based reranker."""

    def __init__(self, input_dim: int = 5, embedding_dim: int = 128, hidden_dim: int = 256):
        self.model = DenseSemanticModel(input_dim, embedding_dim, hidden_dim)
        self.feature_extractor = RerankerFeatureExtractor()
        self.is_trained = False

    def train(
        self,
        train_data: List[Tuple[List[str], List[Dict[str, object]], List[int]]],
        epochs: int = 10,
        lr: float = 0.001,
        margin: float = 0.3,
    ) -> None:
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        self.model.train()
        for epoch in range(epochs):
            total_loss = 0.0
            pair_count = 0

            for query, docs, relevance_labels in train_data:
                for i in range(len(docs)):
                    for j in range(i + 1, len(docs)):
                        if relevance_labels[i] == relevance_labels[j]:
                            continue

                        query_features = self.feature_extractor.extract_features(query, docs[i])
                        doc1_features = self.feature_extractor.extract_features(query, docs[i])
                        doc2_features = self.feature_extractor.extract_features(query, docs[j])

                        query_tensor = torch.tensor(query_features, dtype=torch.float32, device=self.model.device).unsqueeze(0)
                        doc1_tensor = torch.tensor(doc1_features, dtype=torch.float32, device=self.model.device).unsqueeze(0)
                        doc2_tensor = torch.tensor(doc2_features, dtype=torch.float32, device=self.model.device).unsqueeze(0)

                        optimizer.zero_grad()

                        score1 = self.model(query_tensor, doc1_tensor)
                        score2 = self.model(query_tensor, doc2_tensor)

                        if relevance_labels[i] > relevance_labels[j]:
                            loss = torch.clamp(margin - (score1 - score2), min=0.0).mean()
                        else:
                            loss = torch.clamp(margin - (score2 - score1), min=0.0).mean()

                        loss.backward()
                        optimizer.step()
                        total_loss += loss.item()
                        pair_count += 1

            if (epoch + 1) % 2 == 0:
                avg_loss = total_loss / max(pair_count, 1)
                print(f"Dense Semantic Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

        self.is_trained = True

    def rerank(self, query: List[str], docs: List[Dict[str, object]], top_k: int = 5) -> List[Dict[str, object]]:
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call train first.")

        self.model.eval()
        scores = []

        with torch.no_grad():
            query_features = self.feature_extractor.extract_features(query, {"name": "", "description": "", "category": ""})
            query_tensor = torch.tensor(query_features, dtype=torch.float32, device=self.model.device).unsqueeze(0)

            for doc in docs:
                doc_features = self.feature_extractor.extract_features(query, doc)
                doc_tensor = torch.tensor(doc_features, dtype=torch.float32, device=self.model.device).unsqueeze(0)
                score = self.model(query_tensor, doc_tensor).item()
                scores.append(score)

        ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        result = []
        for doc, score in ranked[:top_k]:
            doc_copy = doc.copy()
            doc_copy["dense_semantic_score"] = score
            result.append(doc_copy)

        return result


class HybridReranker:
    """Hybrid reranker combining multiple models."""

    def __init__(self):
        self.cross_encoder = CrossEncoderReranker()
        self.dense_semantic = DenseSemanticReranker()

    def train(
        self,
        train_data: List[Tuple[List[str], List[Dict[str, object]], List[int]]],
        epochs: int = 10,
    ) -> None:
        print("Training Cross-Encoder...")
        self.cross_encoder.train(train_data, epochs=epochs)
        print("Training Dense Semantic Model...")
        self.dense_semantic.train(train_data, epochs=epochs)

    def rerank(self, query: List[str], docs: List[Dict[str, object]], top_k: int = 5, weights: Dict[str, float] = None) -> List[Dict[str, object]]:
        if weights is None:
            weights = {"cross_encoder": 0.5, "dense_semantic": 0.5}

        cross_results = self.cross_encoder.rerank(query, docs, top_k=len(docs))
        dense_results = self.dense_semantic.rerank(query, docs, top_k=len(docs))

        score_map = {}
        for doc in cross_results:
            doc_id = doc.get("id")
            score_map[doc_id] = score_map.get(doc_id, 0.0) + weights["cross_encoder"] * doc.get("cross_encoder_score", 0.0)

        for doc in dense_results:
            doc_id = doc.get("id")
            score_map[doc_id] = score_map.get(doc_id, 0.0) + weights["dense_semantic"] * doc.get("dense_semantic_score", 0.0)

        hybrid_scored = []
        for doc in docs:
            doc_copy = doc.copy()
            doc_copy["hybrid_score"] = score_map.get(doc.get("id"), 0.0)
            hybrid_scored.append(doc_copy)

        ranked = sorted(hybrid_scored, key=lambda x: x["hybrid_score"], reverse=True)
        return ranked[:top_k]


if __name__ == "__main__":
    if not TORCH_AVAILABLE:
        raise SystemExit(f"PyTorch is unavailable in this environment: {TORCH_IMPORT_ERROR}")

    pois = [
        {"id": 1, "name": "Shunjing Hot Spring", "category": "Hot Spring", "description": "well-known resort"},
        {"id": 2, "name": "Jiuhua Mountain Resort", "category": "Hot Spring", "description": "scenic spa resort"},
        {"id": 3, "name": "City Spa Center", "category": "Spa", "description": "relaxing services"},
        {"id": 4, "name": "Capital Hot Spring", "category": "Hot Spring", "description": "popular hotel"},
    ]

    training_data = [
        (["hot", "spring"], pois[:3], [3, 4, 1]),
        (["spa", "city"], pois, [2, 1, 5, 2]),
    ]

    print("=== Cross-Encoder Reranker ===")
    cross_reranker = CrossEncoderReranker()
    cross_reranker.train(training_data, epochs=5)
    results = cross_reranker.rerank(["hot", "spring"], pois, top_k=2)
    for doc in results:
        print(f"  {doc['name']}: {doc.get('cross_encoder_score', 0):.4f}")

    print("\n=== Dense Semantic Reranker ===")
    dense_reranker = DenseSemanticReranker()
    dense_reranker.train(training_data, epochs=5)
    results = dense_reranker.rerank(["hot", "spring"], pois, top_k=2)
    for doc in results:
        print(f"  {doc['name']}: {doc.get('dense_semantic_score', 0):.4f}")

    print("\n=== Hybrid Reranker ===")
    hybrid_reranker = HybridReranker()
    hybrid_reranker.train(training_data, epochs=5)
    results = hybrid_reranker.rerank(["hot", "spring"], pois, top_k=2)
    for doc in results:
        print(f"  {doc['name']}: {doc.get('hybrid_score', 0):.4f}")
