import torch
import torch.nn as nn
from typing import List, Dict, Tuple

from vocabulary import Vocabulary


class IntentionClassifier(nn.Module):
    """PyTorch-based intention classifier using a simple feedforward neural network."""

    def __init__(self, vocab_size: int = 5000, embedding_dim: int = 128, hidden_dim: int = 256, num_classes: int = 4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, num_classes)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(token_ids)
        pooled = embedded.mean(dim=1)
        hidden = self.dropout(self.relu(self.fc1(pooled)))
        hidden = self.dropout(self.relu(self.fc2(hidden)))
        output = self.fc3(hidden)
        return output

    def predict(self, token_ids: torch.Tensor) -> Tuple[List[str], List[float]]:
        self.eval()
        with torch.no_grad():
            logits = self.forward(token_ids)
            probs = torch.softmax(logits, dim=1)
            predictions = torch.argmax(logits, dim=1)
        
        intent_labels = ["purchase", "navigate", "qa", "recommendation"]
        predicted_intents = [intent_labels[pred.item()] for pred in predictions]
        predicted_probs = [prob.item() for prob in probs.max(dim=1).values]
        return predicted_intents, predicted_probs


class IntentionClassificationPipeline:
    """Pipeline for intention classification with vocabulary management."""

    def __init__(self, vocab_size: int = 5000, embedding_dim: int = 128, hidden_dim: int = 256):
        self.vocab = Vocabulary()
        self.model = IntentionClassifier(vocab_size, embedding_dim, hidden_dim, num_classes=4)
        self.is_trained = False

    def train_model(self, train_data: List[Tuple[List[str], str]], epochs: int = 10, lr: float = 0.001) -> None:
        self.vocab.build([tokens for tokens, _ in train_data], min_freq=2)
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        intent_labels = ["purchase", "navigate", "qa", "recommendation"]
        label_to_id = {label: idx for idx, label in enumerate(intent_labels)}

        self.model.train()
        for epoch in range(epochs):
            total_loss = 0.0
            for tokens, label in train_data:
                token_ids = self.vocab.encode(tokens)
                token_ids = torch.tensor([token_ids], dtype=torch.long).to(self.model.device)
                label_id = torch.tensor([label_to_id[label]], dtype=torch.long).to(self.model.device)
                
                optimizer.zero_grad()
                logits = self.model(token_ids)
                loss = criterion(logits, label_id)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            if (epoch + 1) % 2 == 0:
                avg_loss = total_loss / len(train_data)
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")
        
        self.is_trained = True

    def predict(self, tokens: List[str]) -> Tuple[str, float]:
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call train_model first.")
        
        token_ids = self.vocab.encode(tokens)
        token_ids = torch.tensor([token_ids], dtype=torch.long).to(self.model.device)
        predicted_intents, predicted_probs = self.model.predict(token_ids)
        
        return predicted_intents[0], predicted_probs[0]


if __name__ == "__main__":
    training_data = [
        (["buy", "phone", "online"], "purchase"),
        (["order", "pizza", "delivery"], "purchase"),
        (["directions", "to", "restaurant"], "navigate"),
        (["map", "nearest", "hotel"], "navigate"),
        (["what", "is", "weather"], "qa"),
        (["how", "to", "cook", "pasta"], "qa"),
        (["recommend", "movie"], "recommendation"),
        (["suggest", "best", "restaurant"], "recommendation"),
    ]

    pipeline = IntentionClassificationPipeline()
    pipeline.train_model(training_data, epochs=10)

    test_tokens = ["buy", "laptop", "now"]
    intent, confidence = pipeline.predict(test_tokens)
    print(f"Predicted intent: {intent} (confidence: {confidence:.4f})")
