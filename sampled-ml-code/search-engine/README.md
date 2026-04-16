# ML-Based Search Engine System

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A modular, production-ready search engine reference implementation combining classic IR techniques (BM25) with lightweight semantic search, neural ranking, and comprehensive evaluation metrics. Features document preprocessing, semantic retrieval, query understanding, multi-stage retrieval, and reranking.

## Table of Contents

- [Features](#features)
- [System Architecture](#system-architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Code Overview](#code-overview)
- [Module Details](#module-details)
- [API Reference](#api-reference)
- [Configuration](#configuration)
- [Usage Examples](#usage-examples)
- [Performance Characteristics](#performance-characteristics)
- [Future Enhancements](#future-enhancements)
- [Contributing](#contributing)
- [License](#license)

---

## Features

- **Modular Design**: Pluggable components for preprocessing, retrieval, ranking, and evaluation
- **Semantic Search**: In-memory vector stores with dataset-aware filtering
- **Query Understanding**: Intent detection, NER, tokenization, and synonym expansion
- **Multi-Stage Retrieval**: BM25 + dense embeddings with fallback strategies
- **Neural Ranking**: Pointwise and pairwise LTR with cross-encoder reranking
- **Evaluation Metrics**: Context relevance, answer faithfulness, and performance comparison
- **Config-Driven**: YAML-based configuration for preprocessing and segmentation
- **Production Ready**: Graceful degradation, optional dependencies, and lightweight fallbacks
- **Optimized Inference**: Pairwise LTR ranking uses O(n) feature precomputation + antisymmetry to halve forward passes; spaCy models are shared across components via a module-level cache

---

## System Architecture

```
Document Ingestion
    ↓
Preprocessing (cleanup, segmentation, tokenization)
    ↓
Indexing (BM25 + Semantic Vectors)
    ↓
Query Processing (understanding, intent, NER)
    ↓
Retrieval (multi-stage: main → backup → fallback)
    ↓
Ranking (LTR: pointwise/pairwise)
    ↓
Reranking (cross-encoder, hybrid)
    ↓
Evaluation (relevance, faithfulness)
    ↓
Final Results
```

---

## Installation

### Prerequisites

- Python 3.8+
- pip

### Install Dependencies

```bash
pip install numpy nltk pyyaml
# Optional for enhanced NLP
pip install spacy torch
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('punkt'); nltk.download('wordnet'); nltk.download('stopwords')"
```

### Clone and Setup

```bash
git clone <repository-url>
cd search-engine
# Install additional dependencies if needed
pip install -r requirements.txt  # if exists
```

---

## Quick Start

### Basic Semantic Retrieval

```python
from semantic_retriever import SemanticRetriever

# Prepare documents
documents = [
    {
        "dataset_id": "london-dataset",
        "content": "The British Museum houses a vast collection of art and artefacts from world cultures.",
    },
    {
        "dataset_id": "london-dataset",
        "content": "Tate Modern is a contemporary art gallery on the South Bank with free permanent collections.",
    },
]

# Create and index retriever
retriever = SemanticRetriever(dataset_ids=["london-dataset"])
retriever.index_documents(documents)

# Retrieve relevant documents
results = retriever.retrieve("best museum art gallery london", top_k=2)
for doc in results:
    print(f"ID: {doc.metadata['document_id']}, Score: {doc.metadata['score']:.3f}")
```

### Query Processing

```python
from search_engine import SearchQueryProcessor

processor = SearchQueryProcessor()
result = processor.process_query("famous London museum 😊")

print("Tokens:", result["tokens"])
print("Intent:", result["category_intent"])
print("Entities:", result["entities"])
```

### Text Preprocessing

```python
from text_preprocessing import preprocessing

cleaned = preprocessing("Hello 2nd world! This is a test.", debug=True)
print(cleaned)  # "hello second world test"
```

---

## Code Overview

### Core Modules

1. **`text_preprocessing.py`** - Optimized text preprocessing pipeline
   - Decoding, lowercasing, digit-to-words, punctuation removal
   - Spelling correction, stopword removal, lemmatization
   - Company extraction with spaCy

2. **`vocabulary.py`** - Reusable vocabulary management
   - Token-to-index mapping with special tokens
   - Serialization, encoding/decoding, frequency-based building

3. **`pretrained_embeddings.py`** - Annoy-based embedding wrapper
   - Efficient nearest neighbor search for word vectors
   - Analogy solving, similarity queries

4. **`document_processor.py`** - Config-driven document preprocessing
   - YAML-configured rules for cleanup and segmentation
   - Supports plain text and JSON payloads

5. **`search_engine.py`** - Query understanding and processing
   - Tokenization, POS tagging, NER, intent detection
   - Synonym expansion, error correction
   - `simple_error_correction`: first-character pre-filter reduces edit-distance candidate set ~10× before the O(m·n) scan

6. **`semantic_retriever.py`** - Semantic retrieval with vector stores
   - In-memory semantic search with dataset filtering
   - Pluggable vector store interface

7. **`semantic_slicing.py`** - Advanced text segmentation and evaluation
   - Semantic slicing with spaCy
   - Relevance, faithfulness, and difficulty analysis
   - Module-level `_SPACY_NLP_CACHE` ensures each spaCy model is loaded at most once regardless of how many classes instantiate it

8. **`semantic_pipeline.py`** - High-level semantic retrieval pipeline
   - Async indexing and retrieval
   - Performance evaluation

9. **`search_recall.py`** - Multi-stage recall engine
   - Main/backup pool retrieval with modifiers
   - Synonym and term expansion

10. **`learning_to_rank.py`** - Neural ranking models
    - Pointwise and pairwise LTR
    - Feature engineering for ranking
    - `PairwiseLTRRanker.rank`: features precomputed once per doc (O(n)); forward passes halved via antisymmetry `pref(j,i) = 1 − pref(i,j)`

11. **`reranker.py`** - Cross-encoder reranking
    - Dense semantic reranking
    - Hybrid scoring
    - `DenseSemanticReranker.train`: query features extracted once per query; doc features precomputed per doc instead of per pair (fixes a bug where query and doc1 tensors were identical)

12. **`intention_classifier.py`** - PyTorch-based intent classification
    - Neural network for query intent detection
    - Vocabulary management

13. **`bm25_retriever.py`** - BM25 sparse retrieval
    - Classic TF-IDF based ranking

---

## API Reference

### Text Preprocessing

```python
from text_preprocessing import (
    decode, digits_to_words, spelling_correction,
    remove_stop_words, stemming, lemmatizing,
    extract_companies, preprocessing
)

# Example usage
text = preprocessing("Hello 2nd world!", debug=False)
```

### Vocabulary Management

```python
from vocabulary import Vocabulary

vocab = Vocabulary()
vocab.build([["hello", "world"], ["foo", "bar"]])
token_ids = vocab.encode(["hello", "world"])
decoded = vocab.decode(token_ids)
```

### Semantic Retrieval

```python
from semantic_retriever import SemanticRetriever

retriever = SemanticRetriever()
retriever.index_documents(documents)
results = retriever.retrieve(query, top_k=5)
```

### Query Processing

```python
from search_engine import SearchQueryProcessor

processor = SearchQueryProcessor()
result = processor.process_query("query text")
```

### Embeddings

```python
from pretrained_embeddings import PretrainedEmbeddings

embeddings = PretrainedEmbeddings.from_embeddings_file("vectors.txt")
neighbors = embeddings.get_closest_words("king", n=5)
analogy = embeddings.compute_analogy("man", "woman", "king")
```

---

## Configuration

The system uses `config.yaml` for preprocessing rules:

```yaml
dify:
  process_rules:
    pre_processing:
      - id: remove_extra_spaces
        enabled: true
      - id: remove_urls_emails
        enabled: true
    segmentation:
      separator: "(?<=[.!?])\\s+|\\n+"
```

Modify this file to customize document processing behavior.

---

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

```bash
# Install dev dependencies
pip install pytest black flake8 mypy

# Run tests
pytest

# Format code
black .

# Lint
flake8
```

---

## Module Details

| Module | Purpose | Key Classes | Dependencies |
|--------|---------|-------------|--------------|
| `text_preprocessing.py` | Text normalization pipeline | `preprocessing`, `extract_companies` | nltk, spacy |
| `vocabulary.py` | Token-to-index mapping | `Vocabulary` | — |
| `pretrained_embeddings.py` | Annoy-based nearest-neighbour search | `PretrainedEmbeddings` | annoy, numpy |
| `document_processor.py` | Config-driven document cleanup and segmentation | `TextProcessor` | re, json, PyYAML |
| `search_engine.py` | Query understanding core | `SearchQueryProcessor` | nltk, spacy, re |
| `semantic_retriever.py` | Dataset-aware in-memory semantic retrieval | `InMemorySemanticVectorStore`, `SemanticRetriever` | numpy |
| `semantic_slicing.py` | Semantic text segmentation and evaluation | `SpaCySemanticSlicer`, `ContextRelevanceScorer`, `FaithfulnessEvaluator` | spacy, numpy |
| `semantic_pipeline.py` | Async end-to-end semantic retrieval pipeline | `SemanticRetrievalPipeline`, `PerformanceEvaluator` | numpy |
| `search_recall.py` | Multi-stage recall over partner and backup pools | `SearchRecallEngine` | search_engine |
| `intention_classifier.py` | Neural query intent classification | `IntentionClassifier`, `IntentionClassificationPipeline` | torch |
| `bm25_retriever.py` | BM25 sparse retrieval | `BM25`, `BM25Retriever` | math |
| `learning_to_rank.py` | Pointwise and pairwise LTR | `PointwiseLTRRanker`, `PairwiseLTRRanker` | torch, numpy |
| `reranker.py` | Cross-encoder and dense semantic reranking | `CrossEncoderReranker`, `DenseSemanticReranker`, `HybridReranker` | torch, numpy |

---

## Usage Examples

### End-to-End Search Pipeline

```python
from search_engine import SearchQueryProcessor
from search_recall import SearchRecallEngine
from bm25_retriever import BM25Retriever
from learning_to_rank import PointwiseLTRRanker
from reranker import HybridReranker

# 1. Process query
processor = SearchQueryProcessor()
query_result = processor.process_query("museum near london")

# 2. Multi-stage recall
recall_engine = SearchRecallEngine()
recall_result = recall_engine.recall("museum near london")

# 3. Sparse retrieval (BM25)
retriever = BM25Retriever()
retrieved_docs = retriever.retrieve(query_result["lemmas"], top_k=10)

# 4. Neural ranking
ranker = PointwiseLTRRanker()
ranked_docs = ranker.rank(query_result["lemmas"], retrieved_docs, top_k=5)

# 5. Reranking
reranker = HybridReranker()
final_results = reranker.rerank(query_result["lemmas"], ranked_docs, top_k=3)
```

### Async Semantic Pipeline

```python
import asyncio
from semantic_pipeline import SemanticRetrievalPipeline

async def main():
    documents = [
        ("The British Museum houses art and artefacts from world cultures.", "doc1"),
        ("Tate Modern is a contemporary art gallery on the South Bank.", "doc2"),
    ]
    pipeline = SemanticRetrievalPipeline()
    await pipeline.build_index(documents)
    results = pipeline.retrieve("museum art gallery london", top_k=3)
    for r in results:
        print(f"{r['id']}: {r['text'][:80]} (score={r['score']:.3f})")

asyncio.run(main())
```

---

## Performance Characteristics

- **Query Processing**: O(n) where n = query length
- **BM25 Retrieval**: O(q·d) where q = query terms, d = documents
- **Pointwise LTR Ranking**: O(d·f) where d = documents, f = features
- **Pairwise LTR Ranking**: O(d·f) feature extraction + O(d²/2) forward passes — antisymmetry halves the naive O(d²) cost
- **Error Correction**: O(|vocab|·L) per token; first-char pre-filter reduces candidates ~10× before edit-distance
- **Cross-Encoder Reranking**: O(d) one-pass encoding
- **Dense Semantic Reranking**: O(d) — query encoded once, doc features precomputed per doc
- **spaCy Initialization**: O(1) after first load — models cached at module level, shared across all components

---

## Future Enhancements

1. **BERT-based embeddings** for improved semantic understanding
2. **ListWise LTR** for group-level ranking optimization
3. **Comprehensive evaluation metrics** (NDCG, MRR, MAP)
4. **A/B testing framework** for ranking models
5. **Model serialization** for deployment
6. **Real-time index updates**

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
