from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    from annoy import AnnoyIndex
except ImportError:  # pragma: no cover
    AnnoyIndex = None


class PretrainedEmbeddings:
    """Wrapper around word embeddings with an Annoy approximate nearest neighbor index."""

    def __init__(
        self,
        word_to_index: Dict[str, int],
        word_vectors: Sequence[np.ndarray],
        metric: str = "angular",
        n_trees: int = 50,
        build_index: bool = True,
    ):
        if AnnoyIndex is None:
            raise ImportError("annoy is required for PretrainedEmbeddings")

        if not word_vectors:
            raise ValueError("word_vectors must not be empty")

        self.word_to_index = dict(word_to_index)
        self.index_to_word = {index: word for word, index in self.word_to_index.items()}
        self.word_vectors = [np.asarray(vec, dtype=np.float32) for vec in word_vectors]
        self.vector_size = self.word_vectors[0].shape[0]
        self.metric = metric
        self.n_trees = n_trees
        self.index = AnnoyIndex(self.vector_size, metric=self.metric)
        self._is_index_built = False

        if build_index:
            self.build_index()

    @classmethod
    def from_embeddings_file(
        cls,
        embedding_file: str,
        delimiter: Optional[str] = None,
        normalize: bool = True,
        metric: str = "angular",
        n_trees: int = 50,
    ) -> "PretrainedEmbeddings":
        """Load embeddings from a text file into a PretrainedEmbeddings instance."""
        word_to_index: Dict[str, int] = {}
        word_vectors: List[np.ndarray] = []

        with open(embedding_file, encoding="utf-8") as fp:
            for line in fp:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(delimiter) if delimiter is not None else line.split()
                if len(parts) < 2:
                    continue

                word = parts[0]
                vector = np.asarray([float(value) for value in parts[1:]], dtype=np.float32)
                if normalize:
                    norm = np.linalg.norm(vector)
                    if norm > 0:
                        vector /= norm

                word_to_index[word] = len(word_to_index)
                word_vectors.append(vector)

        if not word_vectors:
            raise ValueError(f"No embeddings found in {embedding_file}")

        return cls(word_to_index, word_vectors, metric=metric, n_trees=n_trees)

    def build_index(self) -> None:
        """Build or rebuild the underlying Annoy index."""
        for word, index in self.word_to_index.items():
            self.index.add_item(index, self.word_vectors[index])
        self.index.build(self.n_trees)
        self._is_index_built = True

    def get_embedding(self, word: str, default: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        """Return the embedding for a word, or default if missing."""
        index = self.word_to_index.get(word)
        if index is None:
            return default
        return self.word_vectors[index]

    def get_closest_to_vector(
        self,
        vector: np.ndarray,
        n: int = 1,
        include_distances: bool = False,
    ) -> Sequence[str]:
        """Return the nearest neighbor words for a target vector."""
        if not self._is_index_built:
            self.build_index()

        nn_indices = self.index.get_nns_by_vector(vector.astype(np.float32), n)
        words = [self.index_to_word[i] for i in nn_indices]
        if include_distances:
            distances = self.index.get_nns_by_vector(vector.astype(np.float32), n, include_distances=True)[1]
            return list(zip(words, distances))
        return words

    def get_closest_words(self, word: str, n: int = 1, exclude_self: bool = True) -> List[str]:
        """Return the nearest neighbor words for a known vocabulary word."""
        vector = self.get_embedding(word)
        if vector is None:
            return []

        neighbors = self.get_closest_to_vector(vector, n + (1 if exclude_self else 0))
        if exclude_self:
            return [w for w in neighbors if w != word][:n]
        return neighbors[:n]

    def compute_analogy(
        self,
        word1: str,
        word2: str,
        word3: str,
        top_n: int = 4,
        exclude_self: bool = True,
    ) -> List[str]:
        """Solve analogies of the form word1:word2 :: word3:??."""
        vec1 = self.get_embedding(word1)
        vec2 = self.get_embedding(word2)
        vec3 = self.get_embedding(word3)
        if vec1 is None or vec2 is None or vec3 is None:
            return []

        analogy_vector = vec3 + (vec2 - vec1)
        candidates = self.get_closest_to_vector(analogy_vector, top_n + 3)

        existing = {word1, word2, word3}
        results = [word for word in candidates if word not in existing]
        return results[:top_n]

    @property
    def vocab_size(self) -> int:
        return len(self.word_to_index)
