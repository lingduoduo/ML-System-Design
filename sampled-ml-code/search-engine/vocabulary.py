from __future__ import annotations

from collections import Counter
from typing import Dict, Iterable, List, Optional, Sequence, Union

from text_preprocessing import preprocessing


TokenInput = Union[str, Sequence[str]]


class Vocabulary:
    """A token-to-index mapping with optional text_preprocessing normalization."""

    def __init__(
        self,
        token_to_idx: Optional[Dict[str, int]] = None,
        pad_token: str = "<PAD>",
        unk_token: str = "<UNK>",
        mask_token: str = "<MASK>",
        add_unk: bool = True,
        use_text_preprocessing: bool = False,
    ):
        self._token_to_idx = dict(token_to_idx or {})
        self._idx_to_token = {idx: token for token, idx in self._token_to_idx.items()}
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.mask_token = mask_token
        self.add_unk = add_unk
        self.use_text_preprocessing = use_text_preprocessing

        self.pad_index = self.add_token(self.pad_token)
        self.mask_index = self.add_token(self.mask_token)
        self.unk_index = -1
        if add_unk:
            self.unk_index = self.add_token(self.unk_token)

    def to_serializable(self) -> Dict[str, object]:
        return {
            "token_to_idx": self._token_to_idx,
            "pad_token": self.pad_token,
            "unk_token": self.unk_token,
            "mask_token": self.mask_token,
            "add_unk": self.add_unk,
            "use_text_preprocessing": self.use_text_preprocessing,
        }

    @classmethod
    def from_serializable(cls, contents: Dict[str, object]) -> "Vocabulary":
        return cls(**contents)

    def _normalize_text(self, text: str) -> List[str]:
        normalized = preprocessing(text, debug=False, max_chars=None)
        return [token for token in normalized.split() if token]

    def normalize_tokens(self, tokens: TokenInput) -> List[str]:
        if isinstance(tokens, str):
            if not self.use_text_preprocessing:
                return [token for token in tokens.split() if token]
            return self._normalize_text(tokens)

        token_list = [str(token) for token in tokens if str(token)]
        if not self.use_text_preprocessing:
            return token_list
        return self._normalize_text(" ".join(token_list))

    def add_token(self, token: str) -> int:
        if token in self._token_to_idx:
            return self._token_to_idx[token]

        index = len(self._token_to_idx)
        self._token_to_idx[token] = index
        self._idx_to_token[index] = token
        return index

    def add_many(self, tokens: TokenInput) -> List[int]:
        return [self.add_token(token) for token in self.normalize_tokens(tokens)]

    def build(
        self,
        tokens_list: Iterable[TokenInput],
        min_freq: int = 2,
        max_size: Optional[int] = None,
    ) -> None:
        counter = Counter(token for tokens in tokens_list for token in self.normalize_tokens(tokens))
        for token, frequency in counter.most_common():
            if max_size is not None and len(self._token_to_idx) >= max_size:
                break
            if frequency < min_freq:
                break
            if token not in self._token_to_idx:
                self.add_token(token)

    def lookup_token(self, token: str) -> int:
        normalized = self.normalize_tokens([token])
        token = normalized[0] if normalized else token
        if self.add_unk:
            return self._token_to_idx.get(token, self.unk_index)
        return self._token_to_idx[token]

    def lookup_index(self, index: int) -> str:
        if index not in self._idx_to_token:
            raise KeyError(f"the index ({index}) is not in the Vocabulary")
        return self._idx_to_token[index]

    def encode(self, tokens: TokenInput, max_len: Optional[int] = None) -> List[int]:
        token_ids = [self.lookup_token(token) for token in self.normalize_tokens(tokens)]
        if max_len is None:
            return token_ids
        if len(token_ids) > max_len:
            token_ids = token_ids[:max_len]
        token_ids += [self.pad_index] * (max_len - len(token_ids))
        return token_ids

    def decode(self, token_ids: Iterable[int]) -> List[str]:
        return [self._idx_to_token.get(token_id, self.unk_token) for token_id in token_ids]

    def __len__(self) -> int:
        return len(self._token_to_idx)

    def __contains__(self, token: str) -> bool:
        return token in self._token_to_idx

    @property
    def tokens(self) -> List[str]:
        return list(self._token_to_idx.keys())

    @property
    def token_to_idx(self) -> Dict[str, int]:
        return dict(self._token_to_idx)

    @property
    def idx_to_token(self) -> Dict[int, str]:
        return dict(self._idx_to_token)

    def __str__(self) -> str:
        return f"<Vocabulary(size={len(self)})>"
