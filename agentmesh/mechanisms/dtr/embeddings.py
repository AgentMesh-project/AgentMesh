"""
DTR embedding backends — richer encoders for the *produce* step.

DTR (Delta Tool Retrieval) decomposes a tool query into a reusable *semantic
base* and a novel *semantic delta*. Both the projection-intensity computation
(s_i = v_in · v_i) and hotspot-token extraction (c_j = e_j · r_hat) require a
normalized embedding space shared by queries and tokens. This module provides
pluggable backends for that space:

- ``SentenceBertEmbedder``: a Sentence-BERT-backed encoder (lazy import of
  ``sentence_transformers``) that also exposes token-level embeddings for
  hotspot extraction. When the heavy dependency is unavailable, it degrades
  gracefully to a deterministic TF-IDF / hash backend so the runtime stays
  importable and usable without GPUs or model downloads.
- ``build_embedder(model_name, backend=...)``: a factory that selects a backend
  ("auto", "sentence-bert", "tfidf", or "hash").

All vectors are L2-normalized so that a dot product equals cosine similarity,
matching the DTR cache contract in ``agentmesh.mechanisms.dtr.cache``.

The public class name here (``SentenceBertEmbedder``) is intentionally distinct
from ``cache.SemanticEmbedder`` to avoid clobbering it when the package
re-exports both via ``from .embeddings import *``.
"""

from __future__ import annotations

import hashlib
import logging
import math
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

__all__ = [
    "TokenVector",
    "SentenceBertEmbedder",
    "build_embedder",
]


# ---------------------------------------------------------------------------
# Token-level embedding container
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TokenVector:
    """A token surface form and its normalized embedding vector.

    Used by hotspot-token extraction: the contribution of token ``t_j`` to the
    uncached semantic dimension is ``c_j = vector · r_hat``.
    """

    token: str
    vector: np.ndarray


_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_+.\-]*")


def _tokenize(text: str) -> List[str]:
    """Word-level tokenization that keeps alphanumeric/technical tokens."""
    return _TOKEN_RE.findall(text)


def _l2_normalize(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        return vector
    return vector / norm


# ---------------------------------------------------------------------------
# Sentence-BERT backed embedder with lightweight fallbacks
# ---------------------------------------------------------------------------
@dataclass
class SentenceBertEmbedder:
    """Sentence-BERT compatible embedder for DTR with deterministic fallbacks.

    The preferred backend loads a SentenceTransformers checkpoint
    (default ``sentence-transformers/all-MiniLM-L6-v2``) lazily, so importing
    this module never requires ``torch`` / ``sentence_transformers``. If the
    heavy dependency is missing (or fails to load), the embedder falls back to:

    - ``tfidf``: a hashed-vocabulary TF-IDF projection (no external deps), or
    - ``hash``: a SHA-256-derived deterministic vector.

    Both fallbacks produce L2-normalized vectors in a fixed-dimensional space,
    and provide token-level vectors so the full DTR pipeline (delta projection
    + hotspot extraction) remains operational in CPU-only environments.

    Args:
        model_name: SentenceTransformers checkpoint to load when available.
        backend: One of ``"auto"``, ``"sentence-bert"``, ``"tfidf"``,
            ``"hash"``. ``"auto"`` tries Sentence-BERT then TF-IDF.
        dim: Dimensionality of fallback vectors (ignored by Sentence-BERT).
        device: Optional device override for the Sentence-BERT model.
    """

    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    backend: str = "auto"
    dim: int = 256
    device: Optional[str] = None

    _model: object = field(default=None, init=False, repr=False)
    _active_backend: str = field(default="", init=False, repr=False)
    # document-frequency table for the (light) TF-IDF fallback
    _doc_freq: Dict[int, int] = field(default_factory=dict, init=False, repr=False)
    _num_docs: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.backend not in ("auto", "sentence-bert", "tfidf", "hash"):
            raise ValueError(f"unknown embedding backend: {self.backend!r}")
        if self.dim <= 0:
            raise ValueError("dim must be positive")

    # -- backend selection --------------------------------------------------
    def _ensure_backend(self) -> str:
        """Resolve and cache the active backend, loading heavy deps lazily."""
        if self._active_backend:
            return self._active_backend

        if self.backend in ("auto", "sentence-bert"):
            model = self._try_load_sentence_bert()
            if model is not None:
                self._model = model
                self._active_backend = "sentence-bert"
                return self._active_backend
            if self.backend == "sentence-bert":
                raise RuntimeError(
                    "sentence-transformers backend requested but unavailable"
                )

        # Fallbacks (no heavy deps).
        self._active_backend = "hash" if self.backend == "hash" else "tfidf"
        logger.info("DTR embedder using '%s' backend", self._active_backend)
        return self._active_backend

    def _try_load_sentence_bert(self):
        try:
            from sentence_transformers import SentenceTransformer  # lazy heavy import
        except Exception:  # pragma: no cover - exercised only without the dep
            logger.warning(
                "sentence-transformers unavailable; falling back to TF-IDF backend"
            )
            return None
        try:
            kwargs = {}
            if self.device is not None:
                kwargs["device"] = self.device
            model = SentenceTransformer(self.model_name, **kwargs)
            logger.info("Loaded Sentence-BERT model: %s", self.model_name)
            return model
        except Exception as exc:  # pragma: no cover - download/load failures
            logger.warning("failed to load %s (%s); using TF-IDF", self.model_name, exc)
            return None

    @property
    def active_backend(self) -> str:
        """Return the resolved backend name (loads lazily on first call)."""
        return self._ensure_backend()

    # -- query / sentence embeddings ---------------------------------------
    def encode_query(self, text: str) -> np.ndarray:
        """Return one L2-normalized sentence embedding for ``text``."""
        return self.encode_queries([text])[0]

    def encode_queries(self, texts: List[str]) -> np.ndarray:
        """Return a 2-D array of L2-normalized sentence embeddings."""
        if not texts:
            return np.empty((0, self.dim), dtype=np.float32)

        backend = self._ensure_backend()
        if backend == "sentence-bert":
            vecs = self._model.encode(texts, convert_to_numpy=True).astype(np.float32)
            return np.vstack([_l2_normalize(v) for v in vecs])
        if backend == "tfidf":
            return np.vstack([self._tfidf_vector(t) for t in texts])
        return np.vstack([self._hash_vector(t) for t in texts])

    # -- token embeddings for hotspot extraction ---------------------------
    def token_embeddings(self, text: str) -> List[TokenVector]:
        """Return normalized token vectors in the same space as queries.

        For Sentence-BERT, each token is independently encoded (mean-pooled
        sentence vector of the single token) so it shares the query space. The
        fallbacks embed tokens with the same hashing scheme used for queries.
        """
        tokens = _tokenize(text)
        if not tokens:
            return []

        backend = self._ensure_backend()
        if backend == "sentence-bert":
            vecs = self._model.encode(tokens, convert_to_numpy=True).astype(np.float32)
            return [
                TokenVector(token=tok, vector=_l2_normalize(vec))
                for tok, vec in zip(tokens, vecs)
            ]
        if backend == "tfidf":
            return [
                TokenVector(token=tok, vector=self._hash_vector(tok))
                for tok in tokens
            ]
        return [
            TokenVector(token=tok, vector=self._hash_vector(tok)) for tok in tokens
        ]

    def embed_tokens(self, tokens: List[str]) -> np.ndarray:
        """Return a 2-D array of normalized token vectors (one row per token)."""
        if not tokens:
            return np.empty((0, self.dim), dtype=np.float32)
        backend = self._ensure_backend()
        if backend == "sentence-bert":
            vecs = self._model.encode(tokens, convert_to_numpy=True).astype(np.float32)
            return np.vstack([_l2_normalize(v) for v in vecs])
        return np.vstack([self._hash_vector(t) for t in tokens])

    # -- deterministic fallbacks -------------------------------------------
    def _token_bucket(self, token: str) -> int:
        digest = hashlib.sha256(token.lower().encode("utf-8")).digest()
        return int.from_bytes(digest[:8], "big") % self.dim

    def _hash_vector(self, text: str) -> np.ndarray:
        """Deterministic bag-of-hashed-tokens vector, L2-normalized."""
        vec = np.zeros(self.dim, dtype=np.float32)
        for token in _tokenize(text) or [text]:
            vec[self._token_bucket(token)] += 1.0
        if not np.any(vec):
            # empty text: hash the raw string into one bucket
            vec[self._token_bucket(text or " ")] = 1.0
        return _l2_normalize(vec)

    def _tfidf_vector(self, text: str) -> np.ndarray:
        """Hashed-vocabulary TF-IDF vector with an online document-frequency
        table, L2-normalized. Falls back to term-frequency when a token is
        previously unseen (idf defaults to 1)."""
        tokens = _tokenize(text)
        if not tokens:
            return self._hash_vector(text)

        # term frequencies per hashed bucket
        tf: Dict[int, float] = {}
        seen_buckets = set()
        for token in tokens:
            bucket = self._token_bucket(token)
            tf[bucket] = tf.get(bucket, 0.0) + 1.0
            seen_buckets.add(bucket)

        # update online document frequencies (smoothed idf)
        self._num_docs += 1
        for bucket in seen_buckets:
            self._doc_freq[bucket] = self._doc_freq.get(bucket, 0) + 1

        vec = np.zeros(self.dim, dtype=np.float32)
        n_docs = max(self._num_docs, 1)
        for bucket, freq in tf.items():
            df = self._doc_freq.get(bucket, 1)
            idf = math.log((1.0 + n_docs) / (1.0 + df)) + 1.0
            vec[bucket] = freq * idf
        return _l2_normalize(vec)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def build_embedder(
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    backend: str = "auto",
    *,
    dim: int = 256,
    device: Optional[str] = None,
) -> SentenceBertEmbedder:
    """Construct a DTR embedder.

    Args:
        model_name: SentenceTransformers checkpoint used by the Sentence-BERT
            backend.
        backend: ``"auto"`` (Sentence-BERT then TF-IDF), ``"sentence-bert"``,
            ``"tfidf"``, or ``"hash"``.
        dim: Dimensionality of the deterministic fallback vectors.
        device: Optional device override for the Sentence-BERT model.

    Returns:
        A configured :class:`SentenceBertEmbedder`. The backend is resolved
        lazily on first use, so this call never imports heavy dependencies.
    """
    return SentenceBertEmbedder(
        model_name=model_name,
        backend=backend,
        dim=dim,
        device=device,
    )
