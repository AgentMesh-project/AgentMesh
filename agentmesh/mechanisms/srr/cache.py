"""
SRR (Semantic Residual Retrieval)

Implements semantic caching with projection-based residual extraction and SLM query reformulation for tool interactions.

Core idea: decompose a tool query into a *semantic base* (reusable from cache)
and a *semantic residual* (novel intent requiring incremental retrieval).

Key formulas:
- Projection intensity: s_i = v_in · v_i  (cosine similarity for normalized vectors)
- Semantic residual:    r = v_in - s_max * v_max
- Token contribution:   c_j = e_j · r_hat  (hotspot token selection)
- Item allocation:      reuse ⌊s_max·n⌋ cached items, fetch ⌈(1-s_max)·n⌉ fresh items
- Confidence gating:    activate SRR only when s_max ≥ τ (default τ=0.7)
"""

import numpy as np
from typing import List, Optional, Dict, Tuple, Any, Callable
from dataclasses import dataclass, field
from collections import OrderedDict
import hashlib
import time
import logging
import asyncio
import threading

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SLM Query Reformulation Prompt
# ---------------------------------------------------------------------------
SLM_REFORMULATION_PROMPT = """You are a Query Reformulator for a retrieval tool. You are provided with:

1. Original query: {query}

2. Hotspot tokens: {hotspot_tokens}

Your task is to rewrite a single search query that satisfies the following:

1. Keep the subject of the original query.

2. Highlight the hotspot tokens.

3. Do not include introductory phrases. Focus strictly on the intersection of the subject and the hotspots."""


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class CacheEntry:
    """A single cache entry storing query, retrieved items, and metadata."""
    query: str
    items: List[str]            # Retrieved semantic items (not a single response)
    embedding: np.ndarray       # Normalized query embedding vector
    timestamp: float = field(default_factory=time.time)
    hit_count: int = 0

    def update_hit(self):
        """Update hit statistics."""
        self.hit_count += 1
        self.timestamp = time.time()


@dataclass
class CacheResult:
    """Result of an SRR cache lookup."""
    hit: bool                                           # Whether SRR was activated
    projection_intensity: float = 0.0                   # s_max
    cached_entry: Optional[CacheEntry] = None           # Best-matching cache entry
    reuse_count: int = 0                                # ⌊s_max·n⌋ items reused
    residual_count: int = 0                             # ⌈(1-s_max)·n⌉ items to fetch
    residual_vector: Optional[np.ndarray] = None        # r = v_in - s_max * v_max
    hotspot_tokens: List[str] = field(default_factory=list)  # Top-k residual tokens
    reformulated_query: Optional[str] = None            # SLM-generated differential query

    @property
    def residual_ratio(self) -> float:
        """Ratio of residual items to total retrieval depth."""
        total = self.reuse_count + self.residual_count
        if total == 0:
            return 1.0
        return self.residual_count / total


# ---------------------------------------------------------------------------
# Semantic Embedder
# ---------------------------------------------------------------------------
class SemanticEmbedder:
    """
    Generates normalized semantic embeddings using sentence-transformers.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None

    def _load_model(self):
        """Lazy load the embedding model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name)
                logger.info(f"Loaded embedding model: {self.model_name}")
            except ImportError:
                logger.warning(
                    "sentence-transformers not installed, using hash-based embedding"
                )
                self._model = "hash"

    def embed(self, text: str) -> np.ndarray:
        """
        Generate a **normalized** embedding for text.

        Returns a unit vector so that dot product equals cosine similarity:
            s_i = v_in · v_i  (since ||v|| = 1 for normalized vectors)
        """
        self._load_model()

        if self._model == "hash":
            hash_hex = hashlib.sha256(text.encode()).hexdigest()
            vec = np.array(
                [int(hash_hex[i : i + 2], 16) / 255.0 for i in range(0, 64, 2)],
                dtype=np.float32,
            )
            norm = np.linalg.norm(vec)
            return vec / norm if norm > 0 else vec

        vec = self._model.encode(text, convert_to_numpy=True).astype(np.float32)
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Embed multiple texts, each row normalized."""
        self._load_model()

        if self._model == "hash":
            return np.array([self.embed(t) for t in texts])

        vecs = self._model.encode(texts, convert_to_numpy=True).astype(np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        return vecs / norms

    def embed_tokens(self, tokens: List[str]) -> np.ndarray:
        """Embed individual tokens for hotspot extraction (c_j = e_j · r_hat)."""
        return self.embed_batch(tokens)


# ---------------------------------------------------------------------------
# Hotspot Token Extractor
# ---------------------------------------------------------------------------
class HotspotTokenExtractor:
    """
    Identifies residual-characterized tokens from the query (§4.2).

    For each token t_j in Q_in, computes its contribution to the uncached
    semantic dimension:  c_j = e_j · r_hat

    The top-k tokens (default k=3) are chosen as hotspot tokens for
    SLM query reformulation.
    """

    def __init__(self, embedder: SemanticEmbedder, k: int = 3):
        self.embedder = embedder
        self.k = k

    def tokenize(self, text: str) -> List[str]:
        """Simple word-level tokenization."""
        return text.split()

    def extract_hotspot_tokens(
        self,
        query: str,
        residual_vector: np.ndarray,
    ) -> List[str]:
        """
        Extract top-k hotspot tokens from query based on residual direction.

        Args:
            query: Original query Q_in.
            residual_vector: Normalized residual vector r_hat.

        Returns:
            Top-k tokens with highest projection intensity onto r_hat.
        """
        tokens = self.tokenize(query)
        if not tokens:
            return []

        # Embed each token: {e_j}
        token_embeddings = self.embedder.embed_tokens(tokens)

        # Compute contribution: c_j = e_j · r_hat
        contributions = token_embeddings @ residual_vector

        # Select top-k
        k = min(self.k, len(tokens))
        top_indices = np.argsort(contributions)[-k:][::-1]

        return [tokens[i] for i in top_indices]


# ---------------------------------------------------------------------------
# SLM Query Reformulator
# ---------------------------------------------------------------------------
class QueryReformulator:
    """
    Reformulates a query emphasizing hotspot tokens using a small language model.

    The reformulated query focuses on the semantic residual — intent that
    distinguishes Q_in from cached queries — to ensure incremental
    retrieval is maximally relevant.
    """

    def __init__(
        self,
        slm_model: str = "Qwen/Qwen2.5-0.5B-Instruct",
        slm_client: Optional[Any] = None,
    ):
        self.slm_model = slm_model
        self.slm_client = slm_client

    def reformulate(
        self, query: str, hotspot_tokens: List[str]
    ) -> str:
        """
        Generate a differential query emphasizing hotspot tokens.

        If no SLM client is available, falls back to a simple
        keyword-based reformulation.
        """
        if not hotspot_tokens:
            return query

        # Try SLM-based reformulation
        if self.slm_client is not None:
            try:
                prompt = SLM_REFORMULATION_PROMPT.format(
                    query=query,
                    hotspot_tokens=", ".join(hotspot_tokens),
                )
                response = self.slm_client.chat.completions.create(
                    model=self.slm_model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=64,
                    temperature=0.3,
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                logger.warning(f"SLM reformulation failed, using fallback: {e}")

        # Fallback: simple keyword-emphasis reformulation
        return f"{query} focusing on {', '.join(hotspot_tokens)}"


# ---------------------------------------------------------------------------
# SRR Cache
# ---------------------------------------------------------------------------
class SRRCache:
    """
    Semantic Residual Retrieval Cache.
    Implements the complete SRR pipeline:
    1. Embed incoming query as normalized vector v_in
    2. Compute projection intensity s_i = v_in · v_i for each cached query
    3. Confidence gating: s_max ≥ τ → SRR activation; s_max < τ → cold retrieval
    4. SRR activation:
       a) Reuse ⌊s_max·n⌋ items from best-matching cache entry
       b) Compute residual r = v_in - s_max * v_max
       c) Extract hotspot tokens via c_j = e_j · r_hat
       d) SLM reformulates differential query
       e) Fetch remaining ⌈(1-s_max)·n⌉ items via reformulated query
    5. Asynchronous cache update: background full execution prevents drift

    Key parameters:
    - τ (confidence_threshold): default 0.7 per paper evaluation
    - n (retrieval_depth): standard number of items per tool query
    - k (hotspot_k): top-k residual tokens, default 3
    """

    def __init__(
        self,
        confidence_threshold: float = 0.7,
        retrieval_depth: int = 10,
        cache_size: int = 1000,
        hotspot_k: int = 3,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        slm_model: str = "Qwen/Qwen2.5-0.5B-Instruct",
        slm_client: Optional[Any] = None,
    ):
        """
        Initialize the SRR cache.

        Args:
            confidence_threshold: Semantic confidence threshold τ (default 0.7).
            retrieval_depth: Standard retrieval depth n (items per query).
            cache_size: Maximum number of entries in cache pool.
            hotspot_k: Number of hotspot tokens for residual analysis.
            embedding_model: Sentence-transformer model for embeddings.
            slm_model: Small language model for query reformulation.
            slm_client: Optional pre-configured SLM client.
        """
        self.confidence_threshold = confidence_threshold  # τ
        self.retrieval_depth = retrieval_depth            # n
        self.cache_size = cache_size
        self.hotspot_k = hotspot_k

        # Core components
        self.embedder = SemanticEmbedder(embedding_model)
        self.hotspot_extractor = HotspotTokenExtractor(self.embedder, k=hotspot_k)
        self.reformulator = QueryReformulator(slm_model, slm_client)

        # Cache storage (OrderedDict for LRU eviction)
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()

        # Statistics
        self.stats = {
            "total_lookups": 0,
            "srr_activations": 0,       # s_max ≥ τ
            "cold_retrievals": 0,       # s_max < τ
            "total_items_reused": 0,
            "total_items_fetched": 0,
            "total_reformulations": 0,
            "async_updates": 0,
        }

    def lookup(
        self,
        query: str,
        tool_fn: Optional[Callable] = None,
    ) -> CacheResult:
        """
        SRR cache lookup implementing Algorithm 1.

        Args:
            query: Incoming tool query Q_in.
            tool_fn: Optional tool execution function for async cache update.

        Returns:
            CacheResult with reuse/residual allocation and reformulated query.
        """
        self.stats["total_lookups"] += 1
        n = self.retrieval_depth

        # Step 1: Embed input query as normalized vector
        v_in = self.embedder.embed(query)

        # Step 2: Find max projection intensity across cache
        s_max = 0.0
        v_max = None
        best_entry: Optional[CacheEntry] = None

        for entry in self._cache.values():
            # s_i = v_in · v_i (cosine similarity for normalized vectors)
            s_i = float(np.dot(v_in, entry.embedding))
            if s_i > s_max:
                s_max = s_i
                v_max = entry.embedding
                best_entry = entry

        # Step 3: Confidence gating
        if s_max < self.confidence_threshold or best_entry is None:
            # --- Cold Retrieval ---
            self.stats["cold_retrievals"] += 1
            self.stats["total_items_fetched"] += n
            logger.debug(f"SRR Cold Retrieval: s_max={s_max:.3f} < τ={self.confidence_threshold}")
            return CacheResult(
                hit=False,
                projection_intensity=s_max,
                reuse_count=0,
                residual_count=n,
            )

        # --- SRR Activation (s_max ≥ τ) ---
        self.stats["srr_activations"] += 1

        # Update LRU order
        key = self._generate_key(best_entry.query)
        if key in self._cache:
            self._cache.move_to_end(key)
        best_entry.update_hit()

        # Step 4a: Item allocation
        k_reuse = int(np.floor(s_max * n))
        k_residual = n - k_reuse
        self.stats["total_items_reused"] += k_reuse
        self.stats["total_items_fetched"] += k_residual

        # Step 4b: Compute semantic residual vector
        # r = v_in - s_max * v_max (projection subtraction)
        residual = v_in - s_max * v_max
        r_norm = np.linalg.norm(residual)

        hotspot_tokens = []
        reformulated_query = None

        if k_residual > 0 and r_norm > 1e-8:
            # Normalize residual: r_hat = r / ||r||
            r_hat = residual / r_norm

            # Step 4c: Extract hotspot tokens
            hotspot_tokens = self.hotspot_extractor.extract_hotspot_tokens(query, r_hat)

            # Step 4d: SLM reformulates differential query
            if hotspot_tokens:
                reformulated_query = self.reformulator.reformulate(query, hotspot_tokens)
                self.stats["total_reformulations"] += 1

        # Step 5: Schedule asynchronous cache update (background full execution)
        if tool_fn is not None:
            self._schedule_async_update(query, v_in, tool_fn)

        logger.debug(
            f"SRR Activation: s_max={s_max:.3f}, reuse={k_reuse}, "
            f"residual={k_residual}, hotspots={hotspot_tokens}"
        )

        return CacheResult(
            hit=True,
            projection_intensity=s_max,
            cached_entry=best_entry,
            reuse_count=k_reuse,
            residual_count=k_residual,
            residual_vector=residual,
            hotspot_tokens=hotspot_tokens,
            reformulated_query=reformulated_query,
        )

    def store(
        self,
        query: str,
        items: List[str],
        embedding: Optional[np.ndarray] = None,
    ):
        """
        Store query-items pair in cache pool.

        Args:
            query: Tool query string.
            items: Retrieved semantic items (list of contents/snippets).
            embedding: Optional pre-computed normalized embedding.
        """
        key = self._generate_key(query)

        if embedding is None:
            embedding = self.embedder.embed(query)

        entry = CacheEntry(query=query, items=items, embedding=embedding)

        # LRU eviction
        while len(self._cache) >= self.cache_size:
            self._cache.popitem(last=False)

        self._cache[key] = entry
        logger.debug(f"Stored cache entry: key={key[:8]}..., items={len(items)}")

    def get_reused_items(
        self, result: CacheResult
    ) -> List[str]:
        """
        Get the reused items from a cache hit.

        Returns the top ⌊s_max·n⌋ items from the cached entry.
        """
        if not result.hit or result.cached_entry is None:
            return []
        return result.cached_entry.items[: result.reuse_count]

    def _generate_key(self, query: str) -> str:
        """Generate cache key from query."""
        return hashlib.md5(query.encode()).hexdigest()

    def _schedule_async_update(
        self,
        query: str,
        embedding: np.ndarray,
        tool_fn: Callable,
    ):
        """
        Schedule asynchronous cache update.

        Runs the full tool execution in background to update the cache
        with authentic results, preventing cumulative drift from residual
        approximation.
        """
        def _background_update():
            try:
                items = tool_fn(query)
                self.store(query, items, embedding)
                self.stats["async_updates"] += 1
                logger.debug(f"Async cache update completed for: {query[:50]}...")
            except Exception as e:
                logger.warning(f"Async cache update failed: {e}")

        thread = threading.Thread(target=_background_update, daemon=True)
        thread.start()

    def get_hit_rate(self) -> float:
        """Get SRR activation rate (fraction of lookups with s_max ≥ τ)."""
        total = self.stats["total_lookups"]
        return self.stats["srr_activations"] / total if total > 0 else 0.0

    def get_item_reuse_rate(self) -> float:
        """Get fraction of items served from cache vs. total items."""
        total = self.stats["total_items_reused"] + self.stats["total_items_fetched"]
        return self.stats["total_items_reused"] / total if total > 0 else 0.0

    def get_stats(self) -> Dict:
        """Get cache statistics."""
        return {
            **self.stats,
            "cache_entries": len(self._cache),
            "hit_rate": self.get_hit_rate(),
            "item_reuse_rate": self.get_item_reuse_rate(),
        }

    def clear(self):
        """Clear the cache pool and reset statistics."""
        self._cache.clear()
        self.stats = {
            "total_lookups": 0,
            "srr_activations": 0,
            "cold_retrievals": 0,
            "total_items_reused": 0,
            "total_items_fetched": 0,
            "total_reformulations": 0,
            "async_updates": 0,
        }


# Convenience function
def create_srr_cache(
    confidence_threshold: float = 0.7,
    retrieval_depth: int = 10,
    cache_size: int = 1000,
    hotspot_k: int = 3,
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    slm_model: str = "Qwen/Qwen2.5-0.5B-Instruct",
    **kwargs,
) -> SRRCache:
    """
    Create an SRR cache instance.
    
    Args:
        confidence_threshold: τ — semantic confidence gate (default 0.7).
        retrieval_depth: n — items per query (default 10).
        cache_size: Maximum cache pool size.
        hotspot_k: k — top-k residual tokens (default 3).
        embedding_model: Encoder model name.
        slm_model: SLM for query reformulation.

    Returns:
        Configured SRRCache instance.
    """
    return SRRCache(
        confidence_threshold=confidence_threshold,
        retrieval_depth=retrieval_depth,
        cache_size=cache_size,
        hotspot_k=hotspot_k,
        embedding_model=embedding_model,
        slm_model=slm_model,
        **kwargs,
    )
