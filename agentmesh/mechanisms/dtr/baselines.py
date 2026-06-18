"""
DTR evaluation baselines — apples-to-apples comparison points.

These baselines reproduce the caching strategies DTR is compared against in the
paper's tool-result reuse study. Each exposes the *same* interface as
``DTRCache.lookup`` so a benchmark harness can swap strategies without changing
its driver:

    result = baseline.lookup(query, cache) -> CacheResult

where ``cache`` is a populated :class:`~agentmesh.mechanisms.dtr.cache.DTRCache`
(the baseline reads its stored entries and shared embedder). The returned
:class:`~agentmesh.mechanisms.dtr.cache.CacheResult` carries ``reuse_count`` (base
items served from cache) and ``residual_count`` (delta items that must be fetched),
exactly as DTR reports them — so reuse / fetch volume is directly comparable.

Baselines
---------
- :class:`NoCacheBaseline` (NC): never reuses; every query fetches all ``n`` items.
- :class:`ExactMatchBaseline` (EM): reuses all ``n`` items only on an exact (or
  cosine ≈ 1) query match; otherwise full fetch. No delta retrieval.
- :class:`FixedFractionReuseBaseline` (FFR): when the best match clears a
  similarity threshold, reuse a *fixed fraction* of cached items and fetch the
  rest — but never reformulate a delta query (so the fetch is a plain top-up).
- :class:`CortexBaseline` (Cortex): all-or-nothing reuse gated by an
  LLM-judge (lazy OpenAI/transformers, deterministic heuristic fallback): if the
  judge approves, reuse all ``n`` items; otherwise full fetch.

DTR's own advantage over FFR/Cortex is that it fetches only the *semantic delta*
via a reformulated query rather than reusing/discarding whole result sets.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, List, Optional, Tuple

import numpy as np

from agentmesh.mechanisms.dtr.cache import CacheEntry, CacheResult

logger = logging.getLogger(__name__)

__all__ = [
    "NoCacheBaseline",
    "ExactMatchBaseline",
    "FixedFractionReuseBaseline",
    "CortexBaseline",
]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def _best_projection(
    query: str, cache: Any
) -> Tuple[Optional[CacheEntry], float, np.ndarray]:
    """Embed ``query`` with the cache's embedder and find the max-projection entry.

    Returns ``(best_entry, s_max, v_in)``. ``best_entry`` is ``None`` when the
    cache is empty.
    """
    v_in = cache.embedder.embed(query)
    s_max = 0.0
    best_entry: Optional[CacheEntry] = None
    # DTRCache stores entries in a private OrderedDict; iterate its values.
    for entry in getattr(cache, "_cache").values():
        s_i = float(np.dot(v_in, entry.embedding))
        if s_i > s_max:
            s_max = s_i
            best_entry = entry
    return best_entry, s_max, v_in


def _retrieval_depth(cache: Any) -> int:
    return int(getattr(cache, "retrieval_depth", 10))


# ---------------------------------------------------------------------------
# No-Cache (NC)
# ---------------------------------------------------------------------------
@dataclass
class NoCacheBaseline:
    """NC baseline: disable reuse entirely; every query fetches all ``n`` items."""

    name: str = "no_cache"

    def lookup(self, query: str, cache: Any) -> CacheResult:
        n = _retrieval_depth(cache)
        return CacheResult(
            hit=False,
            projection_intensity=0.0,
            reuse_count=0,
            residual_count=n,
        )


# ---------------------------------------------------------------------------
# Exact-Match (EM)
# ---------------------------------------------------------------------------
@dataclass
class ExactMatchBaseline:
    """EM baseline: reuse the full cached result only on an exact query match.

    A match is exact when the cached query string equals ``query`` or its cosine
    similarity is at/above ``match_threshold`` (defaults to ~1.0 to capture
    embedding round-off). No delta retrieval is performed.
    """

    match_threshold: float = 0.999
    name: str = "exact_match"

    def lookup(self, query: str, cache: Any) -> CacheResult:
        n = _retrieval_depth(cache)
        best_entry, s_max, _ = _best_projection(query, cache)
        exact = best_entry is not None and (
            best_entry.query == query or s_max >= self.match_threshold
        )
        if not exact:
            return CacheResult(
                hit=False,
                projection_intensity=s_max,
                reuse_count=0,
                residual_count=n,
            )
        reuse_count = min(len(best_entry.items), n)
        return CacheResult(
            hit=True,
            projection_intensity=s_max,
            cached_entry=best_entry,
            reuse_count=reuse_count,
            residual_count=n - reuse_count,
        )


# ---------------------------------------------------------------------------
# Fixed-Fraction-Reuse (FFR)
# ---------------------------------------------------------------------------
@dataclass
class _FractionalReuseBudget:
    """Deterministically amortize fractional reuse across repeated hits."""

    reuse_fraction: float
    carry: Fraction = field(default_factory=lambda: Fraction(0, 1))
    _fraction: Fraction = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._fraction = Fraction(str(self.reuse_fraction))

    def allocate(self, available: int, *, minimum_one: bool = True) -> int:
        if available <= 0 or self._fraction <= 0:
            return 0
        desired = self._fraction * available + self.carry
        reuse_count = int(desired)
        if reuse_count <= 0:
            return 1 if minimum_one else 0
        if reuse_count > available:
            reuse_count = available
        self.carry = desired - reuse_count
        return reuse_count


@dataclass
class FixedFractionReuseBaseline:
    """FFR baseline: above a similarity threshold, reuse a fixed fraction.

    When the best cached entry clears ``similarity_threshold``, reuse
    ``floor(reuse_fraction * available)`` items (amortized so the long-run reuse
    rate matches ``reuse_fraction``) and report the remaining items as a plain
    fetch (``residual_count``). Unlike DTR, the fetch is *not* driven by a
    reformulated delta query — the missing items are simply re-retrieved.
    """

    similarity_threshold: float = 0.7
    reuse_fraction: float = 0.8
    name: str = "fixed_fraction_reuse"

    _budget: _FractionalReuseBudget = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not 0.0 <= self.reuse_fraction <= 1.0:
            raise ValueError("reuse_fraction must be in [0, 1]")
        if not 0.0 <= self.similarity_threshold <= 1.0:
            raise ValueError("similarity_threshold must be in [0, 1]")
        self._budget = _FractionalReuseBudget(self.reuse_fraction)

    def lookup(self, query: str, cache: Any) -> CacheResult:
        n = _retrieval_depth(cache)
        best_entry, s_max, _ = _best_projection(query, cache)
        if best_entry is None or s_max < self.similarity_threshold:
            return CacheResult(
                hit=False,
                projection_intensity=s_max,
                reuse_count=0,
                residual_count=n,
            )
        available = min(len(best_entry.items), n)
        reuse_count = self._budget.allocate(available)
        return CacheResult(
            hit=True,
            projection_intensity=s_max,
            cached_entry=best_entry,
            reuse_count=reuse_count,
            residual_count=n - reuse_count,
        )


# ---------------------------------------------------------------------------
# Cortex (LLM-judge-gated all-or-nothing reuse)
# ---------------------------------------------------------------------------
@dataclass
class CortexBaseline:
    """Cortex baseline: all-or-nothing reuse gated by an LLM judge.

    A candidate cached entry is first selected by cosine similarity
    (``candidate_threshold``). An LLM judge then decides whether the cached
    result fully answers the new query. On approval, reuse *all* ``n`` items; on
    rejection, fetch all ``n`` items. There is no partial reuse and no delta
    retrieval.

    The judge uses a lazy OpenAI-compatible client when one is supplied
    (``judge_client``), otherwise a deterministic lexical-overlap heuristic so
    the baseline runs without heavy dependencies.
    """

    candidate_threshold: float = 0.9
    judge_threshold: float = 0.5
    judge_model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    judge_client: Optional[object] = None
    name: str = "cortex"

    _judge_failed: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        if not 0.0 <= self.candidate_threshold <= 1.0:
            raise ValueError("candidate_threshold must be in [0, 1]")
        if not 0.0 <= self.judge_threshold <= 1.0:
            raise ValueError("judge_threshold must be in [0, 1]")

    def lookup(self, query: str, cache: Any) -> CacheResult:
        n = _retrieval_depth(cache)
        best_entry, s_max, _ = _best_projection(query, cache)
        if best_entry is None or s_max < self.candidate_threshold:
            return CacheResult(
                hit=False,
                projection_intensity=s_max,
                reuse_count=0,
                residual_count=n,
            )

        approved, confidence = self._judge(query, best_entry, s_max)
        if approved and confidence >= self.judge_threshold:
            reuse_count = min(len(best_entry.items), n)
            return CacheResult(
                hit=True,
                projection_intensity=s_max,
                cached_entry=best_entry,
                reuse_count=reuse_count,
                residual_count=n - reuse_count,
            )
        # Rejected: full fetch.
        return CacheResult(
            hit=False,
            projection_intensity=s_max,
            cached_entry=best_entry,
            reuse_count=0,
            residual_count=n,
        )

    # -- judging -----------------------------------------------------------
    def _judge(
        self, query: str, entry: CacheEntry, similarity: float
    ) -> Tuple[bool, float]:
        if self.judge_client is not None and not self._judge_failed:
            verdict = self._llm_judge(query, entry, similarity)
            if verdict is not None:
                return verdict
            self._judge_failed = True
        return self._heuristic_judge(query, entry, similarity)

    def _llm_judge(
        self, query: str, entry: CacheEntry, similarity: float
    ) -> Optional[Tuple[bool, float]]:
        prompt = (
            "You are a semantic cache judge for an LLM tool cache.\n"
            "Decide whether the cached query and result can genuinely answer "
            "the new query.\n"
            'Respond with JSON only: {"reuse": true/false, "confidence": 0.0-1.0}.\n'
            f"Incoming query: {query}\n"
            f"Cached query: {entry.query}\n"
            f"Embedding similarity: {similarity:.4f}\n"
            "Cached result summary:\n"
            f"{_summarize_items(entry.items)}\n"
        )
        try:
            response = self.judge_client.chat.completions.create(
                model=self.judge_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=24,
                temperature=0.0,
            )
            text = response.choices[0].message.content or ""
            return _parse_judge_output(text, similarity)
        except Exception as exc:
            logger.warning("Cortex LLM judge failed: %s", exc)
            return None

    def _heuristic_judge(
        self, query: str, entry: CacheEntry, similarity: float
    ) -> Tuple[bool, float]:
        query_terms = _content_terms(query)
        cached_terms = _content_terms(entry.query)
        lexical = _jaccard(query_terms, cached_terms)
        result_terms = _content_terms(_summarize_items(entry.items))
        result_overlap = _jaccard(query_terms, result_terms)
        confidence = 0.65 * float(similarity) + 0.20 * lexical + 0.15 * result_overlap
        if _has_freshness_marker(query_terms) and not _has_freshness_marker(cached_terms):
            confidence -= 0.15
        confidence = float(np.clip(confidence, 0.0, 1.0))
        return confidence >= self.judge_threshold, confidence


# ---------------------------------------------------------------------------
# Judge helpers
# ---------------------------------------------------------------------------
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "by", "for", "from", "how",
    "in", "of", "on", "or", "the", "to", "with",
}

_FRESHNESS_TERMS = {
    "current", "latest", "recent", "today", "now", "new",
    "this", "present", "live", "breaking",
}


def _content_terms(text: str) -> set:
    return {
        token.lower()
        for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9_+.\-]*", text)
        if token.lower() not in _STOPWORDS
    }


def _summarize_items(items: List[str], limit: int = 3) -> str:
    if not items:
        return "- empty"
    lines: List[str] = []
    for item in items[:limit]:
        snippet = re.sub(r"\s+", " ", str(item)).strip()
        if len(snippet) > 160:
            snippet = snippet[:157] + "..."
        lines.append(f"- {snippet}" if snippet else "- item")
    return "\n".join(lines)


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _has_freshness_marker(terms: set) -> bool:
    return any(term in _FRESHNESS_TERMS for term in terms)


def _parse_judge_output(text: str, similarity: float) -> Tuple[bool, float]:
    cleaned = (text or "").strip()
    if not cleaned:
        return False, float(similarity)
    try:
        import json

        candidate = cleaned
        if "{" in cleaned and "}" in cleaned:
            candidate = cleaned[cleaned.find("{"): cleaned.rfind("}") + 1]
        parsed = json.loads(candidate)
        reuse = bool(parsed.get("reuse", False))
        confidence = float(parsed.get("confidence", similarity))
        return reuse, float(np.clip(confidence, 0.0, 1.0))
    except Exception:
        pass
    lowered = cleaned.lower()
    reuse = "yes" in lowered or "true" in lowered
    match = re.search(r"([01](?:\.\d+)?)", cleaned)
    confidence = float(match.group(1)) if match else float(similarity)
    return reuse, float(np.clip(confidence, 0.0, 1.0))
