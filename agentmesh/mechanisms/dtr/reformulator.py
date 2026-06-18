"""
DTR residual-query reformulators — building the *delta* query.

After DTR computes the semantic delta ``r = v_in - s_max * v_max`` and extracts
hotspot tokens via ``c_j = e_j · r_hat``, it must turn ``(query, hotspot_tokens)``
into a single differential query ``q_diff`` that fetches only the missing
(uncached) items. This module provides selectable reformulation modes behind a
uniform callable interface ``(query, hotspot_tokens) -> q_diff``:

- ``"identity"``: return the original query unchanged (controlled-replay baseline).
- ``"heuristic"``: deterministic keyword reformulation — keeps the subject and
  appends hotspot tokens, no external dependencies.
- ``"slm"``: a small (<1B) language model reformulator. Tries an
  OpenAI-compatible chat endpoint first, then a local ``transformers`` causal
  LM, and falls back to the heuristic mode if neither is available. All heavy
  imports (``openai`` / ``transformers`` / ``torch``) are lazy, so importing
  this module never requires them.

``build_reformulator(mode, ...)`` returns a callable reformulator. The public
class name here is intentionally distinct from ``cache.QueryReformulator`` so
re-exporting both via ``from .reformulator import *`` does not clobber it.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger(__name__)

__all__ = [
    "ReformulatorMode",
    "DTR_REFORMULATION_PROMPT",
    "build_reformulation_prompt",
    "DeltaReformulator",
    "build_reformulator",
]


# ---------------------------------------------------------------------------
# Reformulation prompt (paper Appendix A, exact wording)
# ---------------------------------------------------------------------------
DTR_REFORMULATION_PROMPT = """You are a Query Reformulator for a retrieval tool. You are provided with:

1. Original query: {query}

2. Hotspot tokens: {hotspot_tokens}

Your task is to rewrite a single search query that satisfies the following:

1. Keep the subject of the original query.

2. Highlight the hotspot words.

3. Do not include introductory phrases. Focus strictly on the intersection of the subject and the hotspots."""


def build_reformulation_prompt(query: str, hotspot_tokens: List[str]) -> str:
    """Render the Appendix-A reformulation prompt for ``query``/``hotspot_tokens``."""
    return DTR_REFORMULATION_PROMPT.format(
        query=query,
        hotspot_tokens=", ".join(hotspot_tokens),
    )


# ---------------------------------------------------------------------------
# Mode constants
# ---------------------------------------------------------------------------
class ReformulatorMode:
    """String constants for the supported reformulation modes."""

    IDENTITY = "identity"
    HEURISTIC = "heuristic"
    SLM = "slm"

    ALL = (IDENTITY, HEURISTIC, SLM)


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------
_TERM_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_+.\-]*")

_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "by", "for", "from", "how",
    "in", "of", "on", "or", "the", "to", "with",
}


def _clean_query(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip(" \t\n\r\"'")


def _unique_terms(terms: List[str], limit: int) -> List[str]:
    seen: set = set()
    unique: List[str] = []
    for term in terms:
        cleaned = _clean_query(term)
        key = cleaned.lower()
        if not cleaned or key in seen:
            continue
        seen.add(key)
        unique.append(cleaned)
        if len(unique) >= limit:
            break
    return unique


def _subject_terms(query: str, hotspot_tokens: List[str], limit: int) -> List[str]:
    hotspot_keys = {term.lower() for term in hotspot_tokens}
    terms: List[str] = []
    for term in _TERM_RE.findall(query):
        key = term.lower()
        if key in hotspot_keys or key in _STOPWORDS:
            continue
        terms.append(term)
        if len(terms) >= limit:
            break
    return terms


# ---------------------------------------------------------------------------
# Reformulator
# ---------------------------------------------------------------------------
@dataclass
class DeltaReformulator:
    """A callable that turns ``(query, hotspot_tokens)`` into a delta query.

    Instances are callable: ``reformulator(query, hotspot_tokens) -> q_diff``.

    Args:
        mode: One of :class:`ReformulatorMode` values.
        slm_model: Model name for the ``"slm"`` mode.
        slm_client: Optional pre-built OpenAI-compatible client (with
            ``.chat.completions.create``). When provided, it is preferred over
            constructing a local ``transformers`` model.
        max_terms: Max hotspot tokens kept by the heuristic mode.
        max_subject_terms: Max subject terms kept by the heuristic mode.
        max_new_tokens: Generation budget for the SLM mode.
        local_files_only: Whether the local transformers fallback may only use
            cached checkpoints (no network).
    """

    mode: str = ReformulatorMode.HEURISTIC
    slm_model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    slm_client: Optional[object] = None
    max_terms: int = 3
    max_subject_terms: int = 6
    max_new_tokens: int = 32
    local_files_only: bool = True

    _slm_failed: bool = field(default=False, init=False, repr=False)
    _hf_model: object = field(default=None, init=False, repr=False)
    _hf_tokenizer: object = field(default=None, init=False, repr=False)
    _hf_torch: object = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.mode not in ReformulatorMode.ALL:
            raise ValueError(
                f"unknown reformulator mode: {self.mode!r} "
                f"(expected one of {ReformulatorMode.ALL})"
            )

    # -- public callable interface -----------------------------------------
    def __call__(self, query: str, hotspot_tokens: Optional[List[str]] = None) -> str:
        return self.reformulate(query, hotspot_tokens or [])

    def reformulate(self, query: str, hotspot_tokens: List[str]) -> str:
        """Return a single incremental (delta) search query."""
        if self.mode == ReformulatorMode.IDENTITY:
            return query
        if not hotspot_tokens:
            return query
        if self.mode == ReformulatorMode.HEURISTIC:
            return self._heuristic(query, hotspot_tokens)
        # SLM mode with graceful degradation.
        if not self._slm_failed:
            slm_query = self._try_slm(query, hotspot_tokens)
            if slm_query is not None:
                return slm_query
            self._slm_failed = True
        return self._heuristic(query, hotspot_tokens)

    # -- heuristic mode -----------------------------------------------------
    def _heuristic(self, query: str, hotspot_tokens: List[str]) -> str:
        terms = _unique_terms(hotspot_tokens, self.max_terms)
        if not terms:
            return query
        subject_terms = _subject_terms(query, terms, self.max_subject_terms)
        focused = " ".join([*subject_terms, *terms])
        return _clean_query(focused) or query

    # -- slm mode -----------------------------------------------------------
    def _try_slm(self, query: str, hotspot_tokens: List[str]) -> Optional[str]:
        prompt = build_reformulation_prompt(query, hotspot_tokens)

        # Prefer an OpenAI-compatible client (lazy import only if needed).
        client = self.slm_client
        if client is not None:
            try:
                response = client.chat.completions.create(
                    model=self.slm_model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=self.max_new_tokens,
                    temperature=0.3,
                )
                text = response.choices[0].message.content
                return _first_line(text, query)
            except Exception as exc:
                logger.warning("SLM client reformulation failed: %s", exc)

        # Local transformers fallback (lazy heavy import).
        text = self._transformers_generate(prompt)
        if text is not None:
            return _first_line(text, query)
        return None

    def _transformers_generate(self, prompt: str) -> Optional[str]:
        if self._hf_model is None:
            try:
                import torch  # lazy heavy import
                from transformers import AutoModelForCausalLM, AutoTokenizer
            except Exception:
                logger.warning(
                    "transformers/torch unavailable; SLM reformulator falls back"
                )
                return None
            try:
                self._hf_torch = torch
                self._hf_tokenizer = AutoTokenizer.from_pretrained(
                    self.slm_model, local_files_only=self.local_files_only
                )
                self._hf_model = AutoModelForCausalLM.from_pretrained(
                    self.slm_model, local_files_only=self.local_files_only
                )
                # switch to inference mode
                self._hf_model.train(False)
            except Exception as exc:  # pragma: no cover - load failures
                logger.warning("failed to load SLM %s: %s", self.slm_model, exc)
                self._hf_model = None
                return None

        tokenizer = self._hf_tokenizer
        model = self._hf_model
        torch = self._hf_torch
        try:
            if hasattr(tokenizer, "apply_chat_template"):
                rendered = tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            else:
                rendered = prompt
            inputs = tokenizer(rendered, return_tensors="pt")
            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            new_tokens = output[0][inputs["input_ids"].shape[-1]:]
            return tokenizer.decode(new_tokens, skip_special_tokens=True)
        except Exception as exc:  # pragma: no cover - generation failures
            logger.warning("SLM generation failed: %s", exc)
            return None


def _first_line(text: Optional[str], fallback: str) -> str:
    if not text or not text.strip():
        return fallback
    return _clean_query(text.strip().splitlines()[0])


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def build_reformulator(
    mode: str = ReformulatorMode.HEURISTIC,
    *,
    slm_model: str = "Qwen/Qwen2.5-0.5B-Instruct",
    slm_client: Optional[object] = None,
    max_terms: int = 3,
    max_subject_terms: int = 6,
    max_new_tokens: int = 32,
    local_files_only: bool = True,
) -> DeltaReformulator:
    """Build a callable DTR reformulator ``(query, hotspot_tokens) -> q_diff``.

    Args:
        mode: ``"identity"``, ``"heuristic"``, or ``"slm"``.
        slm_model: Small (<1B) model name used by the ``"slm"`` mode.
        slm_client: Optional OpenAI-compatible client preferred by ``"slm"``.
        max_terms: Hotspot-token cap for the heuristic reformulation.
        max_subject_terms: Subject-term cap for the heuristic reformulation.
        max_new_tokens: Generation budget for the ``"slm"`` mode.
        local_files_only: Restrict the local transformers fallback to cached
            checkpoints.

    Returns:
        A :class:`DeltaReformulator` (callable). All heavy dependencies are
        imported lazily, so this call is import-safe without them.
    """
    return DeltaReformulator(
        mode=mode,
        slm_model=slm_model,
        slm_client=slm_client,
        max_terms=max_terms,
        max_subject_terms=max_subject_terms,
        max_new_tokens=max_new_tokens,
        local_files_only=local_files_only,
    )
