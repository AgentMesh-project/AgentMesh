"""
DES Offline Performance Profiler.

``PerformancePredictor`` loads fitted latency coefficients from a JSON file and
predicts prefill / decode / total latency for a given
``(model_gpu, in_len, out_len)`` triple. The fits support three functional
forms per stage:

  * ``linear``     : y = a * x + b
  * ``quadratic``  : y = a * x^2 + b * x + c
  * ``power``      : y = a * x^b + c

These curves are produced by an offline profiling sweep (varying input/output
lengths on a target model-GPU pair and least-squares-fitting prefill and decode
latency). DES uses them *offline* to derive sane chunk-size bounds
``theta_min`` / ``theta_max`` for a deployment: the smallest chunk worth
streaming (so per-chunk overhead does not dominate) and the largest chunk that
keeps a single produce step from creating an unacceptable consumer barrier.

The fitted-coefficients JSON schema (one entry per ``model_gpu`` key)::

    {
      "ExampleModel-8B_ExampleGPU": {
        "prefill": {"model_type": "linear",    "params": [a, b]},
        "decode":  {"model_type": "linear",    "params": [a, b]}
      },
      ...
    }

Latencies are in milliseconds. ``params`` lengths: linear -> 2, quadratic -> 3,
power -> 3.

Anonymity: no real host paths are baked in. With no path argument the predictor
loads ``benchmarks/data/model_latency_fits.example.json`` if that optional
bundle is present, otherwise it falls back to the built-in illustrative fits
(``_EXAMPLE_FITS``); callers may pass any coefficients JSON explicitly.

Pure Python + numpy. No heavy dependencies.
"""

from __future__ import annotations

import json
import math
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

__all__ = [
    "PerformancePredictor",
    "default_fits_path",
    "DEFAULT_FITS_FILENAME",
]

DEFAULT_FITS_FILENAME = "model_latency_fits.example.json"

# Built-in illustrative fits, used when no path is given and the optional bundled
# example file is absent. These are NOT measured hardware numbers — illustrative
# coefficients (latency in ms) so the predictor is usable out of the box. Supply
# a real coefficients JSON for accurate, hardware-specific predictions.
_EXAMPLE_FITS: Dict[str, Dict] = {
    "_comment": "Illustrative example fits (ms); not measured hardware data.",
    "ExampleModel-8B_ExampleGPU": {
        "prefill": {"model_type": "linear", "params": [0.018, 12.0]},
        "decode": {"model_type": "linear", "params": [0.30, 8.0]},
    },
    "ExampleModel-30B_ExampleGPU": {
        "prefill": {"model_type": "quadratic", "params": [3.0e-6, 0.045, 25.0]},
        "decode": {"model_type": "linear", "params": [0.55, 14.0]},
    },
}


def default_fits_path() -> str:
    """
    Resolve the path to the bundled example coefficients JSON.

    Walks up from this file to the repository root (the directory that
    contains ``benchmarks/``) and returns
    ``benchmarks/data/<DEFAULT_FITS_FILENAME>``. Falls back to a path relative
    to the package if the layout is unexpected.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    # .../agentmesh/mechanisms/des -> walk up to the repo root.
    cur = here
    for _ in range(6):
        candidate = os.path.join(cur, "benchmarks", "data", DEFAULT_FITS_FILENAME)
        if os.path.exists(candidate):
            return candidate
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent
    # Best-effort default (may not exist; PerformancePredictor will raise).
    repo_root = os.path.abspath(os.path.join(here, "..", "..", ".."))
    return os.path.join(repo_root, "benchmarks", "data", DEFAULT_FITS_FILENAME)


def _eval_fit(model_type: str, params: List[float], x: float) -> float:
    """Evaluate a single fitted curve at ``x`` (token count)."""
    p = np.asarray(params, dtype=float)
    if model_type == "linear":
        # y = a*x + b
        return float(p[0] * x + p[1])
    if model_type == "quadratic":
        # y = a*x^2 + b*x + c
        return float(p[0] * (x ** 2) + p[1] * x + p[2])
    if model_type == "power":
        # y = a*x^b + c
        return float(p[0] * (x ** p[1]) + p[2])
    raise ValueError(f"Unsupported fit model_type: {model_type!r}")


class PerformancePredictor:
    """
    Predict prefill / decode / total latency from offline-fitted curves.

    Args:
        fitting_results_path: Path to the coefficients JSON. Defaults to the
            bundled example bundle (see :func:`default_fits_path`).
        verbose: If True, prints a one-line load summary.
    """

    def __init__(
        self,
        fitting_results_path: Optional[str] = None,
        verbose: bool = False,
    ):
        raw: Dict[str, Dict]
        if fitting_results_path is None:
            bundled = default_fits_path()
            if os.path.exists(bundled):
                self.fitting_results_path = bundled
                with open(bundled, "r", encoding="utf-8") as f:
                    raw = json.load(f)
            else:
                # Optional bundle absent (e.g. a trimmed repo): use built-in fits
                # so the predictor is usable without any external file.
                self.fitting_results_path = "<built-in example fits>"
                raw = dict(_EXAMPLE_FITS)
        else:
            if not os.path.exists(fitting_results_path):
                raise FileNotFoundError(
                    f"Latency-fit coefficients not found at "
                    f"{fitting_results_path!r}. Pass an existing fitting_results_path "
                    f"(or omit it to use the built-in example fits)."
                )
            self.fitting_results_path = fitting_results_path
            with open(fitting_results_path, "r", encoding="utf-8") as f:
                raw = json.load(f)

        # Keep only well-formed entries: a dict carrying both "prefill" and
        # "decode" fits. This skips metadata keys such as "_comment".
        self.fitting_results: Dict[str, Dict] = {
            k: v
            for k, v in raw.items()
            if isinstance(v, dict) and "prefill" in v and "decode" in v
        }
        self.supported_combinations: List[str] = list(self.fitting_results.keys())
        if verbose:
            print(
                f"Loaded latency fits for {len(self.supported_combinations)} "
                f"model-GPU combinations from {self.fitting_results_path}"
            )

    # ------------------------------------------------------------------ #
    def get_supported_combinations(self) -> List[str]:
        """Return the list of available ``model_gpu`` keys."""
        return list(self.supported_combinations)

    def _stage_fit(self, model_gpu: str, stage: str) -> Tuple[str, List[float]]:
        if model_gpu not in self.fitting_results:
            raise ValueError(
                f"Unsupported model-GPU combination: {model_gpu!r}. "
                f"Available: {self.supported_combinations}"
            )
        info = self.fitting_results[model_gpu][stage]
        return info["model_type"], info["params"]

    def predict_prefill_latency(self, model_gpu: str, in_len: int) -> float:
        """Predict prefill latency (ms) for ``in_len`` input tokens."""
        model_type, params = self._stage_fit(model_gpu, "prefill")
        return _eval_fit(model_type, params, float(in_len))

    def predict_decode_latency(self, model_gpu: str, out_len: int) -> float:
        """Predict decode latency (ms) for ``out_len`` output tokens."""
        model_type, params = self._stage_fit(model_gpu, "decode")
        return _eval_fit(model_type, params, float(out_len))

    def predict_total_latency(self, model_gpu: str, in_len: int, out_len: int) -> float:
        """Predict total latency (ms) = prefill(in_len) + decode(out_len)."""
        return self.predict_prefill_latency(model_gpu, in_len) + \
            self.predict_decode_latency(model_gpu, out_len)

    def compare_combinations(self, in_len: int, out_len: int) -> List[Dict]:
        """
        Compare all combinations at the given (in_len, out_len), sorted by
        total latency ascending.
        """
        results = []
        for combo in self.supported_combinations:
            prefill = self.predict_prefill_latency(combo, in_len)
            decode = self.predict_decode_latency(combo, out_len)
            model, _, gpu = combo.partition("_")
            results.append(
                {
                    "model_gpu": combo,
                    "model": model,
                    "gpu": gpu,
                    "prefill_latency_ms": prefill,
                    "decode_latency_ms": decode,
                    "total_latency_ms": prefill + decode,
                }
            )
        return sorted(results, key=lambda r: r["total_latency_ms"])

    # ------------------------------------------------------------------ #
    # DES theta-bound derivation
    # ------------------------------------------------------------------ #
    def derive_theta_bounds(
        self,
        model_gpu: str,
        overhead_fraction: float = 0.1,
        max_chunk_prefill_ms: float = 200.0,
        theta_floor: int = 64,
        theta_ceiling: int = 4096,
        probe_in_len: int = 512,
    ) -> Dict[str, int]:
        """
        Derive offline ``theta_min`` / ``theta_max`` bounds for DES.

        Rationale:
          * ``theta_min`` is the smallest chunk whose prefill latency exceeds a
            target multiple of the per-call fixed overhead, so streaming many
            tiny chunks does not let fixed cost dominate. We approximate the
            fixed overhead by the prefill intercept (latency at zero tokens)
            and require the *marginal* token cost over a chunk to clear an
            ``overhead_fraction`` share of it.
          * ``theta_max`` is the largest chunk whose single-chunk prefill stays
            under ``max_chunk_prefill_ms`` (bounding the consumer barrier a
            single produce step can create).

        Both are clamped to ``[theta_floor, theta_ceiling]`` and quantized to a
        multiple of 8 (a friendly KV-block granularity). Returns a dict with
        ``theta_min``, ``theta_max``, and diagnostic fields.
        """
        model_type, params = self._stage_fit(model_gpu, "prefill")

        # Fixed overhead ~ prefill latency at (near) zero tokens.
        base_overhead = max(0.0, _eval_fit(model_type, params, 1.0))

        # --- theta_min: marginal cost over the chunk clears the overhead share.
        target_marginal = max(1e-6, overhead_fraction * max(base_overhead, 1.0))
        theta_min = theta_floor
        for theta in range(theta_floor, theta_ceiling + 1, 8):
            marginal = (
                _eval_fit(model_type, params, float(theta))
                - _eval_fit(model_type, params, 1.0)
            )
            if marginal >= target_marginal:
                theta_min = theta
                break
        else:
            theta_min = theta_floor

        # --- theta_max: largest chunk under the per-chunk prefill ceiling.
        theta_max = theta_floor
        for theta in range(theta_floor, theta_ceiling + 1, 8):
            if _eval_fit(model_type, params, float(theta)) <= max_chunk_prefill_ms:
                theta_max = theta
            else:
                break

        # Quantize / clamp / order.
        theta_min = int(np.clip(theta_min, theta_floor, theta_ceiling))
        theta_max = int(np.clip(theta_max, theta_floor, theta_ceiling))
        theta_min = (theta_min // 8) * 8 or theta_floor
        theta_max = (theta_max // 8) * 8 or theta_floor
        if theta_max < theta_min:
            theta_max = theta_min

        return {
            "theta_min": theta_min,
            "theta_max": theta_max,
            "base_overhead_ms": base_overhead,
            "max_chunk_prefill_ms": max_chunk_prefill_ms,
            "probe_prefill_ms": self.predict_prefill_latency(model_gpu, probe_in_len),
        }
