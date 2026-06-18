"""
DES (Dynamic-Equilibrium Streaming).

Adaptive streaming chunk-size control via Online Sensitivity Estimation (OSE).
DES drives the pipeline to a dynamic equilibrium (slack δ→0), overlapping the
produce/transfer/consume stages of a dataflow.
"""

from agentmesh.mechanisms.des.controller import (
    Observation,
    Sensitivity,
    OnlineSensitivityEstimator,
    DESController,
    create_des_controller,
)

__all__ = [
    "Observation",
    "Sensitivity",
    "OnlineSensitivityEstimator",
    "DESController",
    "create_des_controller",
]

# Sibling-contributed simulator (simulator.py) is added by another agent.
# Guard the import so the package keeps importing before that file exists.
try:  # pragma: no cover - optional sibling module
    from agentmesh.mechanisms.des.simulator import *  # noqa: F401,F403
except Exception:
    pass
