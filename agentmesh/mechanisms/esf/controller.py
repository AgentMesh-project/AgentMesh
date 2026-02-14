"""
ESF (Elastic Semantic Flow)

Implements adaptive microbatch size control via Online Sensitivity
Estimation (OSE) to minimize end-to-end latency in semantic streaming.

ESF decomposes total latency:  T_total = L_start + L_work + L_stall
and controls microbatch size θ to minimize stall time L_stall.

Two pipelines served:
  1. Tool-to-agent pipeline — SRR items streamed to the agent's LLM prefill
  2. Agent-to-agent pipeline — worker decode → supervisor prefill

Key formulas:
  S_{pro,i} ≈ Δ(T_{p,i} + T_{n,i}) / Δθ   — Producer sensitivity
  S_{con,i} ≈ ΔT_{c,i} / Δθ               — Consumer sensitivity
  ρ_i      = S_{pro,i} - S_{con,i-1}       — Relative sensitivity
  Δθ_i     = -δ_i / ρ_i                    — Control law
  θ_{i+1}  = clip(θ_i + Δθ_i, θ_min, θ_max)

Variables:
  T_{p,i} — producing time (LLM generation / tool execution)
  T_{n,i} — network transfer time
  T_{c,i} — consuming time (downstream prefill / parsing)
  δ_i     — stall time (idle gap)
  θ_i     — microbatch size (tokens)

Operational regimes:
  ρ > 0 → Production-Bound: increase θ to amortize start-up cost
  ρ < 0 → Consumption-Bound: decrease θ to avoid consumer starvation
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Deque
from dataclasses import dataclass, field
from collections import deque
import logging
import time

logger = logging.getLogger(__name__)


@dataclass
class Observation:
    """
    Observation data point for ESF control loop.

    Captures timing information for a single microbatch transmission:
    - chunk_size (θ_i): Microbatch size in tokens
    - producer_time (T_{p,i} + T_{n,i}): Generation + network transfer time
    - consumer_time (T_{c,i}): Downstream consuming time (prefill / parsing)
    - slack_time (δ_i): Stall time (idle gap)
    """
    chunk_size: int          # θ_i — microbatch size (tokens)
    producer_time: float     # T_{p,i} + T_{n,i} — producing + network time
    consumer_time: float     # T_{c,i} — consuming time
    slack_time: float        # δ_i — stall time
    timestamp: float         # Observation timestamp
    step: int                # Step number in the control loop


@dataclass
class Sensitivity:
    """
    Sensitivity estimates for ESF control (OSE output).

    Computed from consecutive observations:
      S_{pro,i} ≈ Δ(T_p + T_n) / Δθ  — producer sensitivity
      S_{con,i} ≈ ΔT_c / Δθ          — consumer sensitivity
      ρ_i = S_{pro,i} - S_{con,i-1}  — relative sensitivity

    Note: ρ uses S_{con,i-1} (not i) due to pipeline overlap.
    """
    s_pro: float  # S_{pro,i} — producer sensitivity
    s_con: float  # S_{con,i} — consumer sensitivity
    rho: float    # ρ_i — relative sensitivity


class StateObserver:
    """
    Online Sensitivity Estimator (OSE).

    Estimates producer and consumer sensitivities via finite differences
    over a sliding window of observations.

    Key formulas:
      S_{pro,i} ≈ Δ(T_p + T_n) / Δθ
      S_{con,i} ≈ ΔT_c / Δθ
      ρ_i = S_{pro,i} - S_{con,i-1}

    Special case: when θ_i = θ_{i-1} (no size change), the estimator
    retains the previous sensitivity to avoid division by zero.
    """
    
    def __init__(self, window_size: int = 5):
        """
        Initialize the state observer.
        
        Args:
            window_size: Size of sliding window for sensitivity estimation.
        """
        self.window_size = window_size
        self.observations: Deque[Observation] = deque(maxlen=window_size)
        self.sensitivity_history: List[Sensitivity] = []
        
        # Statistics
        self.stats = {
            "total_observations": 0,
            "avg_slack": 0.0,
            "slack_variance": 0.0
        }
    
    def record_observation(self, obs: Observation) -> None:
        """
        Record a new observation.
        
        Args:
            obs: Observation data point.
        """
        self.observations.append(obs)
        self.stats["total_observations"] += 1
        self._update_slack_stats(obs.slack_time)
    
    def estimate_sensitivity(self) -> Optional[Sensitivity]:
        """
        Estimate current sensitivity values via OSE.

        S_{pro,i} ≈ Δ(T_p + T_n) / Δθ
        S_{con,i} ≈ ΔT_c / Δθ
        ρ_i = S_{pro,i} - S_{con,i-1}

        When θ_i = θ_{i-1} (Δθ ≈ 0), keeps previous sensitivity
        estimates unchanged.

        Returns:
            Sensitivity estimates, or None if insufficient data.
        """
        if len(self.observations) < 2:
            return None
        
        # Get most recent observations
        curr = self.observations[-1]
        prev = self.observations[-2]
        
        # Compute deltas
        delta_theta = curr.chunk_size - prev.chunk_size
        
        # Handle zero delta (no chunk size change)
        if abs(delta_theta) < 1e-6:
            if self.sensitivity_history:
                # Use previous estimates
                last = self.sensitivity_history[-1]
                return Sensitivity(
                    s_pro=last.s_pro,
                    s_con=last.s_con,
                    rho=last.rho
                )
            return Sensitivity(s_pro=0.0, s_con=0.0, rho=0.0)
        
        # Producer sensitivity
        delta_producer = curr.producer_time - prev.producer_time
        s_pro = delta_producer / delta_theta
        
        # Consumer sensitivity
        delta_consumer = curr.consumer_time - prev.consumer_time
        s_con = delta_consumer / delta_theta
        
        # Relative sensitivity
        # Use previous s_con for ρ calculation as per paper
        if self.sensitivity_history:
            rho = s_pro - self.sensitivity_history[-1].s_con
        else:
            rho = s_pro - s_con
        
        sensitivity = Sensitivity(s_pro=s_pro, s_con=s_con, rho=rho)
        self.sensitivity_history.append(sensitivity)
        
        return sensitivity
    
    def _update_slack_stats(self, slack: float) -> None:
        """Update running statistics for slack time using Welford's algorithm."""
        n = self.stats["total_observations"]
        
        if n == 0:
            self.stats["avg_slack"] = slack
            self.stats["slack_variance"] = 0.0
            return
        
        old_avg = self.stats["avg_slack"]
        new_avg = old_avg + (slack - old_avg) / n
        
        if n > 1:
            old_var = self.stats["slack_variance"]
            new_var = old_var + (slack - old_avg) * (slack - new_avg)
            self.stats["slack_variance"] = new_var / (n - 1)
        
        self.stats["avg_slack"] = new_avg
    
    def get_current_state(self) -> Optional[Dict]:
        """Get current observer state."""
        if not self.observations:
            return None
        
        latest = self.observations[-1]
        sensitivity = self.estimate_sensitivity()
        
        state = {
            "chunk_size": latest.chunk_size,
            "producer_time": latest.producer_time,
            "consumer_time": latest.consumer_time,
            "slack_time": latest.slack_time,
            "step": latest.step,
            "window_size": len(self.observations),
            "stats": self.stats.copy()
        }
        
        if sensitivity:
            state["sensitivity"] = {
                "s_pro": sensitivity.s_pro,
                "s_con": sensitivity.s_con,
                "rho": sensitivity.rho
            }
            # Determine bound type
            if sensitivity.rho > 0:
                state["bound_type"] = "producer_bound"
            elif sensitivity.rho < 0:
                state["bound_type"] = "consumer_bound"
            else:
                state["bound_type"] = "balanced"
        else:
            state["sensitivity"] = None
            state["bound_type"] = None
        
        return state
    
    def clear(self) -> None:
        """Reset observer state."""
        self.observations.clear()
        self.sensitivity_history.clear()
        self.stats = {
            "total_observations": 0,
            "avg_slack": 0.0,
            "slack_variance": 0.0
        }


class ESFController:
    """
    Elastic Semantic Flow Controller.

    Core control algorithm:
    1. Online Sensitivity Estimation (OSE) via StateObserver
    2. Polarized control based on ρ sign:
       - ρ > 0 → Production-Bound → increase θ (larger microbatches)
       - ρ < 0 → Consumption-Bound → decrease θ (smaller microbatches)
    3. Bounded adaptive microbatch size adjustment

    Control law:
      Δθ_i     = -δ_i / ρ_i
      θ_{i+1}  = clip(θ_i + Δθ_i, θ_min, θ_max)
    """
    
    def __init__(
        self,
        initial_theta: int = 512,
        min_theta: int = 64,
        max_theta: int = 4096,
        damping_factor: float = 1.0,
        epsilon: float = 1e-6,
        window_size: int = 5
    ):
        """
        Initialize ESF controller.
        
        Args:
            initial_theta: Initial microbatch size in tokens.
            min_theta: Minimum allowed microbatch size (θ_min).
            max_theta: Maximum allowed microbatch size (θ_max).
            damping_factor: Smoothing factor for adjustments (1.0 = no damping).
            epsilon: Small constant to prevent division by zero.
            window_size: Window size for OSE sensitivity estimation.
        """
        # Control parameters
        self.initial_theta = initial_theta
        self.current_theta = initial_theta
        self.min_theta = min_theta
        self.max_theta = max_theta
        self.damping_factor = damping_factor
        self.epsilon = epsilon
        
        # Chunk size tracking
        self.chunk_size: List[Tuple[int, int]] = [(initial_theta, 0)]
        self.chunk_pointer: int = 0
        
        # State observer
        self.observer = StateObserver(window_size=window_size)
        
        # Control state
        self.step_count = 0
        self.adjustment_count = 0
        self.stable_steps = 0
        
        # History
        self.theta_history: List[int] = []
        self.slack_history: List[float] = []
        self.adjustment_history: List[Dict] = []
        self.sensitivity_history: List[Dict] = []
        
        # Statistics
        self.stats = {
            "total_steps": 0,
            "total_adjustments": 0,
            "starvation_steps": 0,    # δ > 0 (consumer waiting)
            "saturation_steps": 0,    # δ < 0 (producer waiting)
            "balanced_steps": 0,      # |δ| < epsilon
            "producer_bound_steps": 0,  # ρ > 0
            "consumer_bound_steps": 0,  # ρ < 0
            "avg_theta": initial_theta,
            "avg_slack": 0.0,
            "slack_variance": 0.0
        }
    
    def compute_next_theta(
        self,
        producer_time: float,
        consumer_time: float,
        slack_time: float,
        time_step: int = 0
    ) -> int:
        """
        Compute optimal microbatch size for next iteration.

        Implements control loop:
          1. Record observation (T_p+T_n, T_c, δ)
          2. OSE: estimate S_pro, S_con, ρ
          3. Compute Δθ = -δ/ρ
          4. Clip: θ_{i+1} = clip(θ_i + Δθ, θ_min, θ_max)

        Args:
            producer_time: T_{p,i} + T_{n,i} (generation + network).
            consumer_time: T_{c,i} (consuming time).
            slack_time: δ_i (stall time).
            time_step: Current time step for scheduling.

        Returns:
            Recommended microbatch size θ for next iteration.
        """
        self.step_count += 1
        old_theta = self.chunk_size[-1][0]
        
        # 1. Record observation
        obs = Observation(
            chunk_size=old_theta,
            producer_time=producer_time,
            consumer_time=consumer_time,
            slack_time=slack_time,
            timestamp=time.time(),
            step=self.step_count
        )
        self.observer.record_observation(obs)
        
        # 2. Update history
        self.theta_history.append(old_theta)
        self.slack_history.append(slack_time)
        
        # 3. Update statistics
        self._update_stats(slack_time, old_theta)
        
        # 4. Estimate sensitivity
        sensitivity = self.observer.estimate_sensitivity()
        
        # 5. Compute adjustment
        if sensitivity and abs(sensitivity.rho) > self.epsilon:
            delta_theta = self._compute_adjustment(old_theta, slack_time, sensitivity)
            adjustment_reason = "sensitivity_based"
            
            # Log sensitivity
            self.sensitivity_history.append({
                "step": self.step_count,
                "s_pro": sensitivity.s_pro,
                "s_con": sensitivity.s_con,
                "rho": sensitivity.rho,
                "bound_type": "producer" if sensitivity.rho > 0 else "consumer"
            })
        else:
            delta_theta = self._heuristic_adjustment(old_theta, slack_time)
            adjustment_reason = "heuristic"
        
        # 6. Apply damping
        if self.damping_factor < 1.0:
            delta_theta = int(round(self.damping_factor * delta_theta))
        
        # 7. Clamp to bounds (paper clamping formula)
        if delta_theta > 0:
            delta_theta = min(delta_theta, self.max_theta - old_theta)
        elif delta_theta < 0:
            delta_theta = max(delta_theta, self.min_theta - old_theta)
        
        new_theta = old_theta + delta_theta
        
        # 8. Check if adjustment is significant
        if abs(delta_theta) < self.min_theta * 0.05:
            new_theta = old_theta
            adjustment_reason = "insignificant"
            self.stable_steps += 1
        else:
            self.adjustment_count += 1
            self.stable_steps = 0
        
        # 9. Record adjustment
        self.adjustment_history.append({
            "step": self.step_count,
            "old_theta": old_theta,
            "new_theta": new_theta,
            "delta_theta": new_theta - old_theta,
            "reason": adjustment_reason,
            "slack_time": slack_time,
            "producer_time": producer_time,
            "consumer_time": consumer_time,
            **({"s_pro": sensitivity.s_pro, "s_con": sensitivity.s_con, "rho": sensitivity.rho}
               if sensitivity else {})
        })
        
        self.chunk_size.append((new_theta, time_step))
        logger.debug(f"ESF: θ {old_theta} → {new_theta} (δ={slack_time:.2f}, reason={adjustment_reason})")
        
        return new_theta
    
    def _compute_adjustment(
        self,
        current_theta: int,
        slack_time: float,
        sensitivity: Sensitivity
    ) -> int:
        """
        Compute adjustment using sensitivity estimates.

        Δθ_i = -δ_i / ρ_i

        Operational regimes:
          ρ > 0 → Production-Bound: δ > 0 → Δθ < 0 (shrink microbatch)
          ρ < 0 → Consumption-Bound: δ < 0 → Δθ < 0 (shrink microbatch)

        Args:
            current_theta: Current microbatch size.
            slack_time: Current stall time δ_i.
            sensitivity: Current sensitivity estimates.

        Returns:
            Recommended Δθ.
        """
        rho = sensitivity.rho
        
        delta_theta_raw = -slack_time / rho
        
        # Update bound statistics
        if rho > 0:
            self.stats["producer_bound_steps"] += 1
        else:
            self.stats["consumer_bound_steps"] += 1
        
        return int(round(delta_theta_raw))
    
    def _heuristic_adjustment(
        self,
        current_theta: int,
        slack_time: float
    ) -> int:
        """
        Heuristic adjustment when sensitivity estimates unavailable.
        
        Uses simple proportional control based on slack time magnitude.
        
        Args:
            current_theta: Current chunk size.
            slack_time: Current slack time.
            
        Returns:
            Heuristic Δθ.
        """
        if self.step_count <= 1:
            return 0  # Not enough data
        
        # Proportional adjustment based on slack severity
        if slack_time > 50:  # Severe starvation
            return int(-current_theta * 0.3)
        elif slack_time > 10:  # Mild starvation
            return int(-current_theta * 0.15)
        elif slack_time < -50:  # Severe saturation
            return int(current_theta * 0.3)
        elif slack_time < -10:  # Mild saturation
            return int(current_theta * 0.15)
        else:
            # Near balanced — hold steady
            return 0
    
    def _update_stats(self, slack_time: float, theta: int) -> None:
        """Update running statistics using Welford's algorithm."""
        n = self.stats["total_steps"] + 1
        self.stats["total_steps"] = n
        
        # Update theta average
        old_avg_theta = self.stats["avg_theta"]
        self.stats["avg_theta"] = old_avg_theta + (theta - old_avg_theta) / n
        
        # Update slack average and variance
        old_avg_slack = self.stats["avg_slack"]
        self.stats["avg_slack"] = old_avg_slack + (slack_time - old_avg_slack) / n
        
        if n > 1:
            old_var = self.stats.get("slack_variance", 0.0)
            new_var = old_var + (slack_time - old_avg_slack) * (slack_time - self.stats["avg_slack"])
            self.stats["slack_variance"] = new_var / (n - 1)
        
        # Update state distribution
        if abs(slack_time) < self.epsilon:
            self.stats["balanced_steps"] += 1
        elif slack_time > 0:
            self.stats["starvation_steps"] += 1
        else:
            self.stats["saturation_steps"] += 1
    
    def update_current_theta(self, time_step: int) -> None:
        """
        Update current_theta based on time step.
        
        Finds the appropriate θ value for the given time step
        from the chunk size history.
        
        Args:
            time_step: Current time step.
        """
        while self.chunk_pointer < len(self.chunk_size):
            if time_step >= self.chunk_size[self.chunk_pointer][1]:
                self.current_theta = self.chunk_size[self.chunk_pointer][0]
                self.chunk_pointer += 1
            else:
                break
    
    def get_state(self) -> Dict:
        """Get comprehensive controller state."""
        observer_state = self.observer.get_current_state()
        recent_perf = self._calculate_recent_performance()
        
        state = {
            "current_theta": self.current_theta,
            "step": self.step_count,
            "adjustment_count": self.adjustment_count,
            "stable_steps": self.stable_steps,
            "stats": self.stats.copy(),
            "observer": observer_state,
            "recent_performance": recent_perf,
            "config": {
                "initial_theta": self.initial_theta,
                "min_theta": self.min_theta,
                "max_theta": self.max_theta,
                "damping_factor": self.damping_factor
            }
        }
        
        if self.theta_history:
            state["history_summary"] = {
                "theta_range": (min(self.theta_history), max(self.theta_history)),
                "recent_thetas": self.theta_history[-5:],
                "recent_slacks": self.slack_history[-5:],
                "total_tokens": sum(self.theta_history)
            }
        
        return state
    
    def _calculate_recent_performance(self, window: int = 10) -> Dict:
        """Calculate performance metrics over recent history."""
        if len(self.adjustment_history) < 2:
            return {}
        
        recent = self.adjustment_history[-window:]
        
        avg_slack = np.mean([h["slack_time"] for h in recent])
        adj_freq = sum(1 for h in recent if h["delta_theta"] != 0) / len(recent)
        
        total_tokens = sum(h["old_theta"] for h in recent)
        total_time = sum(
            h["producer_time"] + h["consumer_time"] + max(0, h["slack_time"])
            for h in recent
        )
        efficiency = total_tokens / total_time if total_time > 0 else 0
        
        return {
            "avg_slack": avg_slack,
            "adjustment_frequency": adj_freq,
            "efficiency": efficiency,
            "sample_size": len(recent)
        }
    
    def reset(self, initial_theta: Optional[int] = None) -> None:
        """Reset controller to initial state."""
        if initial_theta is not None:
            self.initial_theta = initial_theta
        
        self.current_theta = self.initial_theta
        self.chunk_size = [(self.initial_theta, 0)]
        self.chunk_pointer = 0
        
        self.observer.clear()
        
        self.step_count = 0
        self.adjustment_count = 0
        self.stable_steps = 0
        
        self.theta_history.clear()
        self.slack_history.clear()
        self.adjustment_history.clear()
        self.sensitivity_history.clear()
        
        self.stats = {
            "total_steps": 0,
            "total_adjustments": 0,
            "starvation_steps": 0,
            "saturation_steps": 0,
            "balanced_steps": 0,
            "producer_bound_steps": 0,
            "consumer_bound_steps": 0,
            "avg_theta": self.initial_theta,
            "avg_slack": 0.0,
            "slack_variance": 0.0
        }


# Convenience function
def create_esf_controller(
    initial_theta: int = 512,
    min_theta: int = 64,
    max_theta: int = 4096,
    damping_factor: float = 1.0,
    window_size: int = 5
) -> ESFController:
    """
    Create an ESF controller instance.
    
    Args:
        initial_theta: Initial chunk size.
        min_theta: Minimum chunk size.
        max_theta: Maximum chunk size.
        damping_factor: Adjustment smoothing factor.
        window_size: Sensitivity estimation window.
        
    Returns:
        Configured ESFController instance.
    """
    return ESFController(
        initial_theta=initial_theta,
        min_theta=min_theta,
        max_theta=max_theta,
        damping_factor=damping_factor,
        window_size=window_size
    )
