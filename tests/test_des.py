"""
Tests for DES (Dynamic-Equilibrium Streaming) — stage overlap.

DES overlaps the produce/transfer/consume stages of a dataflow by driving the
streaming slack δ→0. An Online Sensitivity Estimator (OSE) provides
finite-difference producer/consumer sensitivities; the controller takes a
Newton step Δθ = −δ/ρ (ρ = Ŝ_pro − Ŝ_con) and clips θ to [θ_min, θ_max].

Covered:
  - OSE finite-difference sensitivity (S_pro, S_con, ρ)
  - Δθ = −δ/ρ sign/direction across the four (sign δ, sign ρ) regimes
  - clipping to [θ_min, θ_max]
  - equilibrium behavior (δ≈0 holds θ steady)
"""

import numpy as np
import pytest

from agentmesh.mechanisms.des import (
    DESController,
    create_des_controller,
    OnlineSensitivityEstimator,
    Observation,
    Sensitivity,
)


# ---------------------------------------------------------------------------
# OSE finite-difference sensitivity
# ---------------------------------------------------------------------------
def _obs(theta, prod, cons, slack, step):
    return Observation(
        chunk_size=theta,
        producer_time=prod,
        consumer_time=cons,
        slack_time=slack,
        timestamp=0.0,
        step=step,
    )


def test_ose_needs_two_observations():
    ose = OnlineSensitivityEstimator(window_size=5)
    assert ose.estimate_sensitivity() is None
    ose.record_observation(_obs(100, 1.0, 0.5, 0.5, 1))
    assert ose.estimate_sensitivity() is None


def test_ose_finite_difference_values():
    """S_pro = Δproducer/Δθ, S_con = Δconsumer/Δθ."""
    ose = OnlineSensitivityEstimator(window_size=5)
    ose.record_observation(_obs(100, 1.0, 0.5, 0.5, 1))
    ose.record_observation(_obs(200, 3.0, 1.5, 1.5, 2))  # Δθ=100, Δprod=2, Δcons=1
    sens = ose.estimate_sensitivity()
    assert sens is not None
    assert np.isclose(sens.s_pro, 2.0 / 100.0)
    assert np.isclose(sens.s_con, 1.0 / 100.0)


def test_ose_zero_delta_theta_retains_previous():
    """θ unchanged (Δθ≈0) → retain previous sensitivity rather than divide by 0."""
    ose = OnlineSensitivityEstimator(window_size=5)
    ose.record_observation(_obs(100, 1.0, 0.5, 0.5, 1))
    ose.record_observation(_obs(200, 3.0, 1.5, 1.5, 2))
    first = ose.estimate_sensitivity()
    ose.record_observation(_obs(200, 4.0, 2.0, 2.0, 3))  # same θ → Δθ=0
    second = ose.estimate_sensitivity()
    assert np.isclose(second.s_pro, first.s_pro)
    assert np.isclose(second.s_con, first.s_con)


def test_ose_slack_statistics():
    ose = OnlineSensitivityEstimator(window_size=10)
    for i, s in enumerate([1.0, 2.0, 3.0]):
        ose.record_observation(_obs(100, 1.0, 0.5, s, i + 1))
    assert ose.stats["total_observations"] == 3
    assert np.isclose(ose.stats["avg_slack"], 2.0)


# ---------------------------------------------------------------------------
# Newton step Δθ = −δ/ρ — sign/direction across the four regimes
# ---------------------------------------------------------------------------
# The control law is Δθ = −δ/ρ. With ρ and δ each ±, there are four regimes.
# We exercise the law directly through ``_compute_adjustment`` (the routine the
# controller uses once OSE has a sensitivity estimate), which reads
# ``sensitivity.rho`` and ``slack_time`` and returns Δθ = round(−δ/ρ).
@pytest.mark.parametrize(
    "rho,slack,expect_sign",
    [
        (0.01, 50.0, -1),    # ρ>0, δ>0 → Δθ<0 (shrink): production-bound starvation
        (-0.01, -50.0, -1),  # ρ<0, δ<0 → Δθ<0 (shrink): consumption-bound saturation
        (0.01, -50.0, +1),   # ρ>0, δ<0 → Δθ>0 (grow)
        (-0.01, 50.0, +1),   # ρ<0, δ>0 → Δθ>0 (grow)
    ],
)
def test_newton_step_sign_all_four_regimes(rho, slack, expect_sign):
    ctrl = create_des_controller(initial_theta=512)
    sens = Sensitivity(s_pro=rho, s_con=0.0, rho=rho)
    delta = ctrl._compute_adjustment(current_theta=512, slack_time=slack, sensitivity=sens)
    assert delta == int(round(-slack / rho))
    if expect_sign < 0:
        assert delta < 0
    else:
        assert delta > 0


def test_compute_adjustment_formula_exact():
    """Δθ = round(−δ/ρ): −2.0 / 0.01 = −200."""
    ctrl = create_des_controller(initial_theta=512)
    sens = Sensitivity(s_pro=0.02, s_con=0.01, rho=0.01)  # ρ>0
    delta = ctrl._compute_adjustment(current_theta=512, slack_time=2.0, sensitivity=sens)
    assert delta == -200
    assert delta < 0


def test_controller_moves_theta_under_sustained_slack():
    """End-to-end: sustained nonzero slack drives θ away from its start."""
    ctrl = create_des_controller(initial_theta=1000, min_theta=64, max_theta=4096)
    start = ctrl.chunk_size[0][0]
    # Producer time grows with θ while consumer stays flat → s_pro>0, ρ>0;
    # persistent positive slack → repeated shrink steps.
    for step in range(8):
        ctrl.compute_next_theta(
            producer_time=1.0 + 0.01 * ctrl.chunk_size[-1][0],
            consumer_time=0.5,
            slack_time=20.0,
            time_step=step,
        )
    assert ctrl.chunk_size[-1][0] != start


# ---------------------------------------------------------------------------
# Clipping to [θ_min, θ_max]
# ---------------------------------------------------------------------------
def test_theta_clipped_to_max():
    ctrl = create_des_controller(initial_theta=4000, min_theta=64, max_theta=4096)
    theta = ctrl.current_theta
    for step in range(20):
        # ρ>0 with large negative slack → strong push to grow θ.
        ctrl.compute_next_theta(
            producer_time=10.0 + step, consumer_time=1.0, slack_time=-1000.0, time_step=step
        )
    for t in ctrl.theta_history:
        assert 64 <= t <= 4096
    assert all(t <= 4096 for t, _ in ctrl.chunk_size)


def test_theta_clipped_to_min():
    ctrl = create_des_controller(initial_theta=128, min_theta=64, max_theta=4096)
    for step in range(20):
        # ρ>0 with large positive slack → strong push to shrink θ.
        ctrl.compute_next_theta(
            producer_time=10.0 + step, consumer_time=1.0, slack_time=1000.0, time_step=step
        )
    assert all(t >= 64 for t, _ in ctrl.chunk_size)


def test_bounds_never_violated_under_random_drive():
    rng = np.random.default_rng(0)
    ctrl = create_des_controller(initial_theta=512, min_theta=64, max_theta=4096)
    for step in range(100):
        ctrl.compute_next_theta(
            producer_time=float(rng.uniform(0.1, 5.0)),
            consumer_time=float(rng.uniform(0.1, 5.0)),
            slack_time=float(rng.uniform(-100.0, 100.0)),
            time_step=step,
        )
    assert all(64 <= t <= 4096 for t, _ in ctrl.chunk_size)


# ---------------------------------------------------------------------------
# Equilibrium behavior
# ---------------------------------------------------------------------------
def test_equilibrium_holds_theta_steady():
    """At equilibrium (δ≈0) the controller should not move θ significantly."""
    ctrl = create_des_controller(initial_theta=512, min_theta=64, max_theta=4096)
    last = ctrl.current_theta
    moves = []
    for step in range(10):
        new = ctrl.compute_next_theta(
            producer_time=1.0, consumer_time=1.0, slack_time=0.0, time_step=step
        )
        moves.append(abs(new - last))
        last = new
    # With zero slack, the Newton numerator is 0 → no large excursions.
    assert max(moves) <= ctrl.initial_theta


def test_reset_restores_initial_theta():
    ctrl = create_des_controller(initial_theta=512)
    for step in range(5):
        ctrl.compute_next_theta(1.0, 0.5, 5.0, step)
    ctrl.reset()
    assert ctrl.current_theta == 512
    assert ctrl.step_count == 0
    assert ctrl.theta_history == []


def test_get_state_keys(des_controller):
    des_controller.compute_next_theta(1.0, 0.5, 0.5, 0)
    des_controller.compute_next_theta(2.0, 1.0, 1.0, 1)
    state = des_controller.get_state()
    assert "current_theta" in state
    assert "stats" in state
    assert "config" in state
    assert state["config"]["min_theta"] == 64
    assert state["config"]["max_theta"] == 4096


def test_stats_classify_slack_regimes():
    ctrl = create_des_controller(initial_theta=512)
    ctrl.compute_next_theta(1.0, 0.5, 10.0, 0)    # starvation δ>0
    ctrl.compute_next_theta(1.0, 0.5, -10.0, 1)   # saturation δ<0
    ctrl.compute_next_theta(1.0, 0.5, 0.0, 2)     # balanced δ≈0
    stats = ctrl.stats
    assert stats["starvation_steps"] >= 1
    assert stats["saturation_steps"] >= 1
    assert stats["balanced_steps"] >= 1
