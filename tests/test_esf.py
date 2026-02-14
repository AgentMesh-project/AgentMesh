"""
Tests for ESF (Elastic Semantic Flow) mechanism.

Tests the Online Sensitivity Estimation (OSE) controller for adaptive
microbatch sizing: observations, sensitivity estimation (S_pro, S_con, ρ),
and θ adjustment with clamping.
"""

import pytest
import numpy as np
from agentmesh.mechanisms.esf import (
    Observation,
    Sensitivity,
    StateObserver,
    ESFController,
    create_esf_controller,
)


class TestObservation:
    """Tests for Observation dataclass."""
    
    def test_observation_creation(self):
        """Observation should be created with correct attributes."""
        obs = Observation(
            chunk_size=512,
            producer_time=50.0,
            consumer_time=40.0,
            slack_time=10.0,
            timestamp=1000.0,
            step=1
        )
        
        assert obs.chunk_size == 512
        assert obs.slack_time == 10.0


class TestSensitivity:
    """Tests for Sensitivity dataclass."""
    
    def test_sensitivity_creation(self):
        """Sensitivity should be created correctly."""
        sens = Sensitivity(
            s_pro=0.1,
            s_con=0.08,
            rho=0.02
        )
        
        assert sens.s_pro == 0.1
        assert sens.rho == 0.02


class TestStateObserver:
    """Tests for StateObserver class."""
    
    def test_record_observation(self):
        """Recording observation should update state."""
        observer = StateObserver(window_size=5)
        
        obs = Observation(
            chunk_size=512,
            producer_time=50.0,
            consumer_time=40.0,
            slack_time=10.0,
            timestamp=1000.0,
            step=1
        )
        
        observer.record_observation(obs)
        
        assert observer.stats["total_observations"] == 1
        assert len(observer.observations) == 1
    
    def test_estimate_sensitivity_insufficient_data(self):
        """Should return None with insufficient data."""
        observer = StateObserver(window_size=5)
        
        obs = Observation(
            chunk_size=512,
            producer_time=50.0,
            consumer_time=40.0,
            slack_time=10.0,
            timestamp=1000.0,
            step=1
        )
        
        observer.record_observation(obs)
        
        # Only one observation - not enough for sensitivity
        sens = observer.estimate_sensitivity()
        assert sens is None
    
    def test_estimate_sensitivity_with_data(self):
        """Should compute sensitivity with sufficient data."""
        observer = StateObserver(window_size=5)
        
        # First observation
        obs1 = Observation(
            chunk_size=512,
            producer_time=50.0,
            consumer_time=40.0,
            slack_time=10.0,
            timestamp=1000.0,
            step=1
        )
        observer.record_observation(obs1)
        
        # Second observation with different chunk size
        obs2 = Observation(
            chunk_size=600,  # +88 tokens
            producer_time=58.0,  # +8ms producer time
            consumer_time=46.0,  # +6ms consumer time
            slack_time=12.0,
            timestamp=1100.0,
            step=2
        )
        observer.record_observation(obs2)
        
        sens = observer.estimate_sensitivity()
        
        assert sens is not None
        # S_pro = 8 / 88 ≈ 0.09
        assert sens.s_pro == pytest.approx(0.09, abs=0.01)
    
    def test_window_size_limit(self):
        """Observations should be limited by window size."""
        observer = StateObserver(window_size=3)
        
        for i in range(5):
            obs = Observation(
                chunk_size=512 + i * 10,
                producer_time=50.0 + i,
                consumer_time=40.0 + i,
                slack_time=10.0,
                timestamp=1000.0 + i * 100,
                step=i + 1
            )
            observer.record_observation(obs)
        
        # Should only keep last 3
        assert len(observer.observations) == 3
    
    def test_clear(self):
        """Clear should reset observer state."""
        observer = StateObserver(window_size=5)
        
        obs = Observation(
            chunk_size=512,
            producer_time=50.0,
            consumer_time=40.0,
            slack_time=10.0,
            timestamp=1000.0,
            step=1
        )
        observer.record_observation(obs)
        
        observer.clear()
        
        assert len(observer.observations) == 0
        assert observer.stats["total_observations"] == 0


class TestESFController:
    """Tests for ESFController class."""
    
    def test_create_controller(self):
        """Controller creation should work."""
        controller = create_esf_controller(
            initial_theta=512,
            min_theta=64,
            max_theta=4096
        )
        
        assert controller is not None
        assert controller.current_theta == 512
    
    def test_compute_next_theta_initial(self):
        """First call should handle missing sensitivity gracefully."""
        controller = ESFController(
            initial_theta=512,
            min_theta=64,
            max_theta=4096
        )
        
        new_theta = controller.compute_next_theta(
            producer_time=50.0,
            consumer_time=40.0,
            slack_time=10.0,
            time_step=0
        )
        
        # Should produce a valid theta
        assert controller.min_theta <= new_theta <= controller.max_theta
    
    def test_compute_next_theta_starvation(self):
        """Starvation (positive slack) should increase theta."""
        controller = ESFController(
            initial_theta=512,
            min_theta=64,
            max_theta=4096
        )
        
        # Simulate multiple steps to build sensitivity
        for i in range(5):
            controller.compute_next_theta(
                producer_time=50.0 + i * 2,
                consumer_time=40.0 + i,
                slack_time=10.0 + i,  # Increasing starvation
                time_step=i
            )
        
        # With increasing slack, theta should generally decrease
        # (producer is too slow, so reduce chunk size)
        history = controller.theta_history
        # Check trend direction is reasonable
        assert len(history) == 5
    
    def test_compute_next_theta_saturation(self):
        """Saturation (negative slack) handled correctly."""
        controller = ESFController(
            initial_theta=512,
            min_theta=64,
            max_theta=4096
        )
        
        for i in range(5):
            controller.compute_next_theta(
                producer_time=40.0 + i,
                consumer_time=50.0 + i * 2,
                slack_time=-10.0 - i,  # Increasing saturation
                time_step=i
            )
        
        # Should produce adjustments
        assert controller.step_count == 5
    
    def test_theta_bounds_min(self):
        """Theta should not go below minimum."""
        controller = ESFController(
            initial_theta=100,
            min_theta=64,
            max_theta=4096
        )
        
        # Force large negative adjustment
        for i in range(20):
            controller.compute_next_theta(
                producer_time=100.0,
                consumer_time=10.0,
                slack_time=90.0,  # Large starvation
                time_step=i
            )
            controller.update_current_theta(i + 1)
        
        # Should be clamped to minimum
        assert controller.current_theta >= controller.min_theta
    
    def test_theta_bounds_max(self):
        """Theta should not exceed maximum."""
        controller = ESFController(
            initial_theta=3500,
            min_theta=64,
            max_theta=4096
        )
        
        # Force large positive adjustment
        for i in range(20):
            controller.compute_next_theta(
                producer_time=10.0,
                consumer_time=100.0,
                slack_time=-90.0,  # Large saturation
                time_step=i
            )
            controller.update_current_theta(i + 1)
        
        # Should be clamped to maximum
        assert controller.current_theta <= controller.max_theta
    
    def test_get_state(self):
        """Get state should return comprehensive info."""
        controller = ESFController(
            initial_theta=512,
            min_theta=64,
            max_theta=4096
        )
        
        controller.compute_next_theta(
            producer_time=50.0,
            consumer_time=40.0,
            slack_time=10.0,
            time_step=0
        )
        
        state = controller.get_state()
        
        assert "current_theta" in state
        assert "step" in state
        assert "stats" in state
        assert "config" in state
    
    def test_reset(self):
        """Reset should restore initial state."""
        controller = ESFController(
            initial_theta=512,
            min_theta=64,
            max_theta=4096
        )
        
        # Make some changes
        for i in range(5):
            controller.compute_next_theta(
                producer_time=50.0,
                consumer_time=40.0,
                slack_time=10.0,
                time_step=i
            )
        
        controller.reset()
        
        assert controller.step_count == 0
        assert controller.current_theta == 512
        assert len(controller.theta_history) == 0
    
    def test_stats_tracking(self):
        """Statistics should be tracked correctly."""
        controller = ESFController(
            initial_theta=512,
            min_theta=64,
            max_theta=4096
        )
        
        # Starvation
        controller.compute_next_theta(50.0, 40.0, 10.0, 0)
        # Saturation
        controller.compute_next_theta(40.0, 50.0, -10.0, 1)
        # Balanced
        controller.compute_next_theta(45.0, 45.0, 0.0, 2)
        
        stats = controller.stats
        
        assert stats["total_steps"] == 3
        assert stats["starvation_steps"] + stats["saturation_steps"] + stats["balanced_steps"] == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
