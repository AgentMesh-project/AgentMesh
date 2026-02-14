# ESF — Elastic Semantic Flow

## Problem

Streaming LLM responses to downstream consumers involves a producer (LLM generating tokens) and a consumer (next agent processing them).  If the microbatch size θ is too large the consumer starves; too small and the consumer saturates with scheduling overhead.

## Mechanism — Online Sensitivity Estimation (OSE)

ESF frames the producer–consumer interaction as two pipelines operating on **microbatches** of size θ:

### Observations (per step i)

| Symbol | Meaning |
|--------|---------|
| $T_{p,i} + T_{n,i}$ | Producer time: LLM generation + network transfer |
| $T_{c,i}$ | Consumer time: downstream processing |
| $\delta_i = (T_{p,i} + T_{n,i}) - T_{c,i}$ | Slack: positive → starvation, negative → saturation |

### Sensitivity Estimation

$$S_{pro,i} \approx \frac{\Delta(T_p + T_n)}{\Delta\theta}, \qquad S_{con,i} \approx \frac{\Delta T_c}{\Delta\theta}$$

$$\rho_i = S_{pro,i} - S_{con,i-1}$$

> Note: $S_{con,i-1}$ (lagged) is used because the consumer processes the *previous* microbatch while the producer generates the *current* one.

### θ Adjustment

$$\Delta\theta_i = -\frac{\delta_i}{\rho_i}, \qquad \theta_{i+1} = \text{clamp}(\theta_i + \Delta\theta_i,\; \theta_{\min},\; \theta_{\max})$$

### Operational Regimes

- **Production-Bound** ($\delta > 0$): producer is slower; decrease θ to reduce per-batch generation time.
- **Consumption-Bound** ($\delta < 0$): consumer is slower; increase θ to amortize scheduling overhead.

## API

```python
from agentmesh.mechanisms.esf import ESFController, create_esf_controller

controller = create_esf_controller(
    initial_theta=512,
    min_theta=64,
    max_theta=4096,
)

# Each streaming step
for step in range(num_steps):
    new_theta = controller.compute_next_theta(
        producer_time=t_p,
        consumer_time=t_c,
        slack_time=delta,
        time_step=step,
    )
    # Use new_theta as the next microbatch size
```
