# DES — Dynamic-Equilibrium Streaming

**DES overlaps the *produce*, *transfer*, and *consume* steps of a dataflow, making every
producer→consumer edge barrier-free.**

Both producer→consumer edges expose partial results that are streamable instead of a stop-and-go
barrier:

- on **tool→LLM**, each fetched item (or a divisible portion of one) forms a *chunk* the consumer
  LLM prefills while the producer fetches the next;
- on **agent→agent**, a chunk of a worker's decoded tokens is prefilled by the consumer before the
  worker finishes.

The difficulty is the chunk size `θ`: too coarse leaves network/compute tails unhidden; too fine
adds per-chunk overhead. Worse, under the runtime dynamics of MAS a *fixed* `θ` oscillates between
consumer starvation (bubbles) and backlog — and enlarging `θ` inflates produce, transfer, and consume
time *to different degrees*, so the tuning direction is non-obvious. DES resolves this with an online
controller that drives the pipeline toward a **dynamic equilibrium**, where each chunk is ready just
as the consumer needs it.

## Dynamic pipeline model

Model the stream as `N` unequal chunks, the `i`-th of size `θ_i` (tokens). From the consumer's view,
total dataflow latency decomposes into startup, work, and stall:

```
T_total =  (T_p,1 + T_t,1)        # L_start  — one cold chunk (produce + transfer)
         + Σ_{i=1..N} T_c,i       # L_work   — total consumption work
         + Σ_{i=2..N} max(0, δ_i) # L_stall  — accumulated bubbles
```

where `T_p,i`, `T_t,i`, `T_c,i` are chunk `i`'s produce, transfer, and consume times; `δ_i` is the
**slack** between chunk `i`'s arrival and chunk `(i−1)`'s completion:

- `δ_i > 0` starves the consumer (a **bubble**);
- `δ_i < 0` backs it up (**backlog**).

`L_start` (one cold chunk) and `L_work` (total consumption) are largely inelastic, so DES minimizes
the stall by driving each slack **`δ_i → 0`**.

## Sensitivity-guided adaptation (OSE)

The first obstacle is *direction*: since every term grows with `θ`, a bubble alone does not reveal
whether to grow or shrink `θ`. The **Online Sensitivity Estimator (OSE)** estimates each side's
marginal sensitivity — the slope of its time with respect to `θ` — by a finite difference between
consecutive chunks:

```
Ŝ_pro,i = Δ(T_p + T_t) / Δθ  = [(T_p,i + T_t,i) − (T_p,i−1 + T_t,i−1)] / (θ_i − θ_i−1)   if θ_i ≠ θ_i−1
        = Ŝ_pro,i−1                                                                      otherwise

Ŝ_con,i = ΔT_c / Δθ          = (T_c,i − T_c,i−1) / (θ_i − θ_i−1)                          if θ_i ≠ θ_i−1
        = Ŝ_con,i−1                                                                      otherwise
```

Treating slack as locally linear in `θ` with slope `ρ_i = Ŝ_pro,i − Ŝ_con,i−1` (using `Ŝ_con,i−1`
because producing chunk `i` overlaps consuming `i−1`), DES drives the predicted next slack
`δ_i + ρ_i·Δθ_i` to **zero** (a Newton step):

```
Δθ_i = − δ_i / ρ_i
θ_{i+1} = Clip(θ_i + Δθ_i, θ_min, θ_max)
```

The clip to an offline-profiled range `[θ_min, θ_max]` (`θ_max` = peak-throughput chunk size,
`θ_min` = smallest size with acceptable overhead) also **bounds the step when `ρ_i → 0`** — meaning
producer and consumer are almost equally sensitive, so *no* `θ` can close the gap: a structural,
untunable stall.

## The four polarity regimes

`ρ` acts as a **polarity switch** that always nudges `θ` in the gap-closing direction:

| Regime | `ρ` | Slack `δ` | Action | Why |
|--------|-----|-----------|--------|-----|
| Production-Bound | `ρ > 0` | starvation (`δ > 0`) | **decrease** `θ_{i+1}` | smaller chunk arrives sooner |
| Production-Bound | `ρ > 0` | overwhelm (`δ < 0`) | **increase** `θ_{i+1}` | slow arrival rate, amortize GPU kernel-launch overhead |
| Consumer-Bound | `ρ < 0` | starvation (`δ > 0`) | **increase** `θ_{i+1}` | producer catches up by batching more tokens |
| Consumer-Bound | `ρ < 0` | overwhelm (`δ < 0`) | **decrease** `θ_{i+1}` | smaller chunk makes GPU prefill faster, aligning with producer |

Production-bound (`ρ > 0`) is common in remote tool retrieval and slow-worker decoding;
consumer-bound (`ρ < 0`) in compute-bound GPUs serving concurrent prefill. Unlike one-sided
controllers such as AIMD (which react to a single signal), DES reads **both** sides' sensitivities to
pick the step's exact direction and magnitude.

```
Algorithm: Dynamic-Equilibrium Streaming
Require: initial chunk size θ_1, bounds [θ_min, θ_max]
  for each chunk i, as it arrives and i−1 completes at the consumer:
      obtain T_p,i, T_t,i, T_c,i−1, and δ_i           # runtime telemetry
      compute Ŝ_pro,i and Ŝ_con,i−1                   # sensitivity estimation (OSE)
      ρ_i ← Ŝ_pro,i − Ŝ_con,i−1                       # bottleneck polarity
      θ_{i+1} ← Clip(θ_i − δ_i / ρ_i, θ_min, θ_max)   # adaptation
      signal θ_{i+1} to the producer                  # synchronization
```

### Quasi-static caveat

The finite-difference OSE and the locally-linear Newton step assume the produce/consume rates are
**quasi-static** across adjacent chunks — they vary slowly relative to the chunk cadence. The
`window_size` smoothing and the `[θ_min, θ_max]` clip keep the controller stable when this assumption
is temporarily violated; the `damping_factor` further attenuates the step under noisy telemetry.

## DES in AgentMesh

DES unifies the other two mechanisms into one streaming fabric: it streams [DTR](dtr.md)'s delta
items to the prefill engine as they are fetched (tool→LLM), and feeds partial tokens into
[BPP](bpp.md)'s branch-local prefill while the worker is still decoding (agent→agent). By the time
the last upstream byte arrives, the consumer is nearly ready to decode — masking transfer and prefill
overhead behind source production.

## Configuration

`DESConfig` ([`agentmesh/core/config.py`](../agentmesh/core/config.py)):

```python
from agentmesh.core.config import DESConfig

des = DESConfig(
    enabled=True,
    initial_theta=512,   # θ₀ — initial chunk size (tokens)
    theta_min=64,        # θ_min
    theta_max=4096,      # θ_max
    damping_factor=1.0,  # smoothing (1.0 = no damping)
    window_size=5,       # OSE sliding window
)
```

Public DES surface (`agentmesh.mechanisms.des`): `DESController`, `create_des_controller`,
`OnlineSensitivityEstimator`, `Observation`, `Sensitivity`. See [api_reference.md](api_reference.md).

## Results

- **Throughput:** on both the tool→LLM and agent→agent edges, DES sustains **higher throughput than
  serialized execution (No-Stream, NS)** and **any fixed-chunk Static Stream (SS)**, because it
  adapts chunk granularity to the live rate instead of committing to one offline value
  (`DES > NS > worst-SS`).
- **Dynamics:** on replayed Deep-Research traces (coefficient of variation **0.13–0.75**), the
  controller drives chunk size toward the rate-matched equilibrium as the producer/consumer rate
  shifts, collapsing bubbles or backlog.
- **Ablation contribution:** DES adds **6%–17%** cumulative speedup on top of DTR + BPP by overlapping
  the produce–transfer–consume steps.
