# minimal-enactive-agent

A minimal embodied agent for studying how behavior emerges from internal state in a closed sensorimotor loop.

## Overview

This repository explores a simple question:

> How can an agent generate the next action when no explicit external goal is fixed?

The first proof of concept (PoC) in this repo follows the docs and implements:

1. **Body-environment closed loop**
2. **State-dependent coherence dynamics**

The minimal architecture uses:

- `h_t`: slow internal state
- `m_t`: behavior mode variable

and a simple local-sensing 2D foraging world.

## Implemented PoC components

- `src/env.py` — 2D closed-loop foraging environment
  - one depleting food patch
  - sparse risk signal
  - local sensing only (`[local_food, local_risk, food_delta]`)
- `src/agent.py` — minimal recurrent rate-based agent with `h` and `m`
- `src/run_simulation.py` — single-episode simulation runner
- `src/eval.py` — basic qualitative metrics
- `src/viz.py` — trajectory + state visualization

## Minimal model form

The implementation follows the repository model spec:

\[
\begin{aligned}
h_{t+1} &= f(h_t, i_t, m_t) \\
m_{t+1} &= g(m_t, h_t, i_t) \\
a_t &= \phi(m_t)
\end{aligned}
\]

With ablation flags:

- `use_h: false` for **no internal state h**
- `use_m: false` for **no mode variable m**

## Setup

Python 3.10+ recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install numpy matplotlib pyyaml
```

## Run simulation

### Full model

```bash
python -m src.run_simulation --config configs/full.yaml
```

### Ablation: no h

```bash
python -m src.run_simulation --config configs/ablation_no_h.yaml
```

### Ablation: no m

```bash
python -m src.run_simulation --config configs/ablation_no_m.yaml
```

Outputs are written to each config's `simulation.output_dir`, for example:

- `outputs/full/metrics.json`
- `outputs/full/trajectory.png`
- `outputs/full/states.png`

## Evaluation focus

The PoC is not tuned for benchmark reward. It is designed for inspectable qualitative behavior:

- stay / leave behavior
- explore / exploit switching
- mode persistence and switching
- history dependence proxy
- ablation effects

Current metrics in `src/eval.py` include:

- time in patch
- stay/leave transitions
- average patch residence proxy
- exploration radius
- mode switch count
- mode persistence
- history dependence proxy (food-trend vs leave tendency)

## Notes on simplicity

This first PoC intentionally avoids:

- heavy RL training libraries
- large ML frameworks
- unnecessary abstraction layers

Dynamics are hand-tuned and inspectable to support iterative conceptual validation.

## Documents

- `docs/discussion_summary.md` — background and conceptual path
- `docs/poc_plan.md` — implementation-oriented proof-of-concept plan
- `docs/model_spec.md` — model details and assumptions
- `docs/experiment_plan.md` — evaluation plan
