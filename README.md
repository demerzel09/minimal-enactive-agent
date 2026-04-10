# minimal-enactive-agent

A minimal embodied agent for studying how behavior emerges from internal state in a closed sensorimotor loop.

## Overview

This repository explores a simple question:

> How can an agent generate the next action when no explicit external goal is fixed?

Instead of treating behavior as pure reward maximization, this project starts from two working ideas:

1. **Body-environment closed loop**  
   The agent senses, acts, changes the environment, and is changed by it in return.

2. **State-dependent coherence drive**  
   The agent tends to generate actions that are more coherent with its current internal state and ongoing interaction with the environment.

The goal is to build the smallest useful computational model of **minimal enactive agency**.

## Core hypothesis

A meaningful proto-agent may emerge from only:

- a **slow internal state**
- a **behavior mode variable**
- a **closed-loop environment**

This is meant to test whether simple state-dependent dynamics can generate:

- stay / leave behavior
- explore / exploit switching
- history dependence
- mode persistence and switching

without relying on a fixed external objective.

## Minimal model

The practical minimal form currently considered is:

$$
\begin{aligned}
h_{t+1} &= f(h_t, i_t, m_t) \\
m_{t+1} &= g(m_t, h_t, i_t) \\
a_t &= \phi(m_t)
\end{aligned}
$$

Where:

- \( h_t \): slow internal state
- \( m_t \): behavior mode
- \( i_t \): body-environment input
- \( a_t \): action

This model is intentionally small, interpretable, and easy to ablate.

## First proof of concept

The first PoC should implement:

- a minimal 2D foraging environment
- a depleting food patch
- a sparse risk signal
- local sensing only
- an agent with internal state `h` and mode variable `m`

## Repository direction

The project is intended to proceed in small steps:

1. define the minimal model clearly
2. build a minimal environment
3. implement the smallest working agent
4. run ablations
5. expand only when necessary

## Working terminology

To avoid unnecessary philosophical inflation, this project uses terms such as:

- minimal enactive agency
- biological agency
- state-dependent coherence
- closed-loop adaptation

rather than claiming full intelligence from the outset.

## Implemented PoC components

- `src/env.py` — 2D closed-loop foraging environment
  - one depleting food patch
  - sparse risk signal
  - local sensing only (`[local_food, local_risk, food_delta]`)
- `src/agent.py` — minimal recurrent rate-based agent with `h` and `m`
- `src/run_simulation.py` — single-episode simulation runner
- `src/eval.py` — basic qualitative metrics
- `src/viz.py` — trajectory + state visualization

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

This repository does **not** optimize for generic benchmark performance.

The main questions are:

- Does internal state matter?
- Does mode dynamics matter?
- Can history-dependent behavior emerge?
- Can the agent show nontrivial stay/leave and explore/exploit patterns?

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
