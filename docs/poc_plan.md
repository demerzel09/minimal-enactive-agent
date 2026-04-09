# PoC Plan: Minimal Enactive Agent

## 1. Purpose

Build a first proof of concept for a minimal embodied agent whose behavior emerges from:

- a **body-environment closed loop**
- **state-dependent coherence dynamics**

The point is not to maximize benchmark reward.
The point is to test whether a very small internal architecture can generate meaningful, history-dependent action selection.

## 2. Main research question

> Can a minimal agent with a slow internal state and a behavior mode variable produce coherent closed-loop behavior without relying on a fixed external goal?

More concretely:

- Can the agent switch between staying and leaving?
- Can it shift between exploration and exploitation?
- Can current behavior depend on past interaction history?
- Do these behaviors disappear under ablation?

## 3. Minimal agent design

### State variables

The first PoC uses two core variables:

- `h_t`: slow internal state
- `m_t`: behavior mode variable

### Interpretation

#### `h_t` — slow internal state
Represents slowly varying internal bias, such as:

- persistence of recent experience
- depletion pressure
- exploratory pressure
- internal tendency shaped by history

This is not yet meant to model a specific biological variable.
It is a compact state carrier.

#### `m_t` — behavior mode
Represents the current behavioral regime, for example:

- exploit / stay
- explore / leave
- avoid / escape

The first implementation may use either:

- a continuous vector with soft competition, or
- a small discrete-like softmax mode state

### Action output

Actions may be minimal, for example:

- move forward
- turn left
- turn right
- stay / pause

The exact set should remain small.

## 4. Minimal model form

A practical minimal update structure:

\[
h_{t+1} = f(h_t, i_t, m_t)
\]

\[
m_{t+1} = g(m_t, h_t, i_t)
\]

\[
a_t = \phi(m_t)
\]

Where:

- `i_t` is body-environment input
- `a_t` is action

### Recommended first implementation form

Use a small recurrent rate-based model.

Example structure:

- `h` updated with a slow leak
- `m` updated with a faster leak
- `m` converted to action preference
- no explicit prediction module at first
- no heavy learning framework at first

## 5. Environment

## 5.1 First environment: depleting food patch

A 2D world containing:

- one main food patch
- food depletion when the agent remains nearby
- sparse risk signal
- optional weak background food elsewhere

### Why this environment

It directly probes the behaviors of interest:

- when to stay
- when to leave
- when to switch mode
- whether history matters

It avoids overfitting the project to standard goal-reaching tasks.

## 5.2 Observation

The agent should receive only local or near-local information:

- local food intensity
- local risk intensity
- possibly local gradient estimate
- optional proprioceptive or previous-action signal

The agent should not receive the full global map.

## 5.3 Environment update

The environment should update after each action:

- food is partially consumed or depleted
- the agent position changes
- risk may fluctuate or remain sparse
- the resulting observation becomes the next input

This preserves the body-environment closed loop.

## 6. Implementation constraints

The first PoC should be intentionally small.

### Preferred stack

- Python
- numpy
- matplotlib
- pyyaml if config files are used

### Avoid for now

- large RL frameworks
- deep learning libraries unless clearly needed
- complicated training pipelines
- benchmark-heavy engineering

The first version should be inspectable and easy to modify.

## 7. Evaluation

## 7.1 Primary evaluation questions

We are not mainly asking “How high is reward?”
We are asking:

- Does internal state make a behavioral difference?
- Does mode structure make a behavioral difference?
- Does the system show nontrivial persistence and switching?
- Does it behave differently under different histories despite similar immediate input?

## 7.2 Suggested metrics

### Behavioral
- average patch residence time
- leave probability as a function of local depletion
- frequency of mode switching
- duration of mode persistence
- path tortuosity / exploration spread
- time spent near risk vs food

### History dependence
- whether current action distribution depends on recent interaction history
- whether leave behavior changes after prolonged low yield
- whether behavior differs under matched current observations but different recent pasts

### Robustness
- behavior under input noise
- sensitivity to parameter changes

## 7.3 Ablation tests

These are essential.

### Full model
- slow internal state `h`
- mode variable `m`

### Ablation A: no internal state
- remove `h`
- keep only immediate input and mode dynamics

Question:
- does behavior collapse toward stimulus-response?

### Ablation B: no mode variable
- remove `m`
- act directly from internal state and input

Question:
- does persistence / switching degrade?

### Ablation C: reduced recurrence
- weaken or remove self-recurrence

Question:
- does history dependence disappear?

## 8. Expected outcome of first PoC

Success does **not** require biological realism or high task performance.

A successful first PoC should show at least some of the following:

- state-dependent stay/leave behavior
- nontrivial explore/exploit switching
- mode persistence over time
- history-dependent behavior not reducible to immediate observation
- clear degradation under ablations

## 9. Deliverables for the first coding phase

### Core code
- `src/env.py` — minimal 2D foraging environment
- `src/agent.py` — minimal 2-variable agent
- `src/run_simulation.py` — run one simulation
- `src/eval.py` — compute simple metrics
- `src/viz.py` — trajectory and state visualization

### Configuration
- `configs/base.yaml`
- `configs/ablation_no_h.yaml`
- `configs/ablation_no_m.yaml`

### Documentation
- update `README.md`
- add model notes to `docs/model_spec.md`
- add experiment notes to `docs/experiment_plan.md`

## 10. Recommended development order

### Phase 1
Implement the environment only.

### Phase 2
Implement a fixed hand-tuned agent with the two-state update.

### Phase 3
Visualize trajectories, modes, and internal state.

### Phase 4
Run ablations.

### Phase 5
Only after that, consider parameter search or learning.

## 11. Non-goals for the first PoC

The first PoC should **not** attempt to do all of the following:

- full biological realism
- full worm behavior reproduction
- human-like cognition
- explicit world modeling
- large-scale memory systems
- full predictive coding implementation
- complex multi-agent interaction

These may become later directions, but they are not required for the first proof of concept.

## 12. Working stance

The project is trying to identify an essential mechanism, not to imitate a specific organism directly.

The current working stance is:

> If a tiny recurrent agent with a slow internal state and mode dynamics can already produce coherent closed-loop behavior, then it may capture a meaningful proto-form of enactive agency.

This is the claim the first PoC should test.