# Experiment Plan: Minimal Enactive Agent

## 1. Purpose

This document defines the first experiments for `minimal-enactive-agent`.

The aim is not to optimize benchmark performance.
The aim is to determine whether the minimal architecture produces meaningful closed-loop, state-dependent behavior that cannot be reduced to immediate stimulus-response.

## 2. Main experimental question

> Does a minimal agent with a slow internal state and a behavior mode variable produce behavior that is persistent, history-dependent, and adaptively switchable in a simple closed-loop environment?

## 3. First experimental environment

## 3.1 Environment A: Depleting Food Patch

The first environment should be a small 2D world containing:

- one main food patch
- local food sensing
- depletion when the agent remains in or near the patch
- sparse risk signal
- weak or absent explicit target location

### Rationale

This environment is chosen because it naturally probes:

- stay / leave behavior
- persistence
- history dependence
- exploration after local depletion
- interaction between exploitation and avoidance

It is better aligned with the project than fixed-goal navigation tasks.

## 3.2 Suggested environment mechanics

### World
- 2D continuous plane or small discrete grid
- bounded domain

### Food
- one region with initially high food intensity
- food depletes with local occupation or consumption
- optional slow regeneration

### Risk
- one sparse risk source or stochastic risk field
- risk should not dominate the task
- risk exists to test switching and avoidance

### Observation
- local food level
- local risk level
- local gradient estimate if desired
- optional previous action or heading

### Action
- forward move
- turn left
- turn right
- pause / stay

## 4. Experimental conditions

## 4.1 Full model

This is the reference condition.

Agent includes:

- slow internal state `h`
- mode dynamics `m`
- recurrence
- closed-loop interaction

This condition tests whether the architecture can produce the target behaviors.

## 4.2 Ablation A: no internal state

Remove or freeze `h`.

Possible implementation:
- set `h_t = 0`
- remove recurrence into `h`
- bypass `h` in downstream dynamics

Question:
- does the agent lose history dependence?

Expected failure mode:
- behavior becomes overly tied to immediate observation

## 4.3 Ablation B: no mode state

Remove or bypass `m`.

Possible implementation:
- map directly from input and optional `h` to action
- remove mode persistence

Question:
- does the agent lose stable action regimes?

Expected failure mode:
- behavior becomes noisy or shallowly reactive
- stay / leave switching becomes weak or erratic

## 4.4 Ablation C: reduced recurrence

Weaken self-recurrence or leak persistence.

Question:
- does mode persistence or internal continuity disappear?

Expected failure mode:
- behavior becomes unstable or memoryless

## 5. Core behavioral tests

## 5.1 Stay / leave test

Measure whether the agent remains in the food patch early, but leaves when local yield drops or risk rises.

Questions:
- does residence time depend on recent interaction history?
- does the leave probability increase after depletion?
- is leave timing more structured in the full model than in ablations?

## 5.2 Explore / exploit switching test

Measure whether the agent alternates between:

- local exploitation
- broader exploration

Questions:
- does switching emerge without a fixed external goal?
- do mode dynamics show persistence rather than random oscillation?
- do ablations reduce this structure?

## 5.3 History dependence test

Construct episodes in which immediate observation is similar, but recent history differs.

Example:
- same current food level
- different recent depletion trajectory

Questions:
- does action choice differ across histories?
- can the full model distinguish states that immediate input alone cannot?

## 5.4 Risk-sensitive switching test

Use sparse risk input to test whether the agent changes mode when exploitation becomes unsafe.

Questions:
- does the agent avoid pure greed?
- does risk influence mode switching, not just immediate turning?

## 6. Metrics

## 6.1 Basic behavioral metrics

- average residence time in food patch
- leave probability over time
- average food collected
- average risk exposure
- number of mode switches
- average mode duration
- path length
- exploration radius
- occupancy heatmap

These metrics are useful but not sufficient on their own.

## 6.2 History-sensitive metrics

- divergence in action distribution under matched current input but different recent history
- dependence of leave probability on recent yield trend
- dependence of mode occupancy on recent local depletion

These metrics are especially important because they test whether internal state matters.

## 6.3 Dynamics metrics

- temporal autocorrelation of mode variable
- temporal autocorrelation of action selection
- lagged coupling between internal state and mode transitions
- sensitivity of switching to slow-state drift

These help determine whether the model truly exhibits structured dynamics rather than random fluctuations.

## 7. Visualization plan

Visualization is essential for this project.

At minimum, each experiment should generate:

- trajectory plot
- food field snapshot or depletion map
- time series of `h`
- time series of `m`
- action sequence
- patch occupancy over time

Optional but useful:

- phase plot of `h` vs dominant mode
- comparison plots across ablations
- event markers for entering / leaving patch

## 8. Success criteria for the first PoC

The first PoC is successful if the full model shows at least some of the following:

- persistent behavioral modes
- nontrivial stay / leave behavior
- nontrivial explore / exploit switching
- dependence on recent interaction history
- clear qualitative degradation under ablation

A successful first PoC does **not** require:

- perfect optimization
- biological realism
- high reward score
- match to any one species

## 9. Failure cases and interpretation

## 9.1 If full model behaves like stimulus-response

Possible interpretation:
- internal state is too weak
- recurrence is too weak
- environment does not require history
- mode competition is poorly tuned

## 9.2 If full model is unstable or chaotic

Possible interpretation:
- recurrent gain too strong
- time-scale separation poorly chosen
- mode state too unconstrained

## 9.3 If ablations perform similarly to full model

Possible interpretation:
- environment is too easy
- architecture is unnecessarily complex
- internal state is not actually contributing
- mode variable is redundant in current setup

This would be an important result, not just a failure.

## 10. Recommended experimental order

### Phase 1: Environment sanity check
- visualize food and risk fields
- verify depletion logic
- verify action and movement

### Phase 2: Full model qualitative run
- run hand-tuned full model
- inspect trajectories and state traces

### Phase 3: Ablation comparison
- no `h`
- no `m`
- reduced recurrence

### Phase 4: Parameter sweep
- vary time constants
- vary recurrence strength
- vary mode competition strength

### Phase 5: Stress tests
- add observation noise
- vary patch quality
- vary depletion rate
- vary risk strength

## 11. Comparison logic

The most important comparison is not raw score.
It is behavioral structure.

Preferred order of interpretation:

1. Does full model show the intended qualitative patterns?
2. Do ablations weaken those patterns?
3. Can differences be explained by internal state and mode dynamics?
4. Which architectural element contributes most?

## 12. Deliverables

Each experiment batch should produce:

- saved plots
- summary metrics
- configuration used
- short textual interpretation

Suggested output structure:

- `outputs/<experiment_name>/plots/`
- `outputs/<experiment_name>/metrics.json`
- `outputs/<experiment_name>/config.yaml`
- `outputs/<experiment_name>/notes.txt`

## 13. Future extensions after first PoC

Only after the first PoC is successful should the project consider:

- explicit prediction error
- learned parameters
- multiple food patches
- hierarchical modes
- multiple timescale internal states
- recurrent modular composition
- comparison to biological trajectory data

## 14. Current stance

The first experiments are intended to answer a narrow question:

> Is a tiny recurrent closed-loop agent with slow internal state and mode dynamics already enough to generate a meaningful proto-form of enactive agency?

This is the claim the first experiment suite should test.