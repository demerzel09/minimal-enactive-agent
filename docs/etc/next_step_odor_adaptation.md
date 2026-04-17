# Next Step Proposal: Odor Adaptation Before New Body or State Expansion

## 1. Purpose

This document summarizes the current interpretation of the odor-field experiments and proposes the next architectural step.

The aim is not to add complexity prematurely.
The aim is to identify the smallest missing function that is now required by the environment.

The current conclusion is:

> Before adding new body structure, new spatial state variables, or hierarchical internal state, the project should first test whether the observed failure is caused by the absence of sensory adaptation.

---

## 2. Current observation

Based on the current odor-field experiments, the following pattern appears repeatedly:

- the agent can react to odor gradients
- GA optimization can greatly increase transition counts
- however, the agent still tends to become trapped in local looping behavior
- in practice it reaches only a small subset of available food patches
- the loop is often maintained around depleted but still odorous regions

This suggests that the current architecture is not failing because it lacks action, recurrence, or internal state in general.

Instead, it suggests a narrower problem:

> the agent can detect odor, but cannot distinguish between “currently useful odor” and “odor level that has become locally normal”

In other words, the system reacts to odor **absolute level**, but lacks a mechanism for becoming adapted to the current local baseline.

---

## 3. Working hypothesis

The current failure is primarily a **sensory-layer problem**, not yet a body problem and not yet a higher-level memory problem.

More specifically:

- `h` is functioning as a slow internal state
- `m` is functioning as a mode variable
- GA can tune behavior within this architecture
- but the observation pipeline still treats moderate persistent odor as behaviorally meaningful even after local depletion

This causes local looping and poor escape from depleted odor hotspots.

Therefore the next minimal step should be:

> introduce sensory adaptation at the level of `i`, before modifying body structure or introducing new higher-level state variables

---

## 4. Why this is the right next step

## 4.1 It matches primitive foraging logic

Primitive chemotaxis-like systems do not begin with spatial maps.
They begin with:

- local sensing
- temporal comparison
- adaptation to current baseline
- conditional reorientation

The current agent already has:

- local sensing
- food delta
- slow internal integration (`h`)
- mode structure (`m`)

What it still lacks is:

- adaptation of the sensory baseline itself

This makes sensory adaptation the most natural next candidate.

## 4.2 It preserves the architecture principles

This change does not alter the roles of the existing core variables.

- `h` remains internal and slow
- `m` remains behavioral/regime-like
- `i` remains sensory coupling
- `a` remains action

Only the **construction of `i`** changes.

This is fully consistent with the principle that sensing should be treated as body-environment coupling, not as a passive raw feed.

## 4.3 It respects the environment-first strategy

The odor-field environment created a new failure mode.
That failure should first be addressed at the smallest layer directly implicated by it.

The problem emerged because the environment now contains:

- lingering odor
- local depletion
- partial mismatch between odor intensity and actual food utility

This does not yet require new body structure.
It first requires a better way of sensing.

---

## 5. Proposed minimal change

## 5.1 Single-timescale sensory adaptation

Let the odor sensor maintain a slowly adapting baseline:

```text
baseline_{t+1} = (1 - alpha_adapt) * baseline_t + alpha_adapt * odor_t
odor_dev_t = odor_t - baseline_t

Then use odor_dev_t as the primary sensory signal instead of raw odor alone.

Interpretation:

baseline_t represents the current adapted odor level
odor_dev_t represents deviation from what has become normal
positive odor_dev_t means “odor stronger than expected here”
near-zero odor_dev_t means “still odorous, but no longer improving”

This should allow the agent to stop treating persistent residual odor as fresh evidence of food gain.

5.2 Dual-timescale sensory adaptation (optional second condition)

A second experiment should test whether two sensory baselines are better than one:

a faster adaptive baseline
a slower adaptive baseline

This does not yet mean adding hierarchical h.
It means only testing whether the sensory layer itself benefits from multiple timescales.

This is useful because the project may later need to distinguish:

fast sensory normalization
slower contextual normalization

But that distinction should first be tested inside sensing, not inside global internal state.

6. Recommended experiment set
Experiment A: sensory adaptation minimal comparison

Compare at least these conditions:

Current sensing
raw odor
existing food delta
no adaptive baseline
Single adaptation
adapted baseline
odor deviation
existing h and m unchanged
Dual-timescale adaptation
fast baseline
slow baseline
one or both deviations as inputs

The body and the core h/m architecture should remain unchanged.

7. Recommended evaluation changes

The current odor-field setting makes some existing metrics less reliable.

In particular, raw transition count can become misleading:
high transitions may reflect repeated local looping rather than successful broader foraging.

The following metrics should be added or emphasized.

7.1 Primary metrics
unique_patch_count
productive_patch_fraction
patch_visit_entropy
first_passage_to_new_patch
depleted_patch_residence_time
same_patch_revisit_ratio
coverage_area
7.2 Interpretation metrics
raw odor signal over time
adapted baseline over time
odor deviation over time
food_delta
h time series
m time series
nearest patch identity
nearest productive patch distance

These should be plotted together where possible.

The goal is to determine whether the agent is trapped because it still experiences a region as “good,” or because it lacks any means to escape even after sensing normalization.

8. What should not be done yet

The following changes are likely premature at the current stage.

8.1 Do not add new body structure yet

The current failure does not yet look like a propulsion or morphology failure.
It looks like a sensing-normalization failure.

8.2 Do not introduce hierarchical h yet

Adding slower internal states before fixing the sensory layer risks solving a sensory problem indirectly through internal-state compensation.

That would make the architecture harder to interpret.

8.3 Do not rely on GA alone

The current GA results are already informative:
they suggest that parameter optimization can improve local behavior, but does not naturally solve the deeper representational issue.

This is a sign that the next step should be architectural at the sensory layer, not merely more optimization.

9. Decision tree after adaptation
Case A: adaptation works well

If adaptation significantly reduces local looping and increases the number of meaningfully visited patches, then the conclusion is strong:

primitive foraging required sensory adaptation before any richer map-like or higher-order mechanism

In that case, the project should continue exploring how far adaptation plus existing h/m can go before introducing more structure.

Case B: adaptation helps but does not solve the problem

Then the next candidate should be stochastic reorientation modulation.

For example:
if odor deviation remains low for a sustained time, increase the probability of large reorientation events.

This remains a primitive action-selection extension, not yet a body expansion.

Case C: adaptation barely helps

Only then should the project consider:

hierarchical h
persistent body-space state
externalized or bodily trace memory
richer exploratory mechanisms

That order matters.

10. Current recommended priority

The recommended immediate priority is:

add sensory adaptation to sense()
compare no-adaptation vs single-timescale vs dual-timescale adaptation
replace transition-centered evaluation with patch-diversity and hotspot-trapping metrics
inspect whether local looping is actually broken
only then decide whether the next missing piece is:
stochastic reorientation
hierarchical internal state
body-space state
new body function
11. Summary

The current odor-field experiments appear to reveal a missing primitive function:

the agent can sense odor, but cannot yet adapt to odor as a local baseline

Therefore the next step should not be a major architectural expansion.
It should be the smallest possible change that makes the sensory layer sensitive to deviation from a learned local baseline.

In short:

before adding new body or higher-order state, test whether primitive sensory adaptation is the missing function that the odor environment has now made necessary