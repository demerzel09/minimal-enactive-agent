# Model Specification: Minimal Enactive Agent

## 1. Purpose

This document specifies the first computational model used in `minimal-enactive-agent`.

The model is intentionally minimal.
Its purpose is not to reproduce a specific organism, nor to maximize benchmark reward.
Its purpose is to test whether a very small recurrent agent can generate meaningful state-dependent behavior in a closed body-environment loop.

## 2. Design principles

The model is built from two working principles:

1. **Body-environment closed loop**  
   The agent senses, acts, changes the environment, and then receives changed input.

2. **State-dependent coherence drive**  
   The agent's next action tendency depends on its current internal state and ongoing interaction history, rather than only on immediate input or an externally fixed goal.

## 3. Minimal state structure

The agent uses two core state variables:

- `h_t`: slow internal state
- `m_t`: behavior mode state

## 3.1 Slow internal state `h_t`

`h_t` is the minimal carrier of persistence and history dependence.

It is meant to capture effects such as:

- recent success or failure
- depletion pressure
- exploratory pressure
- internal drift caused by past interaction
- slowly changing behavioral bias

`h_t` should evolve more slowly than `m_t`.

This variable is deliberately abstract.
In the first PoC it is not assigned a single biological interpretation such as hunger or arousal.
It is a compact dynamical state.

## 3.2 Behavior mode `m_t`

`m_t` represents the agent's current behavioral regime.

In the first implementation, it may correspond loosely to a small set of modes such as:

- exploit / stay
- explore / leave
- avoid / escape

`m_t` should evolve faster than `h_t`, but still retain persistence.

The project does not assume that mode must be a hard discrete state.
A soft mode representation is acceptable and may be preferable for implementation.

## 4. Input and action

## 4.1 Input `i_t`

`i_t` denotes body-environment input available at time `t`.

Typical components may include:

- local food intensity
- local risk intensity
- local food gradient estimate
- previous action
- optional proprioceptive signal

The agent should not observe the full environment state.

## 4.2 Action `a_t`

`a_t` denotes the action produced at time `t`.

The first implementation should use a small action set, such as:

- move forward
- turn left
- turn right
- pause / stay

If continuous movement is easier, action can be represented as:

- heading change
- movement magnitude

The action space should remain simple.

## 5. Update equations

The practical minimal model is:

\[
h_{t+1} = f(h_t, i_t, m_t)
\]

\[
m_{t+1} = g(m_t, h_t, i_t)
\]

\[
a_t = \phi(m_t)
\]

This structure should be interpreted as follows:

- `h` accumulates slow internal history
- `m` converts current state plus input into mode dynamics
- `a` is generated from mode state

## 6. Recommended implementation form

The first implementation should use a small recurrent rate-based model.

A recommended form is:

\[
h_{t+1} = (1-\alpha_h) h_t + \alpha_h \tanh(W_{hh} h_t + W_{hm} m_t + W_{hi} i_t + b_h)
\]

\[
u_{t+1} = (1-\alpha_m) u_t + \alpha_m \tanh(W_{uh} h_t + W_{uu} u_t + W_{ui} i_t + b_u)
\]

\[
m_t = \text{softmax}(u_t)
\]

\[
a_t = W_a m_t + b_a
\]

Where:

- `h_t` is the slow internal state vector
- `u_t` is the pre-softmax mode activation
- `m_t` is the behavior mode distribution
- `a_t` is the action output
- `alpha_h < alpha_m` so that `h` is slower than mode activation

## 6.1 Why this form

This form is preferred because it gives:

- recurrence
- persistence
- mode competition
- internal-state dependence
- simple inspectable dynamics

without requiring large frameworks or biologically detailed neuron models.

## 7. Minimal dimensionality

The first PoC should remain small.

Suggested starting dimensions:

- `h`: 1 to 4 dimensions
- `m`: 2 to 4 modes
- `i`: 2 to 5 dimensions depending on environment design

A good starting point might be:

- `h` = 2 dimensions
- `m` = 3 modes
- `i` = 3 inputs

Example interpretation:

- `h[0]` = depletion / dissatisfaction tendency
- `h[1]` = exploratory drift
- `m` = `[exploit, explore, avoid]`

These interpretations are optional.
The implementation should remain flexible.

## 8. Time-scale assumption

A central hypothesis of the model is that behavior emerges from interaction between two time scales:

- **slow** internal drift and persistence
- **faster** mode-level switching

This distinction is important.

If everything updates at the same time scale, the system may collapse toward immediate reaction.
If internal state is too slow or too disconnected, it may become behaviorally irrelevant.

## 9. What counts as memory here

In this model, memory is not introduced as a separate symbolic memory module.

Instead, memory is implemented minimally as:

- persistence of `h_t`
- recurrence in `m_t`
- dependence of future state on past interaction

This is intentional.
The first PoC aims to test whether minimal history dependence is enough to produce interesting behavior.

## 10. What counts as prediction here

The first PoC does not require an explicit predictive model.

Prediction, if present at all in the first stage, is only implicit in the recurrence and dynamics.
The model may later be extended with:

\[
\hat{i}_{t+1} = P(h_t, m_t)
\]

\[
e_t = i_t - \hat{i}_t
\]

but this is not required initially.

The first model should stay simpler unless explicit prediction is necessary to reproduce the target behavior.

## 11. Learning vs hand-tuned dynamics

The first version does not need full learning.

Recommended order:

1. hand-tune weights and parameters
2. verify that interesting behavior is possible
3. add simple parameter search if needed
4. only later consider trainable optimization

This reduces confusion between:

- architecture effects
- training effects

The first goal is architectural validity.

## 12. Required qualitative properties

A successful model should exhibit at least some of the following:

- persistence in behavioral mode
- switching between modes under changing conditions
- sensitivity to recent history
- nontrivial stay/leave behavior
- nontrivial explore/exploit behavior
- degradation under ablations

## 13. Ablation targets

The model is only meaningful if its key parts can be tested by removal.

### Ablation A: remove slow internal state
- set `h_t = 0` or bypass `h`

Expected question:
- does the agent become overly reactive?

### Ablation B: remove mode dynamics
- bypass `m`
- map input directly to action

Expected question:
- does persistence disappear?

### Ablation C: weaken recurrence
- reduce self-connections or leak persistence

Expected question:
- does history dependence degrade?

## 14. Non-claims

This model does not yet claim to capture:

- full biological realism
- worm-specific neural circuitry
- explicit cognitive planning
- episodic memory
- human-like prediction
- symbolic reasoning

It is a proto-agent model, not a full cognitive architecture.

## 15. Current working hypothesis

The working hypothesis of the model is:

> A small recurrent system with a slow internal state and a behavior mode variable may be sufficient to produce the earliest useful form of state-dependent enactive agency in a closed-loop environment.

This hypothesis is what the first PoC is meant to test.