# Discussion Summary: Minimal Enactive Agent

## 1. Background

This project did not start from a standard AI question such as “how to optimize behavior for a fixed goal.”
It started from a different question:

> When an external goal is not explicitly given, what does a system internally optimize in order to select the next action candidate?

This was motivated by a very ordinary human experience:

> “What should I do today?”

This is not just goal-directed planning.
It is closer to the emergence of a goal itself.

The hypothesis developed through discussion is that this kind of behavior requires more than explicit reasoning.
It requires a mechanism that integrates:

- current bodily and environmental conditions
- internal state
- past history
- tendencies and habits
- implicit valuation
- action readiness
- context-dependent behavioral switching

This led to the idea that the problem is not best framed as generic intelligence, but as a minimal form of **biological agency**.

---

## 2. Shift in framing

At first, the discussion used terms such as:

- intrinsic value signal
- minimal intelligence
- prediction
- memory
- mode switching
- internal state

However, these concepts overlapped and were not at the same level of abstraction.

The discussion gradually converged on a cleaner formulation.

### Final high-level framing

Minimal biological agency can be understood through two core concepts:

1. **Body-environment closed loop**
2. **State-dependent coherence drive**

---

## 3. Core concepts

### 3.1 Body-environment closed loop

The system is not treated as a disembodied optimizer.
It exists in a loop:

- it senses
- it acts
- the environment changes
- the changed environment alters future input

Agency is therefore not an internal computation alone.
It is a dynamical relation between body and environment.

### 3.2 State-dependent coherence drive

When an external goal is not fixed, the system still tends to generate a “next action.”
This is not modeled as generic reward maximization.

Instead, the system is assumed to shift toward action tendencies that are more coherent with its current internal state and ongoing body-environment relation.

“Coherence” here does **not** mean logical consistency.
It means something closer to:

- not collapsing
- not getting stuck
- maintaining viability
- preserving the possibility of continued interaction
- balancing local stability and continued exploration

A more operational interpretation is:

> coherence = maintaining viability while preserving exploratory continuity

---

## 4. Reinterpretation of value judgment

Under this framework, value judgment is not treated as an independent primitive.

Instead:

> Value judgment is the local ranking of action candidates produced by state-dependent coherence drive.

In other words:

- the closed loop is the arena
- coherence drive is the principle
- value judgment is the local expression of that principle during action selection

This avoids reducing the problem too early to a single scalar reward signal.

---

## 5. Why prediction and memory appeared in the discussion

The project then turned to the question:

> Are prediction and memory necessary for biological agency to become more sophisticated?

The discussion reached a layered answer.

### 5.1 At minimal levels

In worms, plants, or even simpler systems, what matters first may not be explicit prediction in a human-like sense.
What matters more fundamentally is:

- history-dependent state update
- persistence
- adaptive switching
- environment-coupled regulation

In such systems, prediction and memory may exist only in weak or implicit forms.

### 5.2 At human levels

For human-like cognition, prediction and memory appear to be indispensable.

Not necessarily in the form of explicit symbolic modules,
but at least as:

- persistence of past influence on present state
- anticipatory adjustment of action
- reuse of past structure for future construction
- comparison of possible future states

Thus:

- **minimal agency** may exist without rich human-like prediction/memory
- **human-like cognition** almost certainly cannot

---

## 6. Continuous but non-identical relation across organisms

The discussion considered worms, slime molds, plants, and humans.

The resulting view is:

- there is likely a **continuity of principle**
- but not an identity of mechanism

For example:

- worms likely have history-dependent state transitions and weak anticipatory structure
- humans have layered memory, distributed prediction, reconstruction, and counterfactual comparison

So the project should not try to copy worms as worms.
It should try to identify the **essential mechanism** that remains meaningful across scales.

---

## 7. Minimal formal model

The discussion gradually converged on a compact practical form for the first implementation.

### Practical minimal form

$$
\begin{aligned}
h_{t+1} &= f(h_t, i_t, m_t) \\
m_{t+1} &= g(m_t, h_t, i_t) \\
a_t &= \phi(m_t)
\end{aligned}
$$

Where:

- `h_t`: slow internal state
- `m_t`: behavior mode
- `i_t`: body-environment input
- `a_t`: action

### Interpretation

This form is meant to be the smallest practical expression of the project's current hypothesis.

- `h_t` carries slow persistence, internal bias, and history dependence
- `m_t` represents the current behavioral regime or action tendency
- `i_t` connects the agent to the body-environment loop
- `a_t` is generated from the current mode state

The update functions `f` and `g` are the computational locus of what we have called **state-dependent coherence drive**.

In this framing:

- memory is not a separate symbolic module
- prediction is not necessarily explicit
- value judgment is not a standalone scalar

Instead, persistence, switching, and action selection are all treated as consequences of recurrent state dynamics within a closed loop.

### Why this form was preferred

This formulation was preferred because it is:

- compact
- interpretable
- easy to implement
- easy to ablate
- flexible enough to support later extensions

It is not intended as a full account of intelligence.
It is intended as the smallest useful formal core for testing **minimal enactive agency**.

### Compressed view

At the most compressed level, the model says:

> an agent maintains an internal state, updates it through closed-loop interaction, and generates action through mode dynamics shaped by that state

This is the current minimal formal hypothesis of the project.
---

## 8. Why this is not yet “human intelligence”

A key point from the discussion:

This minimal module may be a good **proto-agent** or **proto-cognitive unit**,
but it is probably not sufficient for human-like intelligence.

Human cognition likely requires additional structure such as:

- multi-timescale hierarchy
- shared latent state across subsystems
- re-usable and reconstructive memory
- explicit or implicit comparison of alternative futures
- counterfactual evaluation
- long-range integration across modalities and contexts

So:

> the minimal model is a candidate ancestor-form, not a full account of human intelligence

---

## 9. Relation to current ANN / Transformer models

The discussion also asked whether modern Transformers already possess this structure.

The conclusion was:

- Transformers do implement a form of context-dependent state update
- but they do **not** intrinsically implement biological agency in the above sense

Main reasons:

- no intrinsic body-environment closed loop by default
- no explicit slow internal viability-oriented state
- no clear distinction between local mismatch and global arbitration
- no endogenous coherence drive comparable to biological regulation

Therefore, Transformers may contain a **partial formal resemblance**,
but they do not yet realize the intended structure.

---

## 10. Implementation direction

The project should begin with a small proof of concept rather than a large benchmark.

### Recommended first target

A minimal embodied agent with:

- a slow internal state `h`
- a behavior mode variable `m`
- a simple closed-loop environment

### Recommended environment

A simple 2D foraging environment with:

- depleting food patch
- sparse risk signal
- local sensing only
- no fixed external goal state

### Evaluation should focus on:

- stay / leave behavior
- explore / exploit switching
- history dependence
- effect of ablations
- emergence of mode persistence and switching

The point is not benchmark score.
The point is to show that the added structure produces behaviors that cannot be reduced to immediate stimulus-response.

---

## 11. Working research stance

The project should avoid getting trapped in broad philosophical disputes such as:

- what exactly counts as intelligence
- whether plants are intelligent
- whether slime molds “really think”

Instead, a more useful stance is:

> build the minimal mechanism of biological agency first, then ask what level of cognition it supports

This suggests using working terms such as:

- minimal enactive agency
- biological agency
- coherence-maintaining control
- state-dependent closed-loop adaptation

rather than making strong claims too early.

---

## 12. Current distilled thesis

At the current stage, the most distilled statement is:

> Minimal biological agency is the capacity to generate the next action through state-dependent coherence drive within a body-environment closed loop.

And the practical implementation hypothesis is:

> A minimal agent may only need a slow internal state, a behavior mode variable, and a closed-loop environment to exhibit the earliest meaningful form of state-dependent agency.

---

## 13. Open questions

The discussion left several important open questions.

### Conceptual
- Is “coherence” best defined as viability maintenance, exploratory continuity, or something more precise?
- At what point does history-dependent state update become genuine prediction?
- At what point does minimal agency become cognition?

### Computational
- Is the two-variable model sufficient to produce robust mode switching?
- Is explicit prediction necessary in the first implementation?
- Can local mismatch and global arbitration be added later without losing minimality?

### Comparative
- Which aspects of worm behavior are best seen as state transition rather than prediction?
- Which features of human cognition require qualitative additions beyond the minimal module?

---

## 14. Immediate next step

The immediate next step is not to solve cognition in general.

It is to implement a small closed-loop proof of concept that tests this claim:

> A minimal internal state plus behavior mode dynamics can generate coherent, history-dependent action selection without relying on an externally fixed goal.

That proof of concept is the purpose of this repository.