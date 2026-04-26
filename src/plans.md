# series: Engineering Agent Systems




# 🔥 ARTICLE 1 — THE ENGINEERING OF UNCERTAINTY

## Purpose:

Set the worldview. Define Kavriq.

---

## Structure

### 1. The Illusion of Capability

* Chatbots look impressive
* But break on real workflows
* Examples:

  * multi-step tasks
  * retries
  * inconsistent outputs

---

### 2. The Real Problem

* LLMs are **stochastic**
* Systems need **determinism**
* This mismatch is the root issue

---

### 3. The Shift: From Prompts → Systems

* Prompt engineering ≠ engineering
* Agents = systems interacting with environments
* Introduce:

  * uncertainty
  * time
  * state

---

### 4. Core Thesis (VERY IMPORTANT)

> Agent systems are **closed-loop systems operating under uncertainty over time**

---

### 5. The 4 Pillars (Introduce, don’t go deep)

#### I. Reasoning

* local decision making

#### II. Action

* interacting with world

#### III. Execution

* state + flow over time

#### IV. Control

* constraints + safety

---

### 6. Why Current Learning Falls Short

* IIT/JHU teach components
* Missing:

  * runtime behavior
  * failure over time
  * state persistence

---

### 7. What This Series Will Do

* define systems thinking
* go beyond tools/frameworks
* focus on production reality

---

### 8. Bridge to Deep Dives

Link:

* Beyond the String
* Prompt as Protocol

---

---

# 🔥 ARTICLE 2 — WHY AGENTS FAIL (EXECUTION GAP)

## Purpose:

Your strongest differentiation

---

## Structure

### 1. The Demo vs Reality Gap

* demos work
* production fails

---

### 2. The Missing Dimension: Time

* single call vs long-running systems
* accumulation of errors

---

### 3. Failure Modes (very important)

#### a. State Drift

* context changes
* memory inconsistency

#### b. Partial Execution

* step 3 fails after step 2 success

#### c. Tool Failure

* APIs break
* invalid outputs

#### d. Infinite Loops

* retry without control

---

### 4. Why Chains Break

* linear execution
* no recovery model
* no memory model

---

### 5. The Need for Loops

* systems must:

  * retry
  * validate
  * correct

---

### 🎬 Murali Visual (MANDATORY)

**Plan → Act → Validate → Fail → Retry loop**

* highlight failure path
* highlight retry loop

---

### 6. State Over Time

* systems evolve
* memory accumulates
* decisions depend on history

---

### 7. Real System Requirements

* persistence
* retry logic
* failure handling
* validation

---

### 8. Bridge to Deep Dives

Link:

* Fallacy of Chains
* Memory Architecture
* Cyclic Reasoning

---

---

# 🔥 ARTICLE 3 — FROM DAGS TO STATE MACHINES

## Purpose:

Define your abstraction layer

---

## Structure

### 1. The Industry Mental Model

* LangChain
* DAGs
* pipelines

---

### 2. Why DAGs Are Not Enough

* no cycles
* no adaptive flow
* no state transitions

---

### 3. What Systems Actually Need

* loops
* branching
* retries
* dynamic decisions

---

### 4. State Machines (Core Concept)

Define:

* state
* transition
* event
* action

---

### 5. Mapping Agents → State Machines

Example:

* planning → state
* tool execution → transition
* validation → transition

---

### 🎬 Murali Visual

* nodes (states)
* arrows (transitions)
* loop edges

---

### 6. Cyclic Reasoning

* re-evaluation
* retry
* fallback

---

### 7. Multi-Agent Systems (brief)

* shared state
* message passing
* coordination

---

### 8. Where This Leads

* runtime systems
* Sutra (preview)

---

### 9. Bridge to Deep Dives

Link:

* Constraint Engine
* Performance Gap

---

---

# 🔥 ARTICLE 4 — CONTROLLED AGENCY (TOOLS + SAFETY)

## Purpose:

Production readiness

---

## Structure

### 1. The Risk of Unbounded Agents

* real-world consequences
* incorrect actions

---

### 2. Action Layer Basics

#### Tooling Contracts

* JSON schema
* structured input/output

---

#### MCP (high-level)

* unified interface
* system integration

---

### 3. Why Tools Alone Are Dangerous

* unrestricted access
* hallucinated actions

---

### 4. Control Systems

#### Bounded Autonomy

* scoped permissions

#### Action Gating

* approvals
* checkpoints

---

### 🎬 Murali Visual

* agent tries action
* blocked by guardrail
* HITL approval flow

---

### 5. Observability

* tracing
* logs
* debugging steps

---

### 6. Production Architecture

* separation:

  * reasoning
  * execution
  * control

---

### 7. Practical Guidelines

* never trust raw output
* validate everything
* restrict access

---

### 8. Bridge to Deep Dives

Link:

* Tooling Contract
* MCP Server
* Bounded Autonomy
* Observability Trace
