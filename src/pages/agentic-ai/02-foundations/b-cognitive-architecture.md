---
title: The Cognitive Architecture of Agents
description: How agents build a world model, set goals, and execute the full observe → reason → plan → act → reflect loop — with working code in Python and Rust.
layout: ../../../layouts/TutorialPage.astro
---

import { Tabs, TabItem, Steps } from '@astrojs/starlight/components';

In the [previous article](/agentic-ai/02-foundations/what-is-an-agent), we established what makes a system an agent. Now we look *inside* it. What model of the world does an agent carry in its head? How does it set and pursue goals? And what does each phase of the loop actually do in code?

---

## The World Model Concept

Every intelligent system — human or machine — needs some representation of its environment in order to act on it. In cognitive science this is called a **world model**: an internal map of what exists, what state things are in, and how actions change that state.

For an LLM-based agent, the world model lives in the **context window**. Every piece of information the agent has about the current situation — the user's goal, tool results so far, conversation history, observations — is encoded as text and placed into the prompt.

```
┌─────────────────────────────────────────────┐
│  AGENT CONTEXT (World Model)                │
│                                             │
│  System prompt:  "You are a research agent" │
│  Goal:           "Find Q3 revenue for AAPL" │
│  Observations:   [tool_result_1, result_2]  │
│  Current state:  "Have 2 sources, need 3"   │
└─────────────────────────────────────────────┘
```

This has a critical implication: **the quality of an agent's decisions is bounded by the quality of its world model**. If the context is missing key information, stale, or poorly formatted, the agent will reason on a broken map.

:::note[World model vs Memory]
The context window is the agent's *active* world model — what it's working with right now. Long-term memory (vector databases, episodic stores) is how agents *persist and retrieve* parts of the world model across sessions. We cover this in detail in [Module 5 — Memory Systems](/agentic-ai/06-memory-systems-rag).
:::

---

## Goal-Driven Behavior in Autonomous Systems

Traditional software is instruction-driven: you call a function, it executes, it returns. Agents are **goal-driven**: you provide a high-level objective, and the agent figures out the sequence of actions to achieve it.

This distinction has a concrete consequence. An instruction-driven system requires you to specify *how*. A goal-driven system only needs you to specify *what*.

| | Instruction-Driven | Goal-Driven (Agent) |
|---|---|---|
| **You specify** | Exact steps | Desired outcome |
| **Flexibility** | Rigid | Adaptive |
| **Handles unexpected inputs** | ❌ Fails or ignores | ✅ Re-plans |
| **Cost** | Predictable | Variable |
| **Example** | `summarize(fetch(url))` | "Summarize the most relevant article about X" |

**Goal representation in practice** — goals are almost always set in the system prompt:

```text
System prompt:
  You are a research assistant. Your goal is to answer the user's question
  using only verified sources. Keep searching until you have at least 3
  independent sources confirming the answer. Then provide a final summary
  with citations.
```

The LLM internalizes this as a goal and will keep iterating through the agent loop until its *own internal evaluation* says the goal is met.

---

## The Expanded Agent Loop

The simple loop from the last article (`call LLM → call tools → repeat`) is correct but incomplete. A production-grade agent runs through **five distinct phases** on every iteration:

```
observe → reason → plan → act → reflect
```

Let's break each phase down technically.

### Phase 1: Observe

The agent collects everything relevant from its environment and formats it as context. This is not passive — it involves *selecting* what to include.

- Recent tool outputs
- Current conversation state
- Retrieved memory snippets
- System state (time, user info, constraints)

The quality of observation directly determines reasoning quality. Noisy or irrelevant context leads to poor decisions.

### Phase 2: Reason

The LLM reads the observation and produces an internal analysis. With chain-of-thought prompting this is visible in the output. With reasoning models (o1, R1) it happens via hidden "thinking tokens" before the response. Either way, this is where the LLM asks itself:

- "What do I know?"
- "What don't I know yet?"  
- "What are my options?"
- "What's most important to do next?"

### Phase 3: Plan

Reasoning produces a *plan* — a decision about what action to take. In simple agents this is implicit (the LLM decides to call a tool). In sophisticated agents it's explicit: the LLM outputs a structured plan before acting.

```json
{
  "thinking": "I need the population of Tokyo. I should search for it first.",
  "next_action": "call_tool",
  "tool": "search",
  "args": { "query": "Tokyo population 2024" }
}
```

### Phase 4: Act

The planned action is executed. This is the only phase where side effects happen — a tool runs, an API is called, a file is written. The output is captured as an **observation** to feed the next iteration.

### Phase 5: Reflect

After acting, the agent evaluates: did the action help? Is the goal closer? Should the approach change? This phase separates basic agents from self-improving ones.

Reflection can be implicit (the LLM re-reads its context and decides naturally) or explicit (a dedicated "self-critique" step where the agent scores its own output and decides to revise or continue).

---

## Implementing the Full Loop

Here's the `observe → reason → plan → act → reflect` loop as annotated code:

<Tabs syncKey="programming-language">
  <TabItem label="Python" icon="python">
    ```python
    # cognitive_loop.py — the full five-phase agent loop
    import json
    from openai import OpenAI
    from dataclasses import dataclass, field
    from typing import Any

    client = OpenAI()

    @dataclass
    class AgentState:
        """The agent's world model — everything it knows right now."""
        goal: str
        observations: list[str] = field(default_factory=list)
        iteration: int = 0
        max_iterations: int = 10

    def observe(state: AgentState, new_observation: str | None = None) -> list[dict]:
        """
        Phase 1 — OBSERVE
        Build the context (world model) from current state.
        """
        if new_observation:
            state.observations.append(new_observation)

        messages = [
            {
                "role": "system",
                "content": (
                    f"You are a goal-driven agent. Your goal: {state.goal}\n"
                    f"Iteration: {state.iteration}/{state.max_iterations}\n"
                    "Think step-by-step. Call tools if you need more information. "
                    "When you have enough to fully answer the goal, respond without calling any tools."
                ),
            }
        ]
        # Add all accumulated observations as assistant context
        for obs in state.observations:
            messages.append({"role": "assistant", "content": f"[Observation]: {obs}"})

        return messages

    def reason_and_plan(messages: list[dict], tools: list[dict]) -> Any:
        """
        Phase 2+3 — REASON + PLAN
        The LLM reasons about the world model and decides what to do next.
        """
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            tools=tools,
        )
        return response.choices[0].message

    def act(tool_calls: list, tool_map: dict) -> list[str]:
        """
        Phase 4 — ACT
        Execute the planned tool calls and collect results.
        """
        results = []
        for tool_call in tool_calls:
            fn_name = tool_call.function.name
            fn_args = json.loads(tool_call.function.arguments)
            result = tool_map[fn_name](**fn_args)
            results.append(f"{fn_name}({fn_args}) → {result}")
        return results

    def reflect(state: AgentState, action_results: list[str]) -> str:
        """
        Phase 5 — REFLECT
        Evaluate the results and update the world model.
        In advanced agents this could call the LLM again for self-critique.
        Here we simply log and accumulate.
        """
        summary = " | ".join(action_results)
        print(f"  [Reflect] Iteration {state.iteration}: {summary}")
        return summary

    # ── Stub tools ─────────────────────────────────────────────
    def search(query: str) -> str:
        return f"Top result for '{query}': [stub data]"

    TOOL_MAP = {"search": search}
    TOOLS = [{
        "type": "function",
        "function": {
            "name": "search",
            "description": "Search the web",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    }]

    # ── Main loop ───────────────────────────────────────────────
    def run_cognitive_agent(goal: str) -> str:
        state = AgentState(goal=goal)

        while state.iteration < state.max_iterations:
            state.iteration += 1
            print(f"\n── Iteration {state.iteration} ──")

            # Phase 1: Observe — build world model
            messages = observe(state)

            # Phases 2+3: Reason + Plan
            decision = reason_and_plan(messages, TOOLS)

            # No tool call = goal achieved (agent decided it's done)
            if not decision.tool_calls:
                print(f"  [Done] {decision.content}")
                return decision.content

            # Phase 4: Act
            results = act(decision.tool_calls, TOOL_MAP)

            # Phase 5: Reflect — update world model with new observations
            observation = reflect(state, results)
            state.observations.append(observation)

        return "Max iterations reached without completing goal."

    if __name__ == "__main__":
        answer = run_cognitive_agent("Find the current CEO of OpenAI and their background.")
        print(f"\nFinal answer:\n{answer}")
    ```
  </TabItem>
  <TabItem label="Rust" icon="rust">
    ```rust
    // cognitive_loop.rs — five-phase agent loop structure
    // Note: Actual OpenAI API calls require the `reqwest` + `serde_json` crates.
    // This shows the structural pattern clearly.

    use serde_json::{json, Value};

    /// The agent's world model — everything it knows right now.
    struct AgentState {
        goal: String,
        observations: Vec<String>,
        iteration: usize,
        max_iterations: usize,
    }

    impl AgentState {
        fn new(goal: &str) -> Self {
            AgentState {
                goal: goal.to_string(),
                observations: Vec::new(),
                iteration: 0,
                max_iterations: 10,
            }
        }
    }

    /// Phase 1 — OBSERVE: Build the context (world model) from current state.
    fn observe(state: &AgentState) -> Vec<Value> {
        let mut messages = vec![json!({
            "role": "system",
            "content": format!(
                "You are a goal-driven agent. Your goal: {}\nIteration: {}/{}\n\
                Think step-by-step. Call tools if you need more information. \
                When you have enough, respond without calling tools.",
                state.goal, state.iteration, state.max_iterations
            )
        })];

        for obs in &state.observations {
            messages.push(json!({
                "role": "assistant",
                "content": format!("[Observation]: {}", obs)
            }));
        }

        messages
    }

    /// Phase 4 — ACT: Dispatch a tool call and return the result.
    fn act(tool_name: &str, args: &Value) -> String {
        match tool_name {
            "search" => {
                let query = args["query"].as_str().unwrap_or("");
                format!("Top result for '{}': [stub data]", query)
            }
            _ => format!("Unknown tool: {}", tool_name),
        }
    }

    /// Phase 5 — REFLECT: Log results and decide whether to continue.
    fn reflect(state: &AgentState, results: &[String]) {
        println!(
            "  [Reflect] Iteration {}: {}",
            state.iteration,
            results.join(" | ")
        );
    }

    fn run_cognitive_agent(goal: &str) -> String {
        let mut state = AgentState::new(goal);

        while state.iteration < state.max_iterations {
            state.iteration += 1;
            println!("\n── Iteration {} ──", state.iteration);

            // Phase 1: Observe — build world model
            let _messages = observe(&state);

            // Phases 2+3: Reason + Plan
            // In production: let response = call_openai_api(&messages, &tools).await;
            // Stubbed — simulate the LLM deciding to call a tool on iteration 1,
            // then returning a final answer on iteration 2.
            let has_tool_call = state.iteration < 2;

            if !has_tool_call {
                let answer = "Final answer from LLM (stub).".to_string();
                println!("  [Done] {}", answer);
                return answer;
            }

            // Phase 4: Act
            let tool_result = act("search", &json!({"query": goal}));
            let results = vec![tool_result.clone()];

            // Phase 5: Reflect
            reflect(&state, &results);
            state.observations.push(tool_result);
        }

        "Max iterations reached without completing goal.".to_string()
    }

    fn main() {
        let answer = run_cognitive_agent(
            "Find the current CEO of OpenAI and their background."
        );
        println!("\nFinal answer:\n{}", answer);
    }
    ```
  </TabItem>
</Tabs>

---

## From Chatbots to Autonomous Systems

Understanding this progression is key to appreciating *why* the five-phase loop exists:

| System Type | Loop? | World Model? | Goal-Driven? | Plans? | Reflects? |
|---|:---:|:---:|:---:|:---:|:---:|
| **Rule-based chatbot** | ❌ | ❌ | ❌ | ❌ | ❌ |
| **Single-turn LLM** | ❌ | Partial | ❌ | ❌ | ❌ |
| **Conversational LLM** | ✅ (manual) | ✅ (context) | ❌ | ❌ | ❌ |
| **Basic agent** | ✅ | ✅ | ✅ | Implicit | ❌ |
| **Cognitive agent** | ✅ | ✅ | ✅ | Explicit | ✅ |

The chatbot responds. The LLM generates. The agent *pursues*. Each step in this progression adds a phase to the loop and expands the agent's capability — but also its cost and complexity.

:::tip[Design principle]
Only add loop phases you genuinely need. A basic agent (observe → act) is faster and cheaper than a cognitive agent (observe → reason → plan → act → reflect). Add the reflection phase only when you need self-correction; add explicit planning only when tasks require multi-step coordination.
:::

---

## What's Next

You now understand the cognitive architecture that underlies every serious agent system. In the next article, we step into 2026 territory: how **inference-time compute** (o1, DeepSeek-R1) changes this loop by internalizing the reasoning and planning phases inside the model itself.

→ **Next up**: [The Inference-Time Compute Revolution](/agentic-ai/02-foundations/inference-time-compute)
