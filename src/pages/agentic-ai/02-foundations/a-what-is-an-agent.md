---
title: What Is an Agent? A Technical Definition
description: Moving beyond the buzzword — a precise technical look at what makes a system an agent, how agents differ from pipelines, and when not to build one.
layout: ../../../layouts/TutorialPage.astro
---

import { Tabs, TabItem, Steps } from '@astrojs/starlight/components';

In the [overview section](/agentic-ai/01-overview/introduction-to-agents), we defined agents conceptually. Now we get precise. This article draws a sharp technical line between an agent and a pipeline, shows you what an agent loop looks like in code, and — critically — explains when you should *not* use an agent at all.

---

## What Actually Makes a System an Agent?

An agent is a system that:

1. **Perceives state** from the environment (a message, a file, a tool result)
2. **Decides autonomously** what to do next — using an LLM as the decision engine
3. **Acts** by calling tools or producing output
4. **Loops** until a goal is reached or a stop condition is hit

The key word is **autonomously**. A pipeline has a fixed execution graph decided by you, the programmer. An agent has a *dynamic* execution graph decided by the LLM at inference time.

:::note[One-liner definition]
A pipeline runs code you wrote. An agent runs code it *decides* to write — one step at a time.
:::

---

## Deterministic Pipelines vs Autonomous LLM Loops

Here is the clearest way to see the difference — the same task implemented as a pipeline vs an agent.

**Task**: Given a user's research question, search the web, extract key points, and write a summary.

### Approach A — Deterministic Pipeline

```
[User Query] → [Search Tool] → [Extract Top 3 URLs] → [Scrape Pages] → [LLM: Summarize] → [Output]
```

**You define every step.** The execution order is fixed. The LLM is called exactly once at the end. There are no decisions — just transformations.

### Approach B — Autonomous Agent Loop

```
[User Query] → [LLM decides] →
   • If it needs more info → calls search tool → feeds result back → [LLM re-evaluates]
   • If search wasn't relevant → calls a different tool
   • If it has enough → writes the summary → stops
```

The LLM decides *which* tools to call, *how many times*, and *when to stop*. This is the agent loop.

---

## The Minimal Agent Loop in Code

Let's see what this looks like in practice. This is the simplest possible agent: it gives an LLM a list of tools and loops until the LLM decides it's done.

<Tabs syncKey="programming-language">
  <TabItem label="Python" icon="python">
    ```python
    # minimal_agent.py
    import json
    from openai import OpenAI

    client = OpenAI()

    def search(query: str) -> str:
        # stub — in reality, call a search API
        return f"Results for '{query}': [result1, result2, result3]"

    def calculator(expression: str) -> str:
        return str(eval(expression))

    TOOLS = [
        {
            "type": "function",
            "function": {
                "name": "search",
                "description": "Search the web for information",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "calculator",
                "description": "Evaluate a math expression",
                "parameters": {
                    "type": "object",
                    "properties": {"expression": {"type": "string"}},
                    "required": ["expression"],
                },
            },
        },
    ]

    TOOL_MAP = {"search": search, "calculator": calculator}

    def run_agent(user_query: str) -> str:
        messages = [{"role": "user", "content": user_query}]

        while True:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                tools=TOOLS,
            )
            msg = response.choices[0].message

            # If no tool call → agent is done
            if not msg.tool_calls:
                return msg.content

            # Execute each tool call the LLM requested
            messages.append(msg)
            for tool_call in msg.tool_calls:
                fn_name = tool_call.function.name
                fn_args = json.loads(tool_call.function.arguments)
                result = TOOL_MAP[fn_name](**fn_args)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result,
                })

    if __name__ == "__main__":
        answer = run_agent("What is the population of Tokyo divided by 1000?")
        print(answer)
    ```
  </TabItem>
  <TabItem label="Rust" icon="rust">
    ```rust
    // main.rs — minimal agent loop
    use serde::{Deserialize, Serialize};
    use serde_json::{json, Value};

    // Stub tool implementations
    fn search(query: &str) -> String {
        format!("Results for '{}': [result1, result2, result3]", query)
    }

    fn calculator(expression: &str) -> String {
        // In production use a safe expression evaluator crate
        format!("Evaluated: {}", expression)
    }

    fn dispatch_tool(name: &str, args: &Value) -> String {
        match name {
            "search" => {
                let query = args["query"].as_str().unwrap_or("");
                search(query)
            }
            "calculator" => {
                let expr = args["expression"].as_str().unwrap_or("");
                calculator(expr)
            }
            _ => format!("Unknown tool: {}", name),
        }
    }

    // In a real implementation, call the OpenAI API via reqwest.
    // This stub shows the loop structure clearly.
    fn run_agent(user_query: &str) {
        let mut messages: Vec<Value> = vec![
            json!({"role": "user", "content": user_query})
        ];

        loop {
            // ── call LLM ──────────────────────────────────────
            // let response = call_openai(&messages, &tools).await;
            // Stubbed here for clarity:
            let has_tool_call = false; // would be response.choices[0].finish_reason == "tool_calls"

            if !has_tool_call {
                // LLM returned a final answer — break the loop
                println!("Agent finished.");
                break;
            }

            // ── execute tool calls ────────────────────────────
            // for each tool_call in response.tool_calls {
            //     let result = dispatch_tool(&tool_call.name, &tool_call.arguments);
            //     messages.push(json!({
            //         "role": "tool",
            //         "tool_call_id": tool_call.id,
            //         "content": result,
            //     }));
            // }
        }
    }

    fn main() {
        run_agent("What is the population of Tokyo divided by 1000?");
    }
    ```
  </TabItem>
</Tabs>

:::tip[What to notice in the loop]
The `while True` / `loop` is the agent. The LLM controls how many iterations happen. Once it returns a response with **no tool calls**, the loop exits with the final answer. This is the fundamental pattern every agentic framework (LangGraph, AutoGen, CrewAI) builds on.
:::

---

## When NOT to Use an Agent

This is the most underrated decision in agentic AI. Agents add latency, cost, unpredictability, and failure modes. **Use a pipeline whenever you can.**

| Use Case | Pipeline or Agent? | Reason |
|---|---|---|
| PDF → extract text → summarize | ✅ Pipeline | Fixed, predictable steps |
| Classify support ticket into category | ✅ Pipeline | Single LLM call, no tools needed |
| Research assistant that searches and refines | 🤖 Agent | Number of searches is unknown upfront |
| Code debugging with iterative test runs | 🤖 Agent | Requires feedback loops and retries |
| Generate a weekly report from a fixed DB schema | ✅ Pipeline | Steps are fully determined |
| Customer service bot that books, refunds, escalates | 🤖 Agent | Path through the system depends on customer input |

**The rule of thumb:** If you can draw the full execution graph *before* running — use a pipeline. If the path depends on what the LLM discovers at runtime — you need an agent.

---

## The Cost of Autonomy

Every iteration of the agent loop is an LLM API call. A simple pipeline might call the LLM once. An agent on the same task might call it 3–10 times. This matters for:

- **Latency**: Each hop adds 1–5 seconds
- **Cost**: 5x API calls = 5x the token bill
- **Reliability**: More steps = more chances to hallucinate or use the wrong tool

This is why Module 1.3 (Inference-Time Compute) is so important: newer reasoning models like o1 *internalize* some of this deliberation, reducing the number of external loop iterations needed.

---

## What's Next

You now have a precise technical definition of an agent and can recognize when to use one vs. a simple pipeline. In the next article, we'll go deeper into the structure of the agent loop itself — the full `observe → reason → plan → act → reflect` cycle and how it maps to real system components.

→ **Next up**: [The Cognitive Architecture of Agents](/agentic-ai/02-foundations/cognitive-architecture)
