---
title: The Inference-Time Compute Revolution
description: Why modern reasoning models like o1, o3, and DeepSeek-R1 "think before they speak" — and what thinking tokens mean for how you design agents.
layout: ../../../layouts/TutorialPage.astro
---

import { Tabs, TabItem } from '@astrojs/starlight/components';

For most of LLM history, scaling meant training bigger models on more data. The bet was: smarter model at training time → better outputs at inference time. That bet still holds — but in 2024–2026, a second axis of intelligence emerged: **scaling at inference time**.

This changes how agents are built. Understanding it is not optional.

---

## Why Modern Models "Think Before They Speak"

Classic LLMs are **fast thinkers**. They read a prompt and immediately predict the next token. The process is:

```
prompt → [single forward pass] → output
```

This is fast and cheap. It's also what causes hallucinations, arithmetic errors, and shallow reasoning. The model doesn't "check its work" — it commits to the first plausible continuation and runs with it.

The insight behind inference-time compute is simple: **give the model more time to think before committing to an answer**.

Just like a human expert pauses, drafts, considers alternatives, and revises before speaking — a reasoning model runs multiple internal steps before producing its final answer. The computation is scaled *after* training, at inference time.

```
prompt → [think... think... revise... think...] → final output
```

The "thinking" isn't just prompt engineering. It's baked into how the model is trained and served.

---

## Understanding Thinking Tokens

The mechanism behind this is **thinking tokens** — a dedicated stream of tokens the model generates *internally* before producing its visible response.

Here's what the raw interaction actually looks like under the hood:

```
User: What is 17 × 24?

<think>
  17 × 24
  = 17 × 20 + 17 × 4
  = 340 + 68
  = 408
</think>

The answer is 408.
```

The `<think>...</think>` block is the thinking token stream. It:

- Does **not** appear in the final API response (usually hidden)
- **Does** consume tokens — and therefore costs money
- Can run for hundreds or thousands of tokens on hard problems
- Is where the model actually reasons, self-corrects, and compresses

:::note[Why "tokens" and not "steps"?]
LLMs don't reason with discrete logical steps like a computer program. They reason by generating tokens in a chain. Each token conditions the next, so a long think-stream is effectively an internal reasoning scratchpad expressed as language. The model is literally writing out its thinking process.
:::

### Thinking Budget

Modern APIs expose a control called the **thinking budget** — a cap on how many thinking tokens the model is allowed to generate before answering.

| Budget | Speed | Cost | Reasoning Depth |
|---|---|---|---|
| Low (0–500 tokens) | Fast | Cheap | Surface-level |
| Medium (500–5,000 tokens) | Moderate | Moderate | Multi-step |
| High (5,000–32,000 tokens) | Slow | Expensive | Complex multi-hop |

```python
# Anthropic Claude — controlling thinking budget
import anthropic

client = anthropic.Anthropic()

response = client.messages.create(
    model="claude-3-7-sonnet-20250219",
    max_tokens=16000,
    thinking={
        "type": "enabled",
        "budget_tokens": 10000  # ← thinking budget
    },
    messages=[{
        "role": "user",
        "content": "Explain the tradeoffs between ReAct and Tree-of-Thought planning for agents."
    }]
)

# Thinking content is in response.content[0] (type: "thinking")
# Final answer is in response.content[1] (type: "text")
for block in response.content:
    if block.type == "thinking":
        print(f"[Thinking]: {block.thinking[:200]}...")  # usually very long
    else:
        print(f"[Answer]: {block.text}")
```

---

## Reasoning Models: o1, o3, DeepSeek-R1

These are the three most important reasoning models as of early 2026. They differ in how they train for reasoning, what they expose, and what they cost.

### OpenAI o1 and o3

OpenAI's o-series models were trained using **Reinforcement Learning from Outcomes** — the model is rewarded for getting correct answers on hard tasks (math, coding, logic), which pushes it to develop internal reasoning strategies.

- **o1**: The first widely-deployed reasoning model. Hidden thinking tokens. Strong at math, code, multi-step logic.
- **o3**: Significantly more capable, especially on complex reasoning benchmarks. Also significantly more expensive.
- **o1-mini**: Smaller, cheaper, still a strong reasoner for structured tasks.

Key characteristic: the thinking process is **completely hidden**. The API returns only the final answer. You cannot see or access the thinking tokens.

```python
# OpenAI o1 — reasoning model (thinking is hidden)
from openai import OpenAI

client = OpenAI()

response = client.chat.completions.create(
    model="o1",  # or "o3", "o1-mini"
    messages=[{
        "role": "user",
        "content": "Design a multi-agent system to automate code review for a GitHub repository."
    }],
    # reasoning_effort controls the thinking budget:
    # "low" | "medium" | "high"
    reasoning_effort="high",
)

print(response.choices[0].message.content)
# Note: response.usage.completion_tokens_details.reasoning_tokens
# shows how many thinking tokens were consumed
print(f"Reasoning tokens used: {response.usage.completion_tokens_details.reasoning_tokens}")
```

### DeepSeek-R1

DeepSeek-R1 is an open-weight reasoning model from DeepSeek.ai that matches or exceeds o1 on many benchmarks — and crucially, **the thinking tokens are visible**.

This is significant for agents: you can read the model's reasoning trace, evaluate it, and feed it back into your system. It makes R1-based agents more debuggable and auditable.

<Tabs syncKey="programming-language">
  <TabItem label="Python" icon="python">
    ```python
    # DeepSeek-R1 via OpenAI-compatible API
    # Thinking tokens ARE visible in the response
    from openai import OpenAI

    client = OpenAI(
        api_key="your-deepseek-api-key",
        base_url="https://api.deepseek.com",
    )

    response = client.chat.completions.create(
        model="deepseek-reasoner",  # R1
        messages=[{
            "role": "user",
            "content": "Should I use an agent or a pipeline for summarizing a fixed PDF report?"
        }],
    )

    msg = response.choices[0].message

    # R1 exposes the chain of thought separately
    if hasattr(msg, "reasoning_content") and msg.reasoning_content:
        print("=== Thinking ===")
        print(msg.reasoning_content)

    print("\n=== Final Answer ===")
    print(msg.content)
    ```
  </TabItem>
  <TabItem label="Rust" icon="rust">
    ```rust
    // DeepSeek-R1 via HTTP (reqwest + serde_json)
    // The API is OpenAI-compatible, so the same pattern works
    use reqwest::Client;
    use serde_json::{json, Value};

    #[tokio::main]
    async fn main() -> Result<(), Box<dyn std::error::Error>> {
        let client = Client::new();

        let body = json!({
            "model": "deepseek-reasoner",
            "messages": [{
                "role": "user",
                "content": "Should I use an agent or a pipeline for summarizing a fixed PDF report?"
            }]
        });

        let response = client
            .post("https://api.deepseek.com/chat/completions")
            .header("Authorization", format!("Bearer {}", std::env::var("DEEPSEEK_API_KEY")?))
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await?
            .json::<Value>()
            .await?;

        let msg = &response["choices"][0]["message"];

        // R1 exposes reasoning separately
        if let Some(thinking) = msg["reasoning_content"].as_str() {
            println!("=== Thinking ===\n{}\n", &thinking[..thinking.len().min(500)]);
        }

        if let Some(answer) = msg["content"].as_str() {
            println!("=== Final Answer ===\n{}", answer);
        }

        Ok(())
    }
    ```
  </TabItem>
</Tabs>

---

## The Cost–Accuracy Tradeoff of Deliberate Reasoning

Reasoning models do not replace fast models. They occupy a specific niche:

| Task | Best Model Type | Why |
|---|---|---|
| Simple Q&A, formatting | Fast model (GPT-4o, Claude Sonnet) | Cheap, fast, sufficient |
| Code generation (simple) | Fast model | Single-pass is adequate |
| Hard math / proofs | Reasoning model (o1, R1) | Needs self-correction |
| Multi-step planning | Reasoning model | Benefits from deliberation |
| Tool selection in agents | Fast model | Low complexity decision |
| Evaluating agent outputs | Reasoning model | Needs careful judgment |

### The tradeoff in numbers (approximate)

| Model | Latency | Cost per 1M tokens | Reasoning visible? |
|---|---|---|---|
| GPT-4o | ~1–2s | ~$5 | ❌ |
| o1-mini | ~5–10s | ~$15 | ❌ |
| o1 | ~10–30s | ~$60 | ❌ |
| o3 | ~20–60s | ~$200+ | ❌ |
| DeepSeek-R1 | ~8–20s | ~$3 | ✅ |

:::tip[The agent designer's decision]
Use reasoning models at **decision forks** — moments in the agent loop where the wrong choice cascades into many wasted steps. Use fast models for **routine actions** (formatting, summarizing, simple tool calls). This hybrid routing is called the **Small Model Strategy** and is covered in depth in [Module 10](/agentic-ai/11-high-perf-engineering).
:::

### What this means for the agent loop

Recall the `observe → reason → plan → act → reflect` loop from the previous article. Reasoning models change which loop phases are *external* vs *internal*:

| Loop Phase | Fast Agent (GPT-4o) | Reasoning Agent (o1/R1) |
|---|---|---|
| Observe | External (your code) | External (your code) |
| Reason | External (prompt + loop iterations) | **Internal (thinking tokens)** |
| Plan | External (explicit prompting) | **Internal (thinking tokens)** |
| Act | External (tool dispatch) | External (tool dispatch) |
| Reflect | External (next loop iteration) | Partial internal |

With a reasoning model, you may need **fewer loop iterations** because the model does more deliberation in one shot. This trades API call latency (multiple fast calls) for single-call latency (one slow call that thinks longer).

---

## What's Next

You now understand the engine behind modern agent intelligence: inference-time compute. In the final article of Module 1, we get into the specific LLM primitives that every agent system is built from — system prompts, structured outputs, function calling, and context management.

→ **Next up**: [Modern LLM Primitives](/agentic-ai/02-foundations/d-modern-llm-primitives)
