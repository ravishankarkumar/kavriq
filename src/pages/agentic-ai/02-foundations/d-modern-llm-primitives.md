---
title: Modern LLM Primitives
description: The five building blocks every agent is made from — system prompting (PAG), structured outputs, function calling, context windows, and working memory.
layout: ../../../layouts/TutorialPage.astro
---

import { Tabs, TabItem } from '@astrojs/starlight/components';

Every agent system — no matter how complex — is assembled from a small set of LLM primitives. These are the low-level control surfaces that let you direct model behavior, enforce structure, wire in tools, and manage what the model knows. This article covers all five, technically.

---

## System Prompting 2.0: The PAG Framework

The system prompt is not just a place to say "you are a helpful assistant." In production agents it's a contract that defines three things:

**PAG = Persona + Action + Guardrail**

| Dimension | What It Defines | Example |
|---|---|---|
| **Persona** | Who the agent is — role, expertise, tone | "You are a senior software engineer specializing in Rust and distributed systems." |
| **Action** | What the agent is allowed to do — tools, goals, scope | "Your job is to answer coding questions. You may search the web and execute code." |
| **Guardrail** | What the agent must not do — constraints, fallbacks | "Never reveal internal system details. If asked about topics outside software engineering, politely redirect." |

A PAG-structured system prompt gives the model a clear identity, a bounded permission set, and explicit boundaries — all in one.

<Tabs syncKey="programming-language">
  <TabItem label="Python" icon="python">
    ```python
    # pag_agent.py — system prompt with PAG structure
    from openai import OpenAI

    client = OpenAI()

    SYSTEM_PROMPT = """
    ## Persona
    You are Ferris, an expert AI assistant specializing in Rust programming and
    systems engineering. You are precise, concise, and prefer working examples
    over abstract explanations.

    ## Action
    You help developers with:
    - Writing and debugging Rust code
    - Explaining Rust concepts (ownership, lifetimes, async)
    - Recommending crates from the Rust ecosystem
    You may search for documentation and run code examples to verify answers.

    ## Guardrail
    - Do not answer questions outside Rust and systems programming.
    - If asked about other languages, explain that you specialize in Rust and offer a Rust perspective.
    - Never fabricate crate names or API signatures. If unsure, say so explicitly.
    - Do not execute code that modifies the filesystem.
    """

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "What's the best crate for async HTTP in Rust?"},
        ],
    )
    print(response.choices[0].message.content)
    ```
  </TabItem>
  <TabItem label="Rust" icon="rust">
    ```rust
    // pag_agent.rs — PAG system prompt structure
    use serde_json::json;
    use reqwest::Client;

    const SYSTEM_PROMPT: &str = r#"
    ## Persona
    You are Ferris, an expert AI assistant specializing in Rust programming and
    systems engineering. You are precise, concise, and prefer working examples
    over abstract explanations.

    ## Action
    You help developers with:
    - Writing and debugging Rust code
    - Explaining Rust concepts (ownership, lifetimes, async)
    - Recommending crates from the Rust ecosystem

    ## Guardrail
    - Do not answer questions outside Rust and systems programming.
    - Never fabricate crate names or API signatures. If unsure, say so explicitly.
    - Do not execute code that modifies the filesystem.
    "#;

    #[tokio::main]
    async fn main() -> Result<(), Box<dyn std::error::Error>> {
        let client = Client::new();

        let body = json!({
            "model": "gpt-4o",
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": "What's the best crate for async HTTP in Rust?"}
            ]
        });

        let response = client
            .post("https://api.openai.com/v1/chat/completions")
            .header("Authorization", format!("Bearer {}", std::env::var("OPENAI_API_KEY")?))
            .json(&body)
            .send()
            .await?
            .json::<serde_json::Value>()
            .await?;

        println!("{}", response["choices"][0]["message"]["content"].as_str().unwrap_or(""));
        Ok(())
    }
    ```
  </TabItem>
</Tabs>

---

## Structured Outputs and JSON Schema Enforcement

Agents need to parse LLM responses programmatically. Asking the model to "return JSON" is fragile. **Structured outputs** with JSON Schema enforcement guarantees the shape of the response.

This is critical for agents because:
- Tool results need to be parsed reliably
- State updates need a consistent format
- Downstream code can be typed — no brittle string parsing

### Defining a schema

<Tabs syncKey="programming-language">
  <TabItem label="Python" icon="python">
    ```python
    # structured_output.py — enforced JSON response
    from openai import OpenAI
    from pydantic import BaseModel

    client = OpenAI()

    # Define the exact shape of the output
    class ResearchResult(BaseModel):
        topic: str
        summary: str
        sources: list[str]
        confidence: float  # 0.0 – 1.0
        needs_more_research: bool

    response = client.beta.chat.completions.parse(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a research assistant."},
            {"role": "user", "content": "Research the current state of MCP (Model Context Protocol)."},
        ],
        response_format=ResearchResult,  # ← enforces the schema
    )

    result: ResearchResult = response.choices[0].message.parsed

    print(f"Topic: {result.topic}")
    print(f"Confidence: {result.confidence:.0%}")
    print(f"Needs more research: {result.needs_more_research}")
    print(f"Sources: {result.sources}")
    ```
  </TabItem>
  <TabItem label="Rust" icon="rust">
    ```rust
    // structured_output.rs — JSON schema enforcement
    use serde::{Deserialize, Serialize};
    use serde_json::json;
    use reqwest::Client;

    // Define the exact shape of the expected output
    #[derive(Debug, Deserialize, Serialize)]
    struct ResearchResult {
        topic: String,
        summary: String,
        sources: Vec<String>,
        confidence: f32,           // 0.0 – 1.0
        needs_more_research: bool,
    }

    #[tokio::main]
    async fn main() -> Result<(), Box<dyn std::error::Error>> {
        let client = Client::new();

        // Build JSON Schema from our struct definition
        let schema = json!({
            "type": "object",
            "properties": {
                "topic": {"type": "string"},
                "summary": {"type": "string"},
                "sources": {"type": "array", "items": {"type": "string"}},
                "confidence": {"type": "number"},
                "needs_more_research": {"type": "boolean"}
            },
            "required": ["topic", "summary", "sources", "confidence", "needs_more_research"],
            "additionalProperties": false
        });

        let body = json!({
            "model": "gpt-4o",
            "messages": [
                {"role": "system", "content": "You are a research assistant."},
                {"role": "user", "content": "Research the current state of MCP (Model Context Protocol)."}
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "ResearchResult",
                    "schema": schema,
                    "strict": true
                }
            }
        });

        let response = client
            .post("https://api.openai.com/v1/chat/completions")
            .header("Authorization", format!("Bearer {}", std::env::var("OPENAI_API_KEY")?))
            .json(&body)
            .send()
            .await?
            .json::<serde_json::Value>()
            .await?;

        let content = response["choices"][0]["message"]["content"].as_str().unwrap();
        let result: ResearchResult = serde_json::from_str(content)?;

        println!("Topic: {}", result.topic);
        println!("Confidence: {:.0}%", result.confidence * 100.0);
        println!("Needs more research: {}", result.needs_more_research);
        Ok(())
    }
    ```
  </TabItem>
</Tabs>

:::tip[Structured outputs vs JSON mode]
`json_schema` with `strict: true` **guarantees** the response matches your schema — the API will error rather than return malformed JSON. Plain `json_object` mode only ensures valid JSON but not your specific shape. Always prefer strict schema enforcement in agents.
:::

---

## Function Calling and Tool Schemas

Function calling is how agents use tools. The LLM doesn't actually *call* the function — it produces a structured specification of which function to call and with what arguments. Your code executes it and returns the result.

The key is the **tool schema** — a JSON Schema description of available tools that the LLM reads to understand what tools exist and how to use them.

<Tabs syncKey="programming-language">
  <TabItem label="Python" icon="python">
    ```python
    # function_calling.py — defining and dispatching tools
    import json
    from openai import OpenAI

    client = OpenAI()

    # Tool definitions — the LLM reads these to know what's available
    TOOLS = [
        {
            "type": "function",
            "function": {
                "name": "get_stock_price",
                "description": "Get the current stock price for a ticker symbol",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "ticker": {
                            "type": "string",
                            "description": "Stock ticker symbol, e.g. AAPL, MSFT",
                        },
                        "currency": {
                            "type": "string",
                            "enum": ["USD", "EUR", "GBP"],
                            "description": "Currency for the price",
                        },
                    },
                    "required": ["ticker"],
                    "additionalProperties": False,
                },
                "strict": True,
            },
        }
    ]

    # Stub tool implementation
    def get_stock_price(ticker: str, currency: str = "USD") -> dict:
        # In production: call a financial data API
        return {"ticker": ticker, "price": 189.50, "currency": currency}

    TOOL_MAP = {"get_stock_price": get_stock_price}

    messages = [{"role": "user", "content": "What is Apple's current stock price in USD?"}]

    # First call — LLM decides to use a tool
    response = client.chat.completions.create(model="gpt-4o", messages=messages, tools=TOOLS)
    msg = response.choices[0].message

    if msg.tool_calls:
        messages.append(msg)  # add assistant message with tool_calls

        for tool_call in msg.tool_calls:
            fn_name = tool_call.function.name
            fn_args = json.loads(tool_call.function.arguments)

            print(f"Calling: {fn_name}({fn_args})")
            result = TOOL_MAP[fn_name](**fn_args)

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(result),
            })

    # Second call — LLM generates final answer with tool result in context
    final = client.chat.completions.create(model="gpt-4o", messages=messages)
    print(final.choices[0].message.content)
    ```
  </TabItem>
  <TabItem label="Rust" icon="rust">
    ```rust
    // function_calling.rs — tool schema definition and dispatch
    use serde_json::{json, Value};
    use reqwest::Client;

    fn build_tools() -> Value {
        json!([{
            "type": "function",
            "function": {
                "name": "get_stock_price",
                "description": "Get the current stock price for a ticker symbol",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "ticker": {
                            "type": "string",
                            "description": "Stock ticker symbol, e.g. AAPL, MSFT"
                        },
                        "currency": {
                            "type": "string",
                            "enum": ["USD", "EUR", "GBP"],
                            "description": "Currency for the price"
                        }
                    },
                    "required": ["ticker"],
                    "additionalProperties": false
                },
                "strict": true
            }
        }])
    }

    // Stub tool implementation
    fn get_stock_price(ticker: &str, currency: &str) -> Value {
        json!({"ticker": ticker, "price": 189.50, "currency": currency})
    }

    fn dispatch_tool(name: &str, args: &Value) -> Value {
        match name {
            "get_stock_price" => {
                let ticker = args["ticker"].as_str().unwrap_or("");
                let currency = args["currency"].as_str().unwrap_or("USD");
                get_stock_price(ticker, currency)
            }
            _ => json!({"error": format!("Unknown tool: {}", name)}),
        }
    }

    #[tokio::main]
    async fn main() -> Result<(), Box<dyn std::error::Error>> {
        let client = Client::new();
        let tools = build_tools();

        let mut messages = vec![
            json!({"role": "user", "content": "What is Apple's current stock price in USD?"})
        ];

        // First call — LLM may respond with a tool_calls field
        let response = client
            .post("https://api.openai.com/v1/chat/completions")
            .header("Authorization", format!("Bearer {}", std::env::var("OPENAI_API_KEY")?))
            .json(&json!({"model": "gpt-4o", "messages": messages, "tools": tools}))
            .send().await?.json::<Value>().await?;

        let msg = &response["choices"][0]["message"];
        messages.push(msg.clone());

        if let Some(tool_calls) = msg["tool_calls"].as_array() {
            for tool_call in tool_calls {
                let fn_name = tool_call["function"]["name"].as_str().unwrap();
                let fn_args: Value = serde_json::from_str(
                    tool_call["function"]["arguments"].as_str().unwrap()
                )?;

                println!("Calling: {}({:?})", fn_name, fn_args);
                let result = dispatch_tool(fn_name, &fn_args);

                messages.push(json!({
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": result.to_string()
                }));
            }

            // Second call — LLM generates final answer
            let final_resp = client
                .post("https://api.openai.com/v1/chat/completions")
                .header("Authorization", format!("Bearer {}", std::env::var("OPENAI_API_KEY")?))
                .json(&json!({"model": "gpt-4o", "messages": messages}))
                .send().await?.json::<Value>().await?;

            println!("{}", final_resp["choices"][0]["message"]["content"].as_str().unwrap_or(""));
        }

        Ok(())
    }
    ```
  </TabItem>
</Tabs>

:::note[Tool schema design matters]
The `description` field of each tool and each parameter is what the LLM reads to decide *when* and *how* to use it. Write descriptions like documentation for a human — precise, with examples. Vague descriptions lead to wrong tool selection.
:::

---

## Context Windows vs Context Management

These two terms are related but distinct:

- **Context window**: A hard technical limit — the maximum number of tokens the model can process in a single call (input + output combined).
- **Context management**: A design strategy — how you decide *what* to put in that limited window.

| | Context Window | Context Management |
|---|---|---|
| **What it is** | Hardware/model limit | Your engineering problem |
| **Who controls it** | Model provider | You |
| **Size (2026)** | 128K–2M tokens | As much as fits usefully |
| **When it matters** | Very long conversations, large documents | Always |

### The trap: bigger is not always better

Modern models support 1M+ token context windows. The temptation is to dump everything in. This causes:

1. **Attention dilution**: The model's attention is spread across all tokens. Important information in a very large context is often ignored or misattributed — this is the "lost-in-the-middle" problem.
2. **Cost explosion**: You pay per token. 1M tokens per call at $5/1M = $5 per LLM call.
3. **Latency**: Larger contexts take longer to process.

The better approach: **retrieval over stuffing**. Instead of putting everything in context, retrieve only the relevant snippets for the current step. This is the core idea behind RAG, covered in [Module 5](/agentic-ai/06-memory-systems-rag).

---

## Working Memory vs Long Context

This is the final and most important mental model for agent designers:

| | Working Memory | Long Context |
|---|---|---|
| **Analogy** | RAM | Hard disk |
| **In agents** | Current context window | Retrieved memory / knowledge base |
| **Speed** | Instant (already in context) | Requires retrieval step |
| **Limit** | Hard cap (128K–2M tokens) | Effectively unlimited |
| **Best for** | Active reasoning, current task state | Background knowledge, history |

**Working memory** is what's in the prompt right now — the user's request, tool results, agent state. The model can "think about" everything in working memory with full attention.

**Long context** refers to information stored *outside* the active prompt — in vector databases, document stores, or episodic memory systems — that gets selectively retrieved when needed.

```
┌─────────────────────────────────────────────────────────┐
│  WORKING MEMORY (context window — active right now)     │
│  • System prompt (PAG)                                  │
│  • Current user request                                 │
│  • Last 3 tool results                                  │
│  • Relevant memory snippet (retrieved)   ←── pulled in  │
└─────────────────────────────────────────────────────────┘
                          ▲
                          │ retrieve relevant chunk
                          │
┌─────────────────────────────────────────────────────────┐
│  LONG CONTEXT (external memory — NOT in the prompt)     │
│  • Past conversation sessions (episodic memory)         │
│  • Knowledge base / documentation (semantic memory)     │
│  • User preferences / history (procedural memory)       │
└─────────────────────────────────────────────────────────┘
```

The art of agent design is deciding **which information belongs in working memory at which step** — not just "make the context bigger."

---

## Module 1 Complete — What You Now Know

You've covered all four articles in the Foundations module. Here's the capability stack you've built:

```
1.1  What Is an Agent          →  precise definition, pipelines vs agents, when not to build one
1.2  Cognitive Architecture    →  world model, goal-driven behavior, the 5-phase loop
1.3  Inference-Time Compute    →  thinking tokens, o1/R1, cost-accuracy tradeoff
1.4  Modern LLM Primitives     →  PAG, structured outputs, function calling, context management
```

→ **Next module**: [Internal Agent Architecture](/agentic-ai/03-agent-architecture) — we go inside the agent and build each component separately.
