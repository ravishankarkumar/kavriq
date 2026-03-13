---

title: The Tool Manager
description: How AI agents discover, validate, and reliably use tools
layout: ../../../layouts/TutorialPage.astro
-------------------------------------------

import { Tabs, TabItem, Aside } from '@astrojs/starlight/components';

<Aside type="note" title="Module Context">
This article is part of **Module 2 — Internal Agent Architecture**.

In the previous article we examined the **Planner / Reasoner**, which converts goals into sequences of actions.

Now we examine the **Tool Manager**, the subsystem responsible for enabling agents to interact with external capabilities such as APIs, databases, and code execution environments.

</Aside>

# The Tool Manager

Large language models are powerful reasoning engines.

But by themselves, they are extremely limited.

They cannot:

* access real-time information
* execute code
* query databases
* interact with the filesystem
* call external APIs

To overcome these limitations, modern agent systems provide **tools**.

Tools extend the agent's capabilities beyond pure language generation.

Examples include:

* web search APIs
* database queries
* code execution environments
* file readers
* mathematical solvers

The component responsible for managing these capabilities is the **Tool Manager**.

---

# Why Tools Matter

Without tools, LLMs are limited to **their training data**.

With tools, agents can interact with the **live world**.

Example task:

> “What is the current weather in Tokyo?”

An LLM cannot answer this accurately using only training data.

But with a weather API tool:

```text
LLM → call_weather_api("Tokyo")
```

The agent receives **real-time information**.

This transforms the system from a chatbot into a **software operator**.

---

# The Role of the Tool Manager

The Tool Manager performs three key responsibilities.

```
Tool Discovery
      ↓
Tool Selection
      ↓
Tool Validation & Execution
```

More specifically, it manages:

* tool registration
* tool selection
* schema validation
* error handling
* retries
* execution policies

---

# Tool Discovery

Before an agent can use a tool, it must **know that the tool exists**.

Tool discovery is the process of making tools visible to the agent.

Most systems represent tools using **structured metadata**.

Example:

```json
{
  "name": "web_search",
  "description": "Search the web for information",
  "parameters": {
    "query": "string"
  }
}
```

This metadata becomes part of the **LLM prompt**.

Example prompt fragment:

```text
You can use the following tools:

1. web_search(query)
2. calculator(expression)
3. database_query(sql)
```

The LLM then decides **which tool to call**.

---

# Tool Registration

Tools are typically registered when the agent runtime starts.

<Tabs>

<TabItem label="Python" icon="seti:python">

```python
tools = {
    "web_search": web_search_tool,
    "calculator": calculator_tool,
    "read_file": read_file_tool
}
```

</TabItem>

<TabItem label="Rust" icon="seti:rust">

```rust
let mut tools = HashMap::new();

tools.insert("web_search".to_string(), web_search_tool);
tools.insert("calculator".to_string(), calculator_tool);
tools.insert("read_file".to_string(), read_file_tool);
```

</TabItem>

</Tabs>

These tools become available to the **planner and reasoning engine**.

---

# Tool Selection

Once tools are registered, the agent must decide **which tool to use**.

This decision is usually made by the **LLM itself**.

Example reasoning:

```text
Thought: I need current weather data
Action: weather_api("Tokyo")
```

The Tool Manager interprets this output and selects the appropriate tool.

Example structured action:

```json
{
  "tool": "web_search",
  "arguments": {
    "query": "Tokyo weather today"
  }
}
```

The runtime parses this output and routes the request to the correct tool.

---

# Tool Selection Strategies

Different agent frameworks implement different strategies.

| Strategy             | Description          |
| -------------------- | -------------------- |
| LLM selection        | Model decides tool   |
| Rule-based selection | Runtime selects tool |
| Hybrid selection     | Combination of both  |

Most modern agents use **LLM-driven selection**, because it allows flexible reasoning.

---

# Schema Validation

When an agent calls a tool, it must provide **correct arguments**.

This is where **schemas** become critical.

A schema defines:

* required parameters
* parameter types
* allowed values

Example schema:

```json
{
  "name": "weather_api",
  "parameters": {
    "city": {
      "type": "string",
      "required": true
    }
  }
}
```

Before executing the tool, the Tool Manager validates the request.

---

# Why Schema Validation Matters

Without validation, agents may produce invalid calls.

Example invalid output:

```json
{
  "tool": "weather_api",
  "arguments": {
    "temperature": "Tokyo"
  }
}
```

The schema validator detects the problem.

```
Expected parameter: city
Received parameter: temperature
```

The system can then:

* reject the call
* request correction
* retry with clarification

---

# Implementing Schema Validation

<Tabs>

<TabItem label="Python" icon="seti:python">

```python
from jsonschema import validate

schema = {
    "type": "object",
    "properties": {
        "city": {"type": "string"}
    },
    "required": ["city"]
}

validate(instance={"city": "Tokyo"}, schema=schema)
```

</TabItem>

<TabItem label="Rust" icon="seti:rust">

```rust
use jsonschema::JSONSchema;

let compiled = JSONSchema::compile(&schema_json).unwrap();

let result = compiled.validate(&instance_json);
```

</TabItem>

</Tabs>

This ensures tools are called with **correct parameters**.

---

# Tool Reliability

External tools are often unreliable.

Failures may include:

* API downtime
* network timeouts
* rate limits
* invalid responses

Therefore the Tool Manager must implement **robust reliability strategies**.

---

# Retry Strategies

One common strategy is **automatic retry**.

Example:

```
Attempt 1 → API timeout  
Attempt 2 → retry after delay  
Attempt 3 → success
```

Retries should use **exponential backoff**.

Example delays:

```
1 second
2 seconds
4 seconds
8 seconds
```

This prevents overwhelming the external service.

---

# Example Retry Logic

<Tabs>

<TabItem label="Python" icon="seti:python">

```python
import time

def retry_call(tool, args, retries=3):

    for i in range(retries):

        try:
            return tool(**args)

        except Exception:

            time.sleep(2 ** i)

    raise RuntimeError("Tool failed after retries")
```

</TabItem>

<TabItem label="Rust" icon="seti:rust">

```rust
fn retry_call<F, T>(tool: F, retries: u32) -> Result<T, String>
where
    F: Fn() -> Result<T, String>,
{
    for i in 0..retries {
        match tool() {
            Ok(v) => return Ok(v),
            Err(_) => std::thread::sleep(std::time::Duration::from_secs(2_u64.pow(i))),
        }
    }

    Err("Tool failed".into())
}
```

</TabItem>

</Tabs>

---

# Tool Timeouts

Agents must also enforce **execution time limits**.

Example policy:

```
web_search → timeout after 5 seconds
database_query → timeout after 10 seconds
code_execution → timeout after 30 seconds
```

This prevents agents from getting stuck waiting for slow services.

---

# Observability for Tool Usage

Production agents must track tool usage.

Useful metrics include:

* tool call frequency
* error rates
* latency
* success rate

Example log entry:

```json
{
  "tool": "web_search",
  "duration_ms": 420,
  "success": true
}
```

These logs are essential for **debugging agent behavior**.

---

# The Tool Manager as the Agent's Interface to the World

In many ways, tools are what make agents **powerful**.

Without tools:

```
Agent = reasoning engine
```

With tools:

```
Agent = autonomous software operator
```

The Tool Manager acts as the **gateway between reasoning and action**.

---

# Looking Ahead

In this article we explored the **Tool Manager**, which enables agents to interact with external systems.

We examined:

* tool discovery and registration
* tool selection strategies
* schema validation
* reliability and retry mechanisms

In the next article we will examine the **Execution Engine**, which is responsible for actually running tool calls, APIs, and code in a safe and deterministic environment.

→ Continue to **2.6 — The Execution Engine**
