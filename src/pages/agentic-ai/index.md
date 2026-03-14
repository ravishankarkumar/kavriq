---
title: A Comprehensive Guide to Agentic AI
layout: ../../layouts/TutorialPage.astro
---

Whether you are just beginning your journey into Agentic AI or already have some exposure to it, this guide is designed to help you build a deep and structured understanding of the field. We start from first principles and gradually move toward advanced concepts, modern techniques, and the latest developments in the Agentic AI ecosystem.

Agentic AI is one of the most important shifts in modern artificial intelligence. It is not just a passing trend. The ideas, tools, and design patterns you learn here will remain valuable as the field continues to evolve.

To make the journey easier, this guide is organized as a 12-part series. By following it step by step, you will develop a solid foundation in Agentic AI — from core concepts to the advanced systems and architectures that matter today.

---

## Prerequisite
- [Setting up Environment](/agentic-ai/00-prerequisite/a-setting-up)

## Module 1 — Foundations
The cognitive and computational foundations of modern agent systems.

- [What is an Agent?](/agentic-ai/01-foundations/a-what-is-an-agent)
- [Cognitive Architecture of Agents](/agentic-ai/01-foundations/b-cognitive-architecture)
- [The Inference-Time Compute Revolution](/agentic-ai/01-foundations/c-inference-time-compute)
- [Modern LLM Primitives](/agentic-ai/01-foundations/d-modern-llm-primitives)

---

## Module 2 — Internal Agent Architecture
The internal components that make an AI system behave like an autonomous agent.

- [The Anatomy of an Agent](/agentic-ai/02-agent-architecture/a-the-anatomy-of-an-agent)
- [The Perception Layer](/agentic-ai/02-agent-architecture/b-the-perception-layer)
- [Working Memory and the Scratchpad](/agentic-ai/02-agent-architecture/c-working-memory-and-the-scratchpad)
- [The Planner / Reasoner](/agentic-ai/02-agent-architecture/d-the-planner-reasoner)
- [The Tool Manager](/agentic-ai/02-agent-architecture/e-tool-manager)
- [The Execution Engine](/agentic-ai/02-agent-architecture/f-execution-engine)
- [The Observation Processor](/agentic-ai/02-agent-architecture/g-observation-processor)
- [Reflection and Termination](/agentic-ai/02-agent-architecture/h-reflection-and-termination)

---

## Module 3 — Planning Systems
Techniques that allow agents to solve complex tasks through multi-step reasoning.

- [Why Planning Matters](/agentic-ai/03-planning-systems/a-why-planning-matters)
- [ReAct: Reason + Act](/agentic-ai/03-planning-systems/b-react)
- [Chain-of-Thought Planning](/agentic-ai/03-planning-systems/c-chain-of-thought)
- [Tree-of-Thought Reasoning](/agentic-ai/03-planning-systems/d-tree-of-thought)
- [Execution Graphs](/agentic-ai/03-planning-systems/e-execution-graphs)
- [Building Agents with LangGraph](/agentic-ai/03-planning-systems/f-langgraph)

---

## Module 4 — Tool Use & Protocols
How agents interact with APIs, databases, and external systems.

- [Why Tools Make Agents Powerful](/agentic-ai/04-tool-use-protocols/a-why-tools)
- [Designing Reliable Tools](/agentic-ai/04-tool-use-protocols/b-designing-reliable-tools)
- [The Model Context Protocol (MCP)](/agentic-ai/04-tool-use-protocols/c-mcp)
- [Building an MCP Server in Rust](/agentic-ai/04-tool-use-protocols/d-mcp-rust)

---

## Module 5 — Memory Systems & RAG
How agents store knowledge and retrieve information across interactions.

- [The Memory Hierarchy of Agents](/agentic-ai/05-memory-systems-rag/a-memory-hierarchy)
- [Episodic Memory](/agentic-ai/05-memory-systems-rag/b-episodic-memory)
- [Semantic Memory](/agentic-ai/05-memory-systems-rag/c-semantic-memory)
- [Procedural Memory](/agentic-ai/05-memory-systems-rag/d-procedural-memory)
- [Agentic RAG](/agentic-ai/05-memory-systems-rag/e-agentic-rag)
- [Multi-Hop Retrieval](/agentic-ai/05-memory-systems-rag/f-multi-hop)

---

## Module 6 — Multi-Agent Systems
Architectures where multiple agents collaborate to solve problems.

- [Why Multi-Agent Systems Exist](/agentic-ai/06-multi-agent/a-why-multi-agent)
- [Manager–Worker Coordination](/agentic-ai/06-multi-agent/b-centralised-coordination)
- [Handoff Pattern (Swarm)](/agentic-ai/06-multi-agent/c-handoff-swarm)
- [Debate Pattern](/agentic-ai/06-multi-agent/d-debate-pattern)
- [Agent-to-Agent Communication (A2A)](/agentic-ai/06-multi-agent/e-a2a-agent-to-agent)

---

## Module 7 — Computer Use & Vision
Agents that interact with software interfaces and visual environments.

- [Computer Use Agents](/agentic-ai/07-computer-use-vision/a-computer-use-agents)
- [GUI Navigation](/agentic-ai/07-computer-use-vision/b-gui-navigation)
- [Visual Grounding](/agentic-ai/07-computer-use-vision/c-visual-grounding)

---

## Module 8 — Guardrails & Safety
Designing safe and reliable agent systems.

- [Prompt Injection Attacks](/agentic-ai/08-guardrails-safety/a-prompt-injection)
- [Tool Permission Systems](/agentic-ai/08-guardrails-safety/b-tool-permissions)
- [Human-in-the-Loop](/agentic-ai/08-guardrails-safety/c-human-in-the-loop)
- [Sandboxing Agent Execution](/agentic-ai/08-guardrails-safety/d-sandboxing)

---

## Module 9 — Evaluation & Metrics
How to measure agent performance and reliability.

- [Why Agent Evaluation Is Hard](/agentic-ai/09-evaluation-metrics/a-why-evaluation-is-hard)
- [LLM-as-a-Judge](/agentic-ai/09-evaluation-metrics/b-llm-judge)
- [Trajectory Evaluation](/agentic-ai/09-evaluation-metrics/c-trajectory-eval)
- [Building Evaluation Pipelines](/agentic-ai/09-evaluation-metrics/d-building-eval-pipelines)

---

## Module 10 — High-Performance Engineering
Engineering techniques for scalable, high-performance agent systems.

- [The Small Model Strategy](/agentic-ai/10-high-perf-engineering/a-small-model)
- [Using Rust for Agent Infrastructure](/agentic-ai/10-high-perf-engineering/b-small-model-using-rust)
- [Observability for Agents](/agentic-ai/10-high-perf-engineering/c-observability)

---

## Module 11 — Agent Internals
Building a minimal agent runtime and understanding the mechanics.

- [Why Build Your Own Agent Runtime](/agentic-ai/11-agent-internals/a-why-build-agent-runtime)
- [Designing a Simple Agent State Machine](/agentic-ai/11-agent-internals/b-state-machine)
- [Implementing Tool Calling](/agentic-ai/11-agent-internals/c-tool-calling)
- [Adding Time-Travel Debugging](/agentic-ai/11-agent-internals/d-time-travel)
- [A 300-Line LangGraph Alternative](/agentic-ai/11-agent-internals/e-langgraph-alternative)

---

## Module 12 — Capstone Projects
Real-world applications built using agentic architectures.

- [The Computer-Use Researcher](/agentic-ai/12-capstone-projects/a-computer-use-researcher)
- [The Multi-Agent Coding Pipeline](/agentic-ai/12-capstone-projects/b-multi-agent-coding)
- [The Privacy-First Local Butler](/agentic-ai/12-capstone-projects/c-privacy-first-butler)

---

## What You Will Learn

By the end of this guide you will understand how to build:

- Autonomous AI agents  
- Tool-using reasoning systems  
- Multi-agent collaboration architectures  
- Production-grade agent infrastructure  
- Privacy-first local AI assistants

This series provides a **complete technical foundation for modern agentic AI systems**.