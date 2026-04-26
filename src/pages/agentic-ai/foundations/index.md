---
title: Agentic AI Foundations
description: A 12-part core curriculum for learning the building blocks of modern agentic AI systems.
layout: ../../../layouts/TutorialPage.astro
---

# Agentic AI Foundations

Agentic AI Foundations is Kavriq's 12-part core curriculum for learning the building blocks of modern agent systems.

This series starts from first principles and moves toward advanced concepts: agent loops, cognitive architecture, planning, tools, memory, multi-agent systems, guardrails, evaluation, runtime internals, and capstone projects.

If you want the systems-first production perspective after learning the components, continue into [Engineering Agent Systems](/agentic-ai/engineering-agent-systems).

> **This is a living roadmap.** Articles are written and published one by one. Items listed without a link are planned and coming. Linked items are published and ready to read.

---

## Prerequisite

- [Setting up Environment](/agentic-ai/foundations/prerequisite/setting-up)

## Module 1 — Core Concepts

The cognitive and computational foundations of modern agent systems.

- [What is an Agent?](/agentic-ai/foundations/core-concepts/what-is-an-agent)
- [Cognitive Architecture of Agents](/agentic-ai/foundations/core-concepts/cognitive-architecture)
- [The Inference-Time Compute Revolution](/agentic-ai/foundations/core-concepts/inference-time-compute)
- [Modern LLM Primitives](/agentic-ai/foundations/core-concepts/modern-llm-primitives)

---

## Module 2 — Internal Agent Architecture

The internal components that make an AI system behave like an autonomous agent.

- [The Anatomy of an Agent](/agentic-ai/foundations/agent-architecture/the-anatomy-of-an-agent)
- [The Perception Layer](/agentic-ai/foundations/agent-architecture/the-perception-layer)
- [Working Memory and the Scratchpad](/agentic-ai/foundations/agent-architecture/working-memory-and-the-scratchpad)
- [The Planner / Reasoner](/agentic-ai/foundations/agent-architecture/the-planner-reasoner)
- [The Tool Manager](/agentic-ai/foundations/agent-architecture/tool-manager)
- [The Execution Engine](/agentic-ai/foundations/agent-architecture/execution-engine)
- [The Observation Processor](/agentic-ai/foundations/agent-architecture/observation-processor)
- [Reflection and Termination](/agentic-ai/foundations/agent-architecture/reflection-and-termination)

---

## Module 3 — Planning Systems

Techniques that allow agents to solve complex tasks through multi-step reasoning.

- [Why Planning Matters](/agentic-ai/foundations/planning-systems/why-planning-matters)
- [ReAct: Reason + Act](/agentic-ai/foundations/planning-systems/react)
- [Chain-of-Thought Planning](/agentic-ai/foundations/planning-systems/chain-of-thought)
- [Tree-of-Thought Reasoning](/agentic-ai/foundations/planning-systems/tree-of-thought)
- [Execution Graphs](/agentic-ai/foundations/planning-systems/execution-graphs)
- [Building Agents with LangGraph - Python](/agentic-ai/foundations/planning-systems/langgraph)

---

## Module 4 — Tool Use & Protocols

How agents interact with APIs, databases, and external systems.

- [Why Tools Make Agents Powerful](/agentic-ai/foundations/tool-use-protocols/why-tools)
- [Designing Reliable Tools](/agentic-ai/foundations/tool-use-protocols/designing-reliable-tools)
- [The Model Context Protocol (MCP)](/agentic-ai/foundations/tool-use-protocols/mcp)

---

## Module 5 — Memory Systems & RAG

How agents store knowledge and retrieve information across interactions.

- [The Memory Hierarchy of Agents](/agentic-ai/foundations/memory-systems-rag/memory-hierarchy)
- [Episodic Memory](/agentic-ai/foundations/memory-systems-rag/episodic-memory)
- [Semantic Memory](/agentic-ai/foundations/memory-systems-rag/semantic-memory)
- [Procedural Memory](/agentic-ai/foundations/memory-systems-rag/procedural-memory)
- [Agentic RAG](/agentic-ai/foundations/memory-systems-rag/agentic-rag)
- [Multi-Hop Retrieval](/agentic-ai/foundations/memory-systems-rag/multi-hop)

---

## Module 6 — Multi-Agent Systems

Architectures where multiple agents collaborate to solve problems.

- [Why Multi-Agent Systems Exist](/agentic-ai/foundations/multi-agent/why-multi-agent)
- [Manager-Worker Coordination](/agentic-ai/foundations/multi-agent/manager-worker-pattern)
- [Handoff Pattern (Swarm)](/agentic-ai/foundations/multi-agent/handoff-swarm)
- [Debate Pattern](/agentic-ai/foundations/multi-agent/debate-pattern)
- [Agent-to-Agent Communication (A2A)](/agentic-ai/foundations/multi-agent/a2a-agent-to-agent)

---

## Module 7 — Computer Use & Vision

Agents that interact with software interfaces and visual environments.

- [Computer Use Agents](/agentic-ai/foundations/computer-use-vision/computer-use-agents)
- [GUI Navigation](/agentic-ai/foundations/computer-use-vision/gui-navigation)
- [Visual Grounding](/agentic-ai/foundations/computer-use-vision/visual-grounding)

---

## Module 8 — Guardrails & Safety

Designing safe and reliable agent systems.

- [Prompt Injection Attacks](/agentic-ai/foundations/guardrails-safety/prompt-injection)
- [Tool Permission Systems](/agentic-ai/foundations/guardrails-safety/tool-permissions)
- [Human-in-the-Loop](/agentic-ai/foundations/guardrails-safety/human-in-the-loop)
- [Sandboxing Agent Execution](/agentic-ai/foundations/guardrails-safety/sandboxing)

---

## Module 9 — Evaluation & Metrics

How to measure agent performance and reliability.

- [Why Agent Evaluation Is Hard](/agentic-ai/foundations/evaluation-metrics/why-evaluation-is-hard)
- [LLM-as-a-Judge](/agentic-ai/foundations/evaluation-metrics/llm-judge)
- [Trajectory Evaluation](/agentic-ai/foundations/evaluation-metrics/trajectory-eval)
- [Building Evaluation Pipelines](/agentic-ai/foundations/evaluation-metrics/building-eval-pipelines)

---

## Module 10 — High-Performance Engineering

Engineering techniques for scalable, high-performance agent systems.

- [The Small Model Strategy](/agentic-ai/foundations/high-perf-engineering/small-model)
- [Observability for Agents](/agentic-ai/foundations/high-perf-engineering/observability)

---

## Module 11 — Agent Internals

Understanding how agents actually work by building a minimal runtime from scratch.

- [Why Build Your Own Agent Runtime](/agentic-ai/foundations/agent-internals/why-build-agent-runtime)
- [Designing a Simple Agent State Machine](/agentic-ai/foundations/agent-internals/state-machine)
- [Implementing Tool Calling & MCP Integration](/agentic-ai/foundations/agent-internals/tool-calling)
- [Adding Time-Travel Debugging](/agentic-ai/foundations/agent-internals/time-travel)
- [A Production-Ready 300-Line Agent Runtime](/agentic-ai/foundations/agent-internals/minimal-runtime)

---

## Module 12 — Capstone Projects

Real-world applications built using agentic architectures.

- [The Computer-Use Researcher](/agentic-ai/foundations/capstone-projects/computer-use-researcher)
- [The Multi-Agent Coding Pipeline](/agentic-ai/foundations/capstone-projects/multi-agent-coding)
- [The Privacy-First Local Butler](/agentic-ai/foundations/capstone-projects/privacy-first-butler)

---

## What You Will Learn

By the end of this guide you will understand how to build:

- Autonomous AI agents
- Tool-using reasoning systems
- Multi-agent collaboration architectures
- Production-grade agent infrastructure
- Privacy-first local AI assistants

This series provides a complete technical foundation for modern agentic AI systems.

