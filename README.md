# KAVRIQ

KAVRIQ is a deep-dive resource for AI internals, transformers, inference, embeddings, and modern AI systems.

Built by [Ravi Shankar Kumar](https://www.linkedin.com/in/ravi-shankar-a725b0225/), AI Engineer at Salesforce.

---

## What This Is

A technical blog and structured learning resource for AI and ML engineers. The goal is to be the single best resource for understanding how modern AI systems actually work — from the mathematics up through production infrastructure.

Content is written manually, from first principles, with no hand-waving. Code samples default to Python so the material can go deeper without splitting attention across languages.

---

## Published Blog Posts

Eight articles published so far, covering core LLM internals:

- **Understanding Tokenizers in AI** — BPE, WordPiece, SentencePiece, Unigram; how ChatGPT, Grok, and Gemini tokenize text; why tokenization affects cost, performance, and multilingual quality
- **Why Embeddings Matter** — what embeddings are, how they work in LLMs, historical evolution from Word2Vec to contextual embeddings, applications in RAG and semantic search
- **What LLMs Do at Inference** — the full inference pipeline: tokenization → embeddings → prefill → KV cache → autoregressive decode → decoding strategies; FlashAttention, quantization, speculative decoding
- **KV Cache Explained** — deep dive into the key-value cache mechanism, memory implications, quantization and compression, MQA/GQA variants
- **Transformers in AI** — the architecture from first principles: attention, positional encoding, encoder/decoder variants, BERT vs GPT vs T5, 2025 efficiency improvements
- **Top-k vs Nucleus Sampling** — how LLMs choose the next token; greedy vs beam search vs top-k vs top-p; empirical comparison and when to use each
- **GPU vs TPU** — hardware comparison for AI workloads; architecture differences, pros/cons, real-world use cases
- **Why RAG Exists** — the problems RAG solves (hallucination, knowledge cutoff, domain specificity); how retrieval-augmented generation works

---

## Content Series

### [Mathematics & Machine Learning](/ml-essentials)
A complete curriculum covering all the math needed for ML (linear algebra, calculus, probability, information theory) and machine learning itself (classical algorithms, neural networks, transformers, diffusion models, RL). 22 modules, each with a hands-on project. Living roadmap — articles published one by one.

### [AI Engineering in Practice](/ai-engineering)
The engineering discipline of building, deploying, and operating AI systems. Covers GPU/CUDA, distributed training, data pipelines, vector databases, RAG, LLM APIs, MLOps, model serving, and AI safety/ethics. 19 modules across 5 parts. Living roadmap.

### [Agentic AI](/agentic-ai)
A complete guide to building autonomous AI agents — from agent architecture and planning systems to tool use, memory, multi-agent coordination, guardrails, and evaluation. 12 modules with capstone projects.

### [Interview Prep](/interview-prep)
Interview preparation for AI Engineer and ML Engineer roles. Q&A sections for ML fundamentals, transformers, LLMs, and agents. Full system design walkthroughs. Coding patterns in Python. Living roadmap.

---

## Tech Stack

- [Astro](https://astro.build/) — static site framework
- [TailwindCSS](https://tailwindcss.com/) — styling
- [KaTeX](https://katex.org/) — LaTeX math rendering
- [Pagefind](https://pagefind.app/) — static search
- [Expressive Code](https://expressive-code.com/) — syntax highlighting
- TypeScript throughout

---

## Running Locally

```bash
npm install
npm run dev
```

Build for production:

```bash
npm run build
```

The build runs `astro check`, builds the site, generates the Pagefind search index, and copies it to `public/`.

---

## Writing Content

Blog posts go in `src/data/blog/` as `.md` or `.mdx` files with this frontmatter:

```yaml
---
title: Your Post Title
description: A short description
pubDatetime: 2026-01-01T00:00:00Z
tags: [transformers, inference]
---
```

Tutorial/series pages go in `src/pages/` as `.md` or `.mdx` files using the `TutorialPage` layout:

```yaml
---
title: Page Title
layout: ../../layouts/TutorialPage.astro
---
```

LaTeX is supported in both `.md` and `.mdx` files:
- Inline: `$E = mc^2$`
- Block: `$$\sum_{i=0}^{n} x_i$$`

---

## Environment Variables

```bash
# optional — adds Google Search Console verification meta tag
PUBLIC_GOOGLE_SITE_VERIFICATION=your-value
```

---

## Commands

| Command | Action |
|---|---|
| `npm run dev` | Start dev server at `localhost:4321` |
| `npm run build` | Build production site to `./dist/` |
| `npm run preview` | Preview production build locally |
| `npm run sync` | Generate Astro TypeScript types |
| `npm run format` | Format with Prettier |
| `npm run lint` | Lint with ESLint |
