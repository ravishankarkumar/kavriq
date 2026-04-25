---
title: "What LLMs Do at Inference: A Deep Dive Under the Hood"
description: "A step-by-step, reference-backed explanation of what happens during LLM inference: tokenization, embeddings, prefill & decode phases, KV caching, decoding strategies, bottlenecks and optimizations like quantization, FlashAttention and speculative decoding."
author: "Ravi Shankar"
date: 2025-12-05
draft: false
slug: /articles/what-llms-do-at-inference
keywords: ["LLM inference","transformer","tokenization","KV cache","flashattention","quantization","speculative decoding","vLLM"]
readingTime: 9
cover: /assets/images/llm-inference-cover.png
canonical: https://kavriq.com/articles/what-llms-do-at-inference
toc: true
# sidebar: auto
# aliases:
#   - /what-llms-do-at-inference
#   - /llm-inference
pubDatetime: 2022-09-25T15:20:35Z
modDatetime: 2026-01-09T15:00:15.170Z
tags:
  - AI
  - LLMs
  - Inference
  - Transformers
  - KV Cache
  - AI Engineering
---



In the rapidly evolving world of artificial intelligence, Large Language Models (LLMs) like GPT-4, Llama 3, and Grok have become the backbone of everything from chatbots to content creation tools. But while we marvel at their ability to generate human-like text, code, or even creative stories, few stop to wonder: what exactly happens when you hit "send" on that prompt? This is the magic of *inference*—the phase where a trained LLM springs into action to produce outputs. Unlike training, which involves massive datasets and weeks of computation to learn patterns, inference is all about real-time application. It's efficient, forward-only, and powers the AI experiences we interact with daily.

In this article, we'll peel back the layers of LLM inference, exploring the step-by-step process, the underlying mechanics, challenges, and cutting-edge optimizations. Whether you're an AI enthusiast building your knowledge base or a developer optimizing models for production, understanding inference is key to unlocking AI's full potential. We'll draw from reliable sources in the AI community to ensure accuracy, and by the end, you'll have a clear picture of what's happening "under the hood." Let's dive in.

## You can react on this post on the LinkedIn thread

<iframe src="https://www.linkedin.com/embed/feed/update/urn:li:share:7402735326626312193?collapsed=1" height="230" width="504" frameborder="0" allowfullscreen="" title="Embedded post"></iframe>

## The Foundations: How LLMs Are Built for Inference

To grasp inference, we first need a quick refresher on LLM architecture. Most modern LLMs are based on the Transformer model, introduced in the seminal 2017 paper "Attention is All You Need" by Vaswani et al. Transformers revolutionized NLP by ditching recurrent layers in favor of attention mechanisms, allowing parallel processing of sequences.

At their core, decoder-only Transformers (like those in GPT series) consist of stacked layers, each with two main components: multi-head self-attention and feed-forward networks. The model processes text as sequences of *tokens*—not words, but smaller units like subwords or characters. This setup enables the model to capture long-range dependencies in data, making it ideal for generation tasks.

Inference differs starkly from training. Training is compute-intensive, involving forward and backward passes to update billions of parameters using gradients. Inference, however, is a forward pass only: no updates, just predictions. It's memory-bound rather than compute-bound for much of the process, especially during generation, which explains why optimizations focus on speed and efficiency.

## The Inference Pipeline: Step by Step

LLM inference unfolds in a structured pipeline, transforming your text prompt into a coherent response. This can be broken into preprocessing, the core computation phases (prefill and decode), and post-processing. Here's how it works.

### Step 1: Tokenization – Turning Text into Numbers

Everything starts with your input prompt, say, "Write a poem about AI." The model can't understand raw text, so it undergoes *tokenization*. Using algorithms like Byte Pair Encoding (BPE), the text is split into tokens—subword units from the model's vocabulary. For instance, "AI" might be one token, while rarer words get broken down.

Each token is mapped to a unique integer ID. This creates a sequence of numbers, like [50256, 198, 198, 198, ...], ready for the model. Tokenizers are model-specific; mismatch them, and you'll get garbage outputs. This step is crucial because it defines how the model "sees" the world—vocabulary size can range from 50,000 to over 100,000 tokens.

### Step 2: Embeddings and Positional Encoding

Next, these token IDs are converted into dense vectors via an *embedding layer*. Think of embeddings as high-dimensional representations (e.g., 4096 dimensions for larger models) that capture semantic meaning. Words with similar meanings end up closer in this vector space.

But sequences have order, so we add *positional encodings*. These could be fixed sinusoidal functions or learned values, like Rotary Position Embeddings (RoPE), which inject position awareness without disrupting the embeddings. Now, the input is a matrix of vectors, primed for the Transformer.

### Step 3: The Prefill Phase – Parallel Processing the Prompt

Inference kicks off with the *prefill phase*, where the entire prompt is processed in parallel. This is compute-intensive and GPU-friendly, as all tokens can be handled at once.

In each Transformer layer:
- **Multi-Head Self-Attention**: The input matrix is projected into Query (Q), Key (K), and Value (V) matrices. Attention scores are computed as the dot product of Q and K, scaled and softmaxed to get weights. These weights multiply V to produce context-aware outputs. Multiple heads allow capturing different relationships.
- **Feed-Forward Network**: A simple MLP (multi-layer perceptron) applies non-linear transformations.

During prefill, the model builds a *KV cache*—storing K and V matrices for every token at each layer. This cache is gold for efficiency, as it avoids recomputing past contexts later. The output here is the initial hidden states, and the Time to First Token (TTFT) measures how long this takes—critical for user experience.

### Step 4: The Decode Phase – Autoregressive Generation

Now comes the heart of inference: *autoregressive generation*. The model predicts tokens one by one, appending each to the sequence and feeding it back in. This is the *decode phase*, memory-bound because it processes sequentially.

For each new token:
- Compute only its Q (using the latest embedding).
- Fetch K and V for all prior tokens from the KV cache.
- Run attention and feed-forward as before.
- Output logits (probabilities over the vocabulary) via a final linear layer and softmax.

The highest-probability token is selected (or sampled), detokenized back to text, and the process repeats until an end-of-sequence token, max length, or stop word. Inter-Token Latency (ITL) tracks speed here. The KV cache grows with each token, ballooning memory use for long outputs.

### Step 5: Decoding Strategies – Beyond Greedy Selection

Not all generations are straightforward. *Greedy decoding* picks the top-probability token each time, but it can lead to repetitive or bland outputs. Alternatives include:
- **Beam Search**: Explores multiple paths (beams) in parallel, keeping the most promising based on cumulative probability. Great for accuracy in tasks like translation.
- **Top-K Sampling**: Samples from the top K probable tokens, adding randomness.
- **Nucleus (Top-P) Sampling**: Samples from tokens whose cumulative probability exceeds P, balancing diversity and coherence.

These strategies trade off quality, speed, and creativity.

### Step 6: Detokenization and Output

Finally, generated token IDs are mapped back to text using the tokenizer's vocabulary. The result? Your AI-generated poem or answer.

## Challenges in LLM Inference

Inference isn't without hurdles. Sequential decoding creates latency bottlenecks—long responses can take seconds or minutes. Memory demands soar with model size; a 70B-parameter model like Llama 2 might need 140GB in FP16, limiting deployment on consumer hardware.

Costs add up too: running inference at scale requires expensive GPUs, and energy consumption is a growing concern. Hallucinations (fabricated facts) and token limits (e.g., 128K in GPT-4) can degrade quality, especially in retrieval-augmented generation (RAG) setups. Scalability for concurrent users risks overload, and immature tooling fragments ecosystems.

## Optimizations: Making Inference Faster and Cheaper

The AI community has rallied with clever fixes. Here's a rundown:

### Memory and Compute Optimizations
- **KV Caching**: As mentioned, this reuses computations, slashing redundancy. Prefix caching extends it across sessions.
- **Quantization**: Compresses weights from FP16 to INT8 or 4-bit, reducing model size by 4x with tools like GPTQ. Minimal accuracy loss, huge wins on edge devices.
- **Pruning**: Removes low-importance weights, creating sparse models that run faster on sparsity-aware hardware.
- **Distillation**: Trains a smaller "student" model to mimic a large "teacher," like DistilBERT shrinking BERT by 40% while retaining 97% performance.

### Attention and Parallelism Tricks
- **FlashAttention**: Fuses operations to minimize GPU memory reads/writes, speeding up by 2-4x via tiling.
- **PagedAttention**: Manages KV cache in non-contiguous blocks, reducing waste and enabling larger batches (used in vLLM).
- **Multi-Query/Group-Query Attention (MQA/GQA)**: Shares K/V across heads, cutting memory by up to 75% with little quality drop.
- **Model Parallelism**: Splits models across GPUs via tensor, pipeline, or sequence parallelism to handle giants like GPT-3.

### Advanced Techniques
- **Speculative Decoding**: A fast draft model guesses multiple tokens; the main model verifies in batches, parallelizing the sequential nature.
- **Batching and Streaming**: Groups requests for efficiency; streams tokens as generated for instant feedback.
- **Hardware Acceleration**: GPUs with Tensor Cores or TPUs shine here, with frameworks like ONNX Runtime optimizing graphs.

Emerging alternatives like Diffusion LLMs (dLLMs) generate entire responses in parallel via denoising, promising 10x speedups, though still nascent.

## Future Trends in LLM Inference

Looking ahead, inference is poised for breakthroughs. Edge inference on devices like smartphones will democratize AI, driven by quantization and efficient architectures. Multimodal models blending text, images, and audio will complicate but enrich inference. Open-source tools from Hugging Face and frameworks like BentoML or TrueFoundry are standardizing deployment.

Sustainability is key—optimizations must curb energy use. And as models grow (trillions of parameters?), hybrid approaches combining local and cloud inference could balance privacy and power.

## Wrapping Up: Why Inference Matters for AI's Future

Inference is the unsung hero of LLMs, turning trained knowledge into actionable intelligence. From tokenization to autoregressive magic, it's a symphony of math and engineering that powers our AI-driven world. But challenges like latency and cost remind us that optimization is ongoing. By leveraging techniques like KV caching and quantization, we're making AI faster, cheaper, and more accessible.



## References
This article draws from the following sources:

1. Vaswani et al., *Attention Is All You Need*, 2017 — https://arxiv.org/abs/1706.03762  
2. Understanding the LLM's inference | by Lathashree Harisha - Medium - https://lathashreeh.medium.com/understanding-the-llms-inference-36a767f98a83  
3. Understanding LLMs: A Comprehensive Overview from Training to ... - https://arxiv.org/html/2401.02038v2  
4. Understanding LLM Inference - by Alex Razvant - Neural Bits - https://multimodalai.substack.com/p/understanding-llm-inference  
5. LLM Inference Optimization Techniques | Clarifai Guide - https://www.clarifai.com/blog/llm-inference-optimization/  
6. Mastering LLM Techniques: Inference Optimization - https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/  
7. [PDF] Database Perspective on LLM Inference Systems - VLDB Endowment - https://www.vldb.org/pvldb/vol18/p5504-li.pdf  
8. How does LLM inference work? - BentoML - https://bentoml.com/llm/llm-inference-basics/how-does-llm-inference-work  
9. LLM Inference guide | Google AI Edge - https://ai.google.dev/edge/mediapipe/solutions/genai/llm_inference  
