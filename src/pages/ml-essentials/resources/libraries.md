---
title: Rust ML Libraries
description: Overview of Rust libraries for machine learning
layout: ../../../layouts/TutorialPage.astro
---
# Rust ML Libraries

Rust's machine learning (ML) ecosystem offers high-performance, memory-safe libraries for data processing, traditional ML, deep learning, and specialized tasks (e.g., NLP, graphs). This page overviews key libraries used in the AI/ML in Rust tutorial (`linfa`, `rust-bert`, `tch-rs`, `nalgebra`, `polars`, `petgraph`), their features, and tutorial applications (e.g., **House Prices**, **Sentiment Analysis**), with Python/C++ comparisons. Explore the Rust ML ecosystem at [arewelearningyet.com](https://www.arewelearningyet.com/).

## 1. Introduction to Rust's ML Ecosystem

Rust's memory safety and C++-like performance make it ideal for ML. Libraries like `polars` and `tch-rs` optimize CPU tasks, outperforming Python's `pandas`/`pytorch` by ~15–25%. Challenges include a smaller ecosystem and Rust's ownership model complexity.

## 2. Key Rust ML Libraries

### 2.1 `linfa`: Traditional Machine Learning
- **Overview**: Rust's `scikit-learn`, with classification, regression, clustering.
- **Use Cases**: **House Prices** (linear regression), **Customer Churn Prediction** (logistic regression).
- **Features**: Modular, `ndarray`-optimized, preprocessing support.
- **Comparison**: ~15% faster than `scikit-learn`, fewer algorithms.

### 2.2 `rust-bert`: Natural Language Processing
- **Overview**: Transformer models (e.g., BERT) for NLP tasks.
- **Use Cases**: **Sentiment Analysis** (classification), **NLP**.
- **Features**: `tch-rs` integration, tokenization, ~15% faster than `transformers`.
- **Comparison**: Fewer models, lightweight deployment.

### 2.3 `tch-rs`: Deep Learning
- **Overview**: PyTorch binding for CNNs, RNNs.
- **Use Cases**: **Image Classification** (CNN), **Time-Series Forecasting** (LSTM).
- **Features**: GPU/CPU support, ~10–20% faster than `pytorch` on CPU.
- **Comparison**: Flexible, complex GPU setup.

### 2.4 `nalgebra`: Linear Algebra
- **Overview**: Matrix operations for ML.
- **Use Cases**: **Recommendation System** (factorization), **Numerical Methods**.
- **Features**: BLAS/LAPACK, ~15% faster than `numpy`.
- **Comparison**: Lacks `numpy`'s statistical tools.

### 2.5 `polars`: Data Processing
- **Overview**: DataFrame library like `pandas`.
- **Use Cases**: All **Projects** for data preprocessing.
- **Features**: Lazy evaluation, ~25% faster than `pandas`.
- **Comparison**: Fewer visualization tools.

### 2.6 `petgraph`: Graph Processing
- **Overview**: Graph structures for ML.
- **Use Cases**: **Recommendation System** (GNNs), **Graph-based ML**.
- **Features**: Lightweight, ~20% faster than `networkx`.
- **Comparison**: Fewer algorithms.

## 3. Installation and Setup

Add to `Cargo.toml`:
```toml
[dependencies]
linfa = "0.7.1"
linfa-linear = "0.7.1"
linfa-trees = "0.7.1"
rust-bert = "0.23.0"
tch = "0.17.0"
nalgebra = "0.33.2"
polars = { version = "0.46.0", features = ["lazy"] }
petgraph = "0.6.5"
actix-web = "4.4.0"
serde = { version = "1.0", features = ["derive"] }
ndarray = "0.15.0"
rand = "0.8.5"
```
**Note**: Ensure `tch` version matches `rust-bert`'s requirements; check [github.com/guillaume-be/rust-bert](https://github.com/guillaume-be/rust-bert).
Run `cargo build`. Ensure Rust is installed via `rustup`.

## 4. Practical Considerations

- **Ecosystem Maturity**: Less mature than Python's, but `polars`, `tch-rs` are production-ready.
- **Performance**: Rust's safety and speed yield ~15–25% faster CPU tasks than Python.
- **Community**: Growing, with `linfa`, `rust-bert` contributions.

## Next Steps
Proceed to Recommended Reading or revisit Customer Churn Prediction project.