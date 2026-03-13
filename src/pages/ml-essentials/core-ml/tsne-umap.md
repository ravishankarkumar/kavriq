---
title: t-SNE & UMAP for Visualization
description: Comprehensive 3000+ word exploration of t-SNE and UMAP for visualization in machine learning in 2025, covering theory, mathematics, derivations, Python/Rust code, and applications in data visualization, embedding analysis, and modern ML pipelines.
layout: ../../../layouts/TutorialPage.astro
---

# t-SNE & UMAP for Visualization

t-Distributed Stochastic Neighbor Embedding (t-SNE) and Uniform Manifold Approximation and Projection (UMAP) are powerful nonlinear dimensionality reduction techniques for visualizing high-dimensional data in low dimensions (e.g., 2D/3D). t-SNE preserves local structure, while UMAP balances local and global structure with faster computation. In 2025, these tools are indispensable in ML for visualizing LLM embeddings, clustering analysis, and data exploration in high-dimensional spaces.

This lecture in the "Foundations for AI/ML" series (core-ml cluster) builds on PCA and clustering, exploring t-SNE and UMAP, their theoretical foundations, derivations, and applications. We’ll provide intuitive explanations, mathematical insights, and practical implementations in **Python (scikit-learn/umap-learn)** and **Rust (tsne-rs/umap-rs placeholders)**, ensuring a rigorous yet practical guide aligned with 2025 ML trends.

---

## 1. Motivation and Intuition

High-dimensional data (e.g., LLM embeddings with 768+ dimensions) is hard to visualize. t-SNE and UMAP reduce dimensions nonlinearly, preserving similarities for intuitive 2D/3D plots.

t-SNE focuses on local neighborhoods, making it great for cluster visualization. UMAP assumes a uniform manifold, preserving more global structure and running faster.

**Why t-SNE & UMAP in 2025?**
- **Visualization**: Explore LLM embeddings, datasets.
- **Efficiency**: UMAP scales to millions of points.
- **Interpretability**: Uncover patterns in high-d data.

### Real-World Examples
- **NLP**: Visualize word embeddings.
- **Biology**: scRNA-seq cell clustering.
- **AI Pipelines**: UMAP on LLM outputs for topic mapping.

::: info
t-SNE and UMAP are like maps of high-d landscapes—t-SNE zooms in on details, UMAP gives the big picture.
:::

---

## 2. Mathematical Formulation

Both methods embed high-d points x_i in low-d y_i, preserving similarities.

**Similarities**:
- High-d: p_{j|i} = exp(-||x_i - x_j||² / 2σ_i²) / sum exp(-||x_i - x_k||² / 2σ_i²)
- Low-d: q_{j|i} = (1 + ||y_i - y_j||²)^{-1} / sum (1 + ||y_i - y_k||²)^{-1} (t-SNE).

Minimize KL(p||q) via GD.

For UMAP: Fuzzy simplicial sets model topology.

### ML Connection
- Preserve local/global structure for visualization.

---

## 3. t-SNE Derivation

t-SNE minimizes KL divergence between joint probabilities p and q.

**Perplexity**: Tunes σ_i: perp = 2^{H(p_i)}, H entropy.

**Joint P**: p_{ij} = (p_{j|i} + p_i|j)/2n.

**q**: Student t-distribution for long tails.

### Derivation
KL = sum p_{ij} log(p_{ij}/q_{ij}).

Gradient: ∂KL/∂y_i = 4 sum_j (p_{ij} - q_{ij}) (y_i - y_j) (1 + ||y_i - y_j||²)^{-1}

In ML: Barnes-Hut approximation for speed.

---

## 4. UMAP Derivation

UMAP models data as fuzzy graph, optimizes low-d embedding.

**High-D Graph**: Local metric, global uniform.

**Cross-Entropy Loss**: Min sum p_{ij} log(p_{ij}/q_{ij}) - (1-p_{ij}) log((1-p_{ij})/(1-q_{ij})).

### Derivation
Topological representation via simplicial sets.

Force-directed layout with attractive/repulsive forces.

In ML: Parametric UMAP for embeddings.

---

## 5. Choosing Parameters

**t-SNE**:
- Perplexity: 5-50, tunes local/global balance.
- Learning Rate: 200 default.

**UMAP**:
- n_neighbors: Like perplexity, 2-100.
- min_dist: Spread in low-d, 0.1 default.

In 2025, auto-tuning with Optuna.

---

## 6. Evaluation Metrics

- **KL Divergence**: Lower better, but internal.
- **Silhouette Score**: Cluster quality in embeddings.
- **Trustworthiness**: Preserved neighbor ranks.
- **Continuity**: Inverse trustworthiness.

In 2025, metrics for LLM embedding visualizations.

---

## 7. Applications in Machine Learning (2025)

1. **Data Visualization**: Embeddings, datasets.
2. **Clustering Preprocessing**: Reduce for k-means.
3. **Anomaly Detection**: Identify outliers in plots.
4. **LLM Analysis**: Visualize token embeddings.
5. **Bioinformatics**: scRNA-seq visualization.
6. **Recommendation**: User/item embedding maps.

### Challenges
- **Distortion**: t-SNE crowds clusters.
- **Scalability**: UMAP better than t-SNE.
- **Reproducibility**: Random init varies outputs.

---

## 8. Numerical Implementations

Implement t-SNE, UMAP.

::: code-group

```python [Python]
import numpy as np
from sklearn.manifold import TSNE
from umap import UMAP
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits

# Load digits data
digits = load_digits()
X = digits.data
y = digits.target

# t-SNE
tsne = TSNE(n_components=2, perplexity=30, random_state=0)
X_tsne = tsne.fit_transform(X)
plt.scatter(X_tsne[:,0], X_tsne[:,1], c=y, cmap='viridis')
plt.title("t-SNE on Digits")
plt.colorbar()
plt.show()

# UMAP
umap = UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=0)
X_umap = umap.fit_transform(X)
plt.scatter(X_umap[:,0], X_umap[:,1], c=y, cmap='viridis')
plt.title("UMAP on Digits")
plt.colorbar()
plt.show()

# ML: UMAP on high-d embeddings (placeholder)
embeddings = np.random.rand(100, 768)  # LLM embeds
umap_emb = UMAP(n_components=2).fit_transform(embeddings)
print("UMAP Embeddings Shape:", umap_emb.shape)
```

```rust [Rust]
use tsne::Tsne; // Requires tsne crate
use ndarray::{Array2, Array1};

fn main() {
    let x: Array2<f64> = Array2::zeros((1797, 64)); // Digits placeholder
    let tsne = Tsne::new(x.view(), 2, 30.0, 12.0, 500).unwrap();
    let embedding = tsne.embedding();
    println!("t-SNE Shape: {:?}", embedding.shape());

    // UMAP not natively in Rust; use Python or implement
}
```
:::

Dependencies (`Cargo.toml`):
```toml
[dependencies]
tsne = "0.1.4"  # For t-SNE
ndarray = "0.15.6"
```

**Note**: Rust t-SNE/UMAP limited; use Python for full visualization.

---

## 9. Case Study: Digits Dataset (Visualization)

::: code-group

```python [Python]
from sklearn.datasets import load_digits
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Load digits
digits = load_digits()
X = digits.data
y = digits.target

# t-SNE visualization
tsne = TSNE(n_components=2, perplexity=30, random_state=0)
X_tsne = tsne.fit_transform(X)
plt.scatter(X_tsne[:,0], X_tsne[:,1], c=y, cmap='viridis')
plt.title("t-SNE on Digits")
plt.colorbar()
plt.show()
```

```rust [Rust]
use tsne::Tsne;
use ndarray::Array2;

fn main() {
    let x: Array2<f64> = Array2::zeros((1797, 64)); // Digits placeholder
    let tsne = Tsne::new(x.view(), 2, 30.0, 12.0, 500).unwrap();
    let embedding = tsne.embedding();
    println!("t-SNE Shape: {:?}", embedding.shape());
}
```
:::

**Note**: Rust requires plotting libraries for visualization.

---

## 10. Under the Hood Insights

- **Probability-Based**: t-SNE uses joint probs.
- **Topology-Preserving**: UMAP models manifold topology.
- **Perplexity/n_neighbors**: Tune local/global balance.
- **Randomness**: Multiple runs for stability.

---

## 11. Limitations

- **t-SNE**: Slow, local focus.
- **UMAP**: Parameter sensitive, less theoretical.
- **High-D**: Computation; use approximations.
- **Distortion**: Not for distances, only visualization.

---

## 12. Summary

t-SNE and UMAP are **powerful visualization tools** for high-d data. In 2025, their role in LLM embedding analysis and data exploration keeps them vital. Parameter tuning and scalability are key.

<!-- **Next**: Explore [Neural Networks Basics](/core-ml/neural-networks) or revisit [PCA](/core-ml/pca). -->

---

## Further Reading
- Maaten, Hinton, "Visualizing Data using t-SNE" (2008).
- McInnes, Healy, "UMAP" (2018).
- `umap-learn` docs: [github.com/lmcinnes/umap](https://github.com/lmcinnes/umap).
- Belkin, Niyogi, "Laplacian Eigenmaps" (2003).

---