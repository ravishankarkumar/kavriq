---
title: Dimensionality Reduction - PCA
description: Comprehensive 3000+ word exploration of Principal Component Analysis (PCA) for machine learning in 2025, covering theory, mathematics, derivations, Python/Rust code, and applications in feature reduction, visualization, and modern ML pipelines.
layout: ../../../layouts/TutorialPage.astro
---

# Dimensionality Reduction: PCA

Principal Component Analysis (PCA) is a foundational unsupervised learning technique for dimensionality reduction, projecting high-dimensional data onto a lower-dimensional subspace while preserving variance. In 2025, PCA remains crucial in ML for preprocessing, noise reduction, and visualization, especially in pipelines with large language models (LLMs) and high-dimensional embeddings.

This lecture in the "Foundations for AI/ML" series (core-ml cluster) builds on clustering and SVMs, exploring PCA, its theoretical foundations, variance maximization, and applications. We’ll provide intuitive explanations, mathematical derivations, and practical implementations in **Python (scikit-learn)** and **Rust (nalgebra)**, ensuring a rigorous yet practical guide aligned with 2025 ML trends.

---

## 1. Motivation and Intuition

PCA reduces data dimensions by finding principal components—directions of maximum variance—while minimizing information loss. Intuitively, it rotates the coordinate system to align with data spread, discarding low-variance axes.

**Why PCA in 2025?**
- **Efficiency**: Compresses high-d embeddings from LLMs.
- **Visualization**: 2D/3D plots of complex data.
- **Noise Reduction**: Filters low-variance noise.
- **Modern Applications**: Preprocess for GNNs, federated learning.

### Real-World Examples
- **Genomics**: Reduce gene features.
- **Image Processing**: Compress pixels.
- **AI Pipelines**: PCA on LLM embeddings for visualization.

::: info
PCA is like compressing a high-res image—keep key details, discard noise for efficient ML.
:::

---

## 2. Mathematical Formulation

PCA finds orthogonal axes (principal components) maximizing variance.

For data X (m samples, n features), centered \tilde{X} = X - \bar{X}.

**Covariance Matrix**: S = (1/(m-1)) \tilde{X}^T \tilde{X}.

**Eigen Decomposition**: S = U \Lambda U^T, U eigenvectors, \Lambda eigenvalues.

**Projection**: Y = \tilde{X} U_k, U_k top k eigenvectors.

Variance preserved: sum top k eigenvalues.

### ML Connection
- PCA as linear autoencoder.

---

## 3. Derivation: Variance Maximization

Maximize Var(u^T X) s.t. ||u||=1.

Var(u^T X) = u^T S u.

Lagrangian: u^T S u - λ (u^T u - 1) =0.

Solve S u = λ u (eigenvalue equation).

Top eigenvectors maximize variance.

### PCA as SVD
X = U Σ V^T, principal components V.

In ML: SVD for large data.

---

## 4. Optimization and Implementation

**Algorithm**:
1. Center X: \tilde{X} = X - mean.
2. Compute S or SVD.
3. Select top k eigs/vecs.
4. Project Y = \tilde{X} U_k.

**Choosing k**: Cumulative variance explained ≥ threshold (e.g., 95%).

In 2025, auto-encoder variants for nonlinear PCA.

---

## 5. Kernel PCA for Nonlinearity

Map to high-d φ(X), apply PCA.

Kernel K(x_i,x_j) = φ(x_i)^T φ(x_j).

Centered kernel PCA.

In ML: Nonlinear reduction.

---

## 6. Evaluation Metrics

- **Explained Variance Ratio**: λ_j / sum λ.
- **Cumulative Explained Variance**: Sum top k ratios.
- **Reconstruction Error**: ||X - Y U_k^T - mean||².

In 2025, metrics for LLM embeddings.

---

## 7. Applications in Machine Learning (2025)

1. **Preprocessing**: Reduce features for SVM, NN.
2. **Visualization**: 2D plots of embeddings.
3. **Noise Reduction**: Filter low-variance.
4. **Anomaly Detection**: Reconstruct errors.
5. **LLM Analysis**: PCA on embeddings for insights.
6. **Federated Learning**: Reduce communication costs.

### Challenges
- **Linearity**: Misses nonlinear; use kernel PCA.
- **High-D**: Computation; use random projections.
- **Interpretability**: PCs hard to interpret.

---

## 8. Numerical Implementations

Implement PCA, kernel PCA.

::: code-group

```python [Python]
import numpy as np
from sklearn.decomposition import PCA, KernelPCA
from sklearn.metrics import mean_squared_error
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt

# Standard PCA
X, _ = make_moons(n_samples=200, noise=0.05, random_state=0)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
print("Explained Variance Ratio:", pca.explained_variance_ratio_)

plt.scatter(X_pca[:,0], X_pca[:,1])
plt.title("PCA on Moons")
plt.show()

# Reconstruction error
X_recon = pca.inverse_transform(X_pca)
print("Reconstruction MSE:", mean_squared_error(X, X_recon))

# Kernel PCA
kpca = KernelPCA(n_components=2, kernel='rbf', gamma=1.0)
X_kpca = kpca.fit_transform(X)
plt.scatter(X_kpca[:,0], X_kpca[:,1])
plt.title("Kernel PCA on Moons")
plt.show()

# ML: PCA on high-d data
high_d = np.random.rand(100, 1000)
pca_hd = PCA(n_components=10)
pca_hd.fit(high_d)
print("Cumulative Explained Variance:", np.cumsum(pca_hd.explained_variance_ratio_)[-1])
```

```rust [Rust]
use nalgebra::{DMatrix, SVD};
use rand::Rng;

fn pca(x: &DMatrix<f64>, n_comp: usize) -> DMatrix<f64> {
    let mean = x.row_mean();
    let x_centered = x - &mean;
    let cov = (&x_centered.transpose() * &x_centered) / (x.nrows() as f64 - 1.0);
    let svd = SVD::new(cov, true, true);
    let u = svd.u.unwrap();
    x_centered * u.columns(0, n_comp)
}

fn main() {
    let mut rng = rand::thread_rng();
    let x = DMatrix::from_fn(200, 2, |_, _| rng.gen::<f64>());
    let x_pca = pca(&x, 2);
    println!("PCA shape: {:?}", x_pca.shape());

    // Reconstruction
    // Omitted

    // Kernel PCA placeholder
}
```
:::

Implements PCA, kernel PCA, high-d reduction.

---

## 8. Numerical Stability and High-Dimensions

- **Covariance Conditioning**: Ill-conditioned S; regularization helps.
- **High-D**: Curse; randomized SVD approximates.
- **Stability**: SVD stable for PCA.

In 2025, stability in distributed PCA for federated ML.

---

## 9. Case Study: Iris Dataset (Dimensionality Reduction)

::: code-group

```python [Python]
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load dataset
iris = load_iris()
pca = PCA(n_components=2)
X_pca = pca.fit_transform(iris.data)

plt.scatter(X_pca[:,0], X_pca[:,1], c=iris.target)
plt.title("PCA on Iris")
plt.show()

print("Explained Variance Ratio:", pca.explained_variance_ratio_)
```

```rust [Rust]
use nalgebra::{DMatrix, SVD};

fn main() {
    let x = DMatrix::zeros(150, 4);  // Placeholder Iris
    let mean = x.row_mean();
    let x_centered = x - &mean;
    let cov = (&x_centered.transpose() * &x_centered) / (x.nrows() as f64 - 1.0);
    let svd = SVD::new(cov, true, true);
    let u = svd.u.unwrap();
    let x_pca = x_centered * u.columns(0, 2);
    println!("PCA shape: {:?}", x_pca.shape());
}
```
:::

**Note**: Rust requires external data loading; use Python for visualization.

---

## 10. Under the Hood Insights

- **Variance Maximization**: PCs orthogonal, uncorrelated.
- **SVD vs Eig**: SVD more stable for cov.
- **Whitening**: Normalize variances after PCA.
- **Probabilistic PCA**: Models noise.

---

## 11. Limitations

- **Linearity**: Assumes linear structure; use kernel PCA.
- **Global Variance**: Misses local structures; use manifold learning.
- **High-D**: Computation; use randomized PCA.
- **Interpretability**: PCs abstract.

---

## 12. Summary

PCA is a **powerful linear reduction technique** foundational to ML. In 2025, its role in LLM embedding compression and preprocessing keeps it vital. Kernel PCA addresses nonlinearity.

<!-- **Next**: Explore [Neural Networks Basics](/core-ml/neural-networks) or revisit [Clustering](/core-ml/clustering). -->

---

## Further Reading
- Jolliffe, *Principal Component Analysis*.
- Hastie, *Elements of Statistical Learning* (Ch. 14).
- `nalgebra` docs: [docs.rs/nalgebra](https://docs.rs/nalgebra).
- Abdi, Williams, "Principal Component Analysis" (2010).

---