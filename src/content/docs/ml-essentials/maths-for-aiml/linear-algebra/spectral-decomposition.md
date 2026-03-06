---
title: Spectral Decomposition & Applications
description: Understanding spectral decomposition, graph Laplacians, and PCA connections in ML
---

# Spectral Decomposition & Applications

Spectral decomposition is at the heart of many machine learning techniques.  
It connects eigenvalues, eigenvectors, and matrix diagonalization to practical applications like **PCA** and **spectral clustering**.

---

## 1. Spectral Decomposition of Symmetric Matrices

If $A$ is a real symmetric matrix ($A = A^T$), then it can be diagonalized as:

$$
A = Q \Lambda Q^T
$$

where:  
- $Q$ = orthogonal matrix of eigenvectors  
- $\Lambda$ = diagonal matrix of eigenvalues  

This is known as the **spectral theorem**.

::: info Why important?
- Guarantees orthogonal eigenvectors for symmetric matrices.  
- Forms the basis of PCA (covariance matrices are symmetric).  
:::

---

## 2. Graph Laplacians & Spectral Clustering

For a graph $G$ with adjacency matrix $W$ and degree matrix $D$, the **graph Laplacian** is:

$$
L = D - W
$$

- $L$ is symmetric and positive semi-definite.  
- Its eigenvectors capture graph structure.  

**Spectral Clustering:**  
1. Compute Laplacian $L$.  
2. Find the $k$ smallest eigenvectors.  
3. Treat them as features, cluster with k-means.  

This uses spectral decomposition to reveal community structure in graphs.

---

## 3. Links to PCA & Dimensionality Reduction

- **PCA**: Eigen-decomposition of covariance matrix.  
  - $\Sigma = \frac{1}{m} X^T X$ (symmetric PSD).  
  - Eigenvectors = principal components.  
  - Eigenvalues = variance explained.  

- **Manifold Learning**: Spectral methods generalize PCA to nonlinear settings (e.g., Laplacian eigenmaps).  

---

## Hands-on with Python and Rust

::: code-group

```python [Python]
import numpy as np
from sklearn.cluster import KMeans

# Example symmetric matrix
A = np.array([[2, 1], [1, 2]])

# Spectral decomposition
eigvals, eigvecs = np.linalg.eigh(A)
print("Eigenvalues:", eigvals)
print("Eigenvectors:\n", eigvecs)

# Simple spectral clustering example (tiny graph)
W = np.array([[0, 1, 0],
              [1, 0, 1],
              [0, 1, 0]])
D = np.diag(W.sum(axis=1))
L = D - W

eigvals, eigvecs = np.linalg.eigh(L)
print("\nGraph Laplacian Eigenvalues:", eigvals)

# Use 2 smallest eigenvectors as features
X_spec = eigvecs[:, :2]
labels = KMeans(n_clusters=2, random_state=42).fit_predict(X_spec)
print("Cluster labels:", labels)
```

```rust [Rust]
use ndarray::array;
use ndarray_linalg::Eigh;

fn main() {
    // Example symmetric matrix
    let a = array![[2.0, 1.0],
                   [1.0, 2.0]];

    // Spectral decomposition
    let (eigvals, eigvecs) = a.eigh(UPLO::Lower).unwrap();
    println!("Eigenvalues: {:?}", eigvals);
    println!("Eigenvectors:\n{:?}", eigvecs);

    // Graph Laplacian example (tiny graph)
    let w = array![
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 0.0]
    ];
    let d = w.sum_axis(ndarray::Axis(1)).into_diag();
    let l = &d - &w;

    let (lap_eigvals, lap_eigvecs) = l.eigh(UPLO::Lower).unwrap();
    println!("Graph Laplacian Eigenvalues: {:?}", lap_eigvals);
    println!("Eigenvectors:\n{:?}", lap_eigvecs);
}
```

:::

---

## Summary

- **Spectral decomposition** diagonalizes symmetric matrices.  
- **Graph Laplacians** enable spectral clustering.  
- **PCA** is an application of spectral decomposition to covariance matrices.  

<!-- --- -->

<!-- ## Next Steps -->

<!-- With this, we finish the **Advanced Linear Algebra** cluster.   -->
<!-- Next: [Functions and Limits](/maths-for-aiml/calculus/functions-limits) in the **Calculus module**. -->
