---
title: Positive Semi-Definite Matrices and Covariance
description: Understanding PSD matrices and covariance in ML
layout: ../../../../layouts/TutorialPage.astro
---

# Positive Semi-Definite Matrices and Covariance

Positive Semi-Definite (PSD) matrices and covariance matrices play a central role in machine learning. They describe variance, correlation, and structure in data, forming the backbone of PCA, kernel methods, and Gaussian models.

---

## Positive Semi-Definite (PSD) Matrices

A symmetric matrix $A \in \mathbb{R}^{n \times n}$ is **positive semi-definite (PSD)** if:

$$
x^T A x \geq 0 \quad \forall x \in \mathbb{R}^n
$$

- If $x^T A x > 0$ for all nonzero $x$, the matrix is **positive definite (PD)**.  
- Eigenvalues of a PSD matrix are always non-negative.  

::: info Explanation of PSD
The quadratic form $x^T A x$ represents “energy” or variance along direction $x$.  
- PSD means the matrix never produces negative variance.  
- In ML: covariance matrices and kernel matrices are PSD.  
:::

---

## Covariance Matrix

For a dataset $X \in \mathbb{R}^{m \times n}$ (m samples, n features), the **covariance matrix** is:

$$
\Sigma = \frac{1}{m-1}(X - \mu)^T (X - \mu)
$$

where $\mu$ is the mean vector of features.

- Diagonal entries = variances of features.  
- Off-diagonal entries = covariances between features.  

**Mini Example:**  
For data points $[1,2], [2,3], [3,4]$:

- Feature 1 variance = variance of [1,2,3].  
- Feature 2 variance = variance of [2,3,4].  
- Covariance = how feature 1 and feature 2 vary together.

---

## Properties of Covariance Matrices

- Symmetric.  
- Positive semi-definite.  
- Encodes feature relationships (correlation structure).  

---

## Applications in ML

- **PCA**: Eigen-decomposition of covariance matrix finds principal components.  
- **Gaussian Models**: Multivariate normal distribution defined by mean vector and covariance matrix.  
- **Kernels**: Kernel (Gram) matrices are PSD by construction.  

---

## Hands-on with Python and Rust

::: code-group

```python [Python]
import numpy as np

# Dataset: 3 samples, 2 features
X = np.array([[1, 2], [2, 3], [3, 4]])

# Covariance matrix (rows as observations, columns as features)
cov_matrix = np.cov(X, rowvar=False)

# Check PSD via eigenvalues
eigenvalues = np.linalg.eigvals(cov_matrix)

print("Covariance matrix:\n", cov_matrix)
print("Eigenvalues (should be >= 0):", eigenvalues)
```

```rust [Rust]
use ndarray::{array, Array2};
use ndarray_stats::CorrelationExt;
use ndarray_linalg::Eig;

fn main() {
    // Dataset: 3 samples, 2 features
    let x: Array2<f64> = array![
        [1.0, 2.0],
        [2.0, 3.0],
        [3.0, 4.0]
    ];

    // Covariance matrix
    let cov = x.cov(0.0).unwrap();

    // Eigenvalues to check PSD
    let eigenvalues = cov.eig().unwrap().0;

    println!("Covariance matrix:\n{:?}", cov);
    println!("Eigenvalues (should be >= 0): {:?}", eigenvalues);
}
```

:::

---

## Connection to ML

- PSD matrices guarantee **valid variance/covariance structures**.  
- Covariance underlies dimensionality reduction (PCA), Gaussian models, and kernels.  
- Understanding covariance helps diagnose multicollinearity in regression.  

---

## Next Steps

Continue to [Linear Transformations and Geometry](/ml-essentials/maths-for-aiml/linear-algebra/linear-transformations).
