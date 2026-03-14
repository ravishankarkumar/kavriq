---
title: Singular Value Decomposition (SVD)
description: Understanding Singular Value Decomposition for ML
layout: ../../../../layouts/TutorialPage.astro
---

# Singular Value Decomposition (SVD)

The **Singular Value Decomposition (SVD)** is one of the most powerful tools in linear algebra. It is widely used in machine learning for **dimensionality reduction, noise reduction, recommendation systems, and data compression**.

---

## Definition

Any real matrix $A \in \mathbb{R}^{m \times n}$ can be decomposed as:

$$
A = U \Sigma V^T
$$

- $U \in \mathbb{R}^{m \times m}$: orthogonal matrix (left singular vectors).  
- $\Sigma \in \mathbb{R}^{m \times n}$: diagonal matrix with non-negative values (singular values).  
- $V \in \mathbb{R}^{n \times n}$: orthogonal matrix (right singular vectors).  

::: info Explanation of SVD
- **Singular values** represent the importance (variance captured) of each dimension.  
- **$U$** gives the directions in the original space.  
- **$V$** gives the directions in feature space.  
- Truncating $\Sigma$ gives low-rank approximations of $A$.  
:::

---

## Mini Example

Let:

$$
A =
\begin{bmatrix}
1 & 0 \\
0 & 1 \\
0 & 0
\end{bmatrix}
$$

Performing SVD:

- $U$ contains orthonormal basis vectors in row space.  
- $\Sigma$ has singular values $[1, 1]$.  
- $V$ contains orthonormal basis vectors in column space.  

---

## Applications in ML

- **PCA**: PCA is essentially SVD on the centered data matrix.  
- **Latent Semantic Analysis (LSA)**: Uses SVD in NLP for dimensionality reduction.  
- **Recommendation Systems**: Matrix factorization with SVD finds latent features of users and items.  
- **Noise Reduction**: Keep only top $k$ singular values to approximate data.  

---

## Hands-on with Python and Rust

::: code-group

```python [Python]
import numpy as np

A = np.array([[1, 0], [0, 1], [0, 0]])

# Perform SVD
U, S, Vt = np.linalg.svd(A)

print("U =\n", U)
print("Singular values =", S)
print("V^T =\n", Vt)
```

```rust [Rust]
use ndarray::array;
use ndarray::Array2;
use ndarray_linalg::SVD;

fn main() {
    let a: Array2<f64> = array![
        [1.0, 0.0],
        [0.0, 1.0],
        [0.0, 0.0]
    ];

    // Perform SVD
    let (u, s, vt) = a.svd(true, true).unwrap();

    println!("U = {:?}", u.unwrap());
    println!("Singular values = {:?}", s);
    println!("V^T = {:?}", vt.unwrap());
}
```

:::

---

## Connection to ML

- **Data compression** → keep only top singular values.  
- **Noise filtering** → discard small singular values.  
- **Feature extraction** → reduced representations (PCA, LSA).  

---

## Next Steps

Continue to [Positive Semi-Definite Matrices and Covariance](/ml-essentials/maths-for-aiml/linear-algebra/psd-covariance).
