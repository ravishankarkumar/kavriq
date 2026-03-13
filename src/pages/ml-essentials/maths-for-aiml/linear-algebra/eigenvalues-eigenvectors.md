---
title: Eigenvalues and Eigenvectors
description: Understanding eigenvalues and eigenvectors for ML
layout: ../../../../layouts/TutorialPage.astro
---

# Eigenvalues and Eigenvectors

Eigenvalues and eigenvectors are fundamental concepts in linear algebra with deep applications in machine learning, such as **PCA (Principal Component Analysis)**, dimensionality reduction, and stability analysis of algorithms.

---

## Definition

For a square matrix $A \in \mathbb{R}^{n \times n}$, a nonzero vector $v$ is an **eigenvector** if:

$$
A v = \lambda v
$$

where $\lambda$ is the corresponding **eigenvalue**.

- $v$ gives a direction that is unchanged (up to scaling) by $A$.  
- $\lambda$ tells how much the vector is stretched or shrunk.  

::: info Explanation of Eigenvalues & Eigenvectors
Think of a matrix $A$ as a transformation. Most vectors change direction when transformed, but **eigenvectors keep their direction**, only scaling by $\lambda$.  
- In ML: PCA finds eigenvectors of the covariance matrix → principal directions of data variance.  
:::

---

## Mini Example

$$
A =
\begin{bmatrix}
2 & 0 \\
0 & 3
\end{bmatrix}
$$

If $v = [1, 0]^T$, then:  
$$
Av =
\begin{bmatrix}
2 & 0 \\
0 & 3
\end{bmatrix}
\begin{bmatrix}
1 \\
0
\end{bmatrix}
=
\begin{bmatrix}
2 \\
0
\end{bmatrix}
= 2v
$$

So $v$ is an eigenvector with eigenvalue $\lambda = 2$.  

Similarly, $[0,1]^T$ is an eigenvector with eigenvalue $\lambda = 3$.

---

## Characteristic Equation

Eigenvalues are found by solving:

$$
\det(A - \lambda I) = 0
$$

This yields an $n$-degree polynomial in $\lambda$. Each solution is an eigenvalue.

---

## Hands-on with Python and Rust

::: code-group

```python [Python]
import numpy as np

A = np.array([[2, 0], [0, 3]])

# Eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(A)

print("Eigenvalues:", eigenvalues)
print("Eigenvectors:\n", eigenvectors)
```

```rust [Rust]
use ndarray::array;
use ndarray::Array2;
use ndarray_linalg::Eig;

fn main() {
    let a: Array2<f64> = array![
        [2.0, 0.0],
        [0.0, 3.0]
    ];

    // Eigen decomposition
    let (eigenvalues, eigenvectors) = a.eig().unwrap();

    println!("Eigenvalues: {:?}", eigenvalues);
    println!("Eigenvectors:\n{:?}", eigenvectors);
}
```

:::

---

## Connection to ML

- **PCA** → uses eigenvectors of covariance matrix to find directions of maximum variance.  
- **Spectral clustering** → uses eigenvalues of Laplacian matrices.  
- **Stability analysis** → eigenvalues determine convergence rates of iterative methods.  

---

## Next Steps

Continue to [Singular Value Decomposition (SVD)](/ml-essentials/maths-for-aiml/linear-algebra/svd).
