---
title: Matrix Factorizations in ML (LU, QR, Cholesky)
description: Understanding LU, QR, and Cholesky factorizations and their role in machine learning
layout: ../../../../layouts/TutorialPage.astro
---

# Matrix Factorizations in ML (LU, QR, Cholesky)

Matrix factorizations break down a matrix into simpler building blocks.  
They are not just abstract math — they are **workhorses of numerical linear algebra** that make solving systems, regression, and probabilistic ML efficient and stable.

In this lesson, we cover three fundamental factorizations:

- **LU Decomposition** → solving linear systems efficiently  
- **QR Decomposition** → numerical stability in least squares  
- **Cholesky Decomposition** → covariance matrices, Gaussian processes  

---

## 1. LU Decomposition

**Definition**: Any square matrix $A$ can (under certain conditions) be decomposed as:

$$
A = L U
$$

where:
- $L$ is a lower triangular matrix (ones on the diagonal)  
- $U$ is an upper triangular matrix  

This is extremely useful for solving systems of equations $Ax = b$:

1. Compute $LU = A$ once.  
2. Solve $Ly = b$ (forward substitution).  
3. Solve $Ux = y$ (back substitution).  

Much faster than computing $A^{-1}$.

::: info ML relevance
- Linear regression can involve solving $(X^T X)w = X^T y$. LU factorization speeds this up.  
- Appears in optimization routines and numerical solvers.  
:::

---

## 2. QR Decomposition

**Definition**: Any (rectangular) matrix $A$ can be factored as:

$$
A = QR
$$

where:  
- $Q$ is an orthogonal matrix ($Q^T Q = I$)  
- $R$ is an upper triangular matrix  

**Why useful?**  
Instead of solving $A^T A w = A^T y$ (which may be unstable if $A^T A$ is ill-conditioned), we can solve least squares via QR:

$$
Aw \approx y \quad \Rightarrow \quad Rw = Q^T y
$$

This is more numerically stable.

::: info ML relevance
- QR is widely used in **least squares regression** solvers.  
- Preferred when feature matrices are ill-conditioned (highly correlated features).  
:::

---

## 3. Cholesky Decomposition

**Definition**: A symmetric, positive-definite matrix $A$ can be decomposed as:

$$
A = L L^T
$$

where $L$ is lower triangular.

**Why useful?**  
- Efficient way to invert covariance matrices.  
- More efficient than LU for positive-definite systems.  

::: info ML relevance
- **Gaussian Processes (GPs)**: covariance kernel matrices are symmetric positive-definite → use Cholesky for efficient inference.  
- **Optimization**: Cholesky is used in second-order methods where Hessians are PSD.  
:::

---

## Hands-on with Python and Rust

::: code-group

```python [Python]
import numpy as np

# Example matrix
A = np.array([[4, 2], [2, 3]])

# LU decomposition
from scipy.linalg import lu
P, L, U = lu(A)
print("LU Decomposition:")
print("P=\n", P, "\nL=\n", L, "\nU=\n", U)

# QR decomposition
Q, R = np.linalg.qr(A)
print("\nQR Decomposition:")
print("Q=\n", Q, "\nR=\n", R)

# Cholesky decomposition
L = np.linalg.cholesky(A)
print("\nCholesky Decomposition:")
print("L=\n", L)
```

```rust [Rust]
use ndarray::array;
use ndarray_linalg::{LU, QR, Cholesky};

fn main() {
    let a = array![[4.0, 2.0],
                   [2.0, 3.0]];

    // LU decomposition
    let lu = a.clone().lu().unwrap();
    let (l, u) = (lu.l().to_owned(), lu.u().to_owned());
    println!("LU Decomposition:\nL=\n{:?}\nU=\n{:?}", l, u);

    // QR decomposition
    let qr = a.clone().qr().unwrap();
    let (q, r) = (qr.q().unwrap(), qr.r().unwrap());
    println!("QR Decomposition:\nQ=\n{:?}\nR=\n{:?}", q, r);

    // Cholesky decomposition
    let chol = a.cholesky().unwrap();
    println!("Cholesky Decomposition:\nL=\n{:?}", chol);
}
```

:::

---

## Summary

- **LU** → solve systems efficiently, appears in regression & optimization  
- **QR** → stable least squares solutions  
- **Cholesky** → Gaussian processes & covariance matrices  

---

## Next Steps

Continue to [Pseudo-Inverse & Ill-Conditioned Systems](/ml-essentials/maths-for-aiml/linear-algebra/pseudo-inverse).
