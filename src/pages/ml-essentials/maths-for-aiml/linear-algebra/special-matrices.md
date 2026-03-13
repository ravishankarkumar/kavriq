---
title: Special Matrices - Identity, Diagonal, Orthogonal
description: Special matrices in ML – identity, diagonal, and orthogonal matrices
layout: ../../../../layouts/TutorialPage.astro
---

# Special Matrices: Identity, Diagonal, Orthogonal

Some matrices have special properties that make them particularly important in machine learning and linear algebra. In this lesson, we'll cover **identity matrices**, **diagonal matrices**, and **orthogonal matrices**.


## Identity Matrix

The **identity matrix** $I_n$ is a square matrix with ones on the diagonal and zeros elsewhere:

$$
I_n =
\begin{bmatrix}
1 & 0 & 0 \\
0 & 1 & 0 \\
0 & 0 & 1
\end{bmatrix}
$$

Property:  
$$
AI = IA = A
$$

for any matrix $A$ of compatible dimensions.

::: info Explanation of Identity
The identity matrix is like the number 1 for matrices. Multiplying by $I$ leaves the matrix unchanged.  
- In ML: shows up in regularization (e.g., $X^T X + \lambda I$ in Ridge regression).  
:::


## Diagonal Matrix

A **diagonal matrix** has nonzero entries only on its main diagonal:

$$
D =
\begin{bmatrix}
d_1 & 0 & 0 \\
0 & d_2 & 0 \\
0 & 0 & d_3
\end{bmatrix}
$$

Property: multiplying a vector scales each coordinate by the corresponding diagonal element.

**Mini Example:**  
$$
D =
\begin{bmatrix}
2 & 0 \\
0 & 3
\end{bmatrix},
\quad
x =
\begin{bmatrix}
4 \\
5
\end{bmatrix}
\quad \Rightarrow \quad
Dx =
\begin{bmatrix}
8 \\
15
\end{bmatrix}
$$

::: info Explanation of Diagonal
Diagonal matrices represent **scaling transformations**.  
- In ML: eigenvalues often appear in diagonal matrices after decomposition (PCA, SVD).  
:::


## Orthogonal Matrix

A **square matrix** $Q$ is orthogonal if:

$$
Q^T Q = Q Q^T = I
$$

This means:  
- Rows (and columns) of $Q$ are **orthonormal vectors** (unit length and mutually perpendicular).  
- $Q^{-1} = Q^T$.

**Mini Example:**  
$$
Q =
\begin{bmatrix}
0 & 1 \\
-1 & 0
\end{bmatrix}
$$

Here, $Q^T Q = I$.

::: info Explanation of Orthogonal
Orthogonal matrices preserve lengths and angles. They represent **rotations and reflections**.  
- In ML: orthogonal transformations are used in PCA, whitening, and optimization stability.  
:::


## Hands-on with Python and Rust

::: code-group

```python [Python]
import numpy as np

# Identity matrix
I = np.eye(3)

# Diagonal matrix
D = np.diag([2, 3])

# Orthogonal matrix (rotation by 90 degrees)
Q = np.array([[0, 1], [-1, 0]])

print("Identity matrix:\n", I)
print("Diagonal matrix:\n", D)
print("Orthogonal matrix:\n", Q)
print("Check Q^T Q = I:\n", Q.T.dot(Q))
```

```rust [Rust]
use ndarray::{array, Array2};
use ndarray_linalg::Norm;

fn main() {
    // Identity matrix
    let i: Array2<f64> = Array2::eye(3);

    // Diagonal matrix
    let d: Array2<f64> = array![[2.0, 0.0], [0.0, 3.0]];

    // Orthogonal matrix (rotation by 90 degrees)
    let q: Array2<f64> = array![[0.0, 1.0], [-1.0, 0.0]];
    let qtq = q.t().dot(&q);

    println!("Identity matrix:\n{:?}", i);
    println!("Diagonal matrix:\n{:?}", d);
    println!("Orthogonal matrix:\n{:?}", q);
    println!("Q^T Q =\n{:?}", qtq);
}
```

:::



## Connection to ML

- **Identity** → regularization, basis for “do nothing” transformations.  
- **Diagonal** → scaling features, eigenvalues, covariance structure.  
- **Orthogonal** → rotations, PCA, numerical stability in optimization.  


## Next Steps

Continue to [Rank, Determinant, and Inverses](/ml-essentials/maths-for-aiml/linear-algebra/rank-determinant-inverses).
