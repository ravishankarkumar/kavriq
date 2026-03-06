---
title: Subspaces and Basis
description: Understanding subspaces and basis for ML
---

# Subspaces and Basis

In linear algebra, a **subspace** is a smaller space that lives inside a vector space, and a **basis** is a minimal set of vectors that span a space. These concepts are essential in machine learning, where we often represent data in lower-dimensional subspaces (e.g., PCA).

---

## Subspaces

A **subspace** of $\mathbb{R}^n$ is a set of vectors that:

1. Contains the zero vector.  
2. Is closed under vector addition.  
3. Is closed under scalar multiplication.  

**Examples:**  
- The span of one vector $v$ is a line through the origin.  
- The span of two independent vectors is a plane through the origin.  

---

## Span

The **span** of a set of vectors $\{v_1, v_2, ..., v_k\}$ is the set of all linear combinations:

$$
\text{span}\{v_1, ..., v_k\} = \{a_1 v_1 + a_2 v_2 + ... + a_k v_k \mid a_i \in \mathbb{R}\}
$$

- If vectors are linearly independent, their span defines a higher-dimensional subspace.  
- If vectors are dependent, the span is redundant.  

---

## Basis

A **basis** of a subspace is a set of **linearly independent vectors** that span the subspace.

- Minimal representation of a subspace.  
- Any vector in the subspace can be written uniquely as a linear combination of basis vectors.  

**Dimension** of a subspace = number of vectors in its basis.

**Mini Example:**  
In $\mathbb{R}^2$, the standard basis is:

$$
e_1 =
\begin{bmatrix}
1 \\
0
\end{bmatrix}, 
\quad
e_2 =
\begin{bmatrix}
0 \\
1
\end{bmatrix}
$$

They span the entire 2D plane.

---

## Hands-on with Python and Rust

::: code-group

```python [Python]
import numpy as np

# Vectors
v1 = np.array([1, 0])
v2 = np.array([0, 1])

# Stack as matrix
A = np.column_stack([v1, v2])

# Rank gives dimension of span
rank = np.linalg.matrix_rank(A)

print("Matrix A:\n", A)
print("Rank (dimension of span):", rank)
```

```rust [Rust]
use ndarray::{array, Array2};
use ndarray_linalg::Rank;

fn main() {
    // Vectors
    let v1 = array![1.0, 0.0];
    let v2 = array![0.0, 1.0];

    // Stack into matrix
    let a: Array2<f64> = array![
        [v1[0], v2[0]],
        [v1[1], v2[1]]
    ];

    // Rank = dimension of span
    let rank = a.rank().unwrap();

    println!("Matrix A:\n{:?}", a);
    println!("Rank (dimension of span): {}", rank);
}
```

:::

---

## Connection to ML

- **Subspaces** → Data often lies in lower-dimensional subspaces (manifolds).  
- **Basis** → PCA finds a new orthogonal basis aligned with variance.  
- **Span & Rank** → Help identify redundancy among features.  

---

## Next Steps

Continue to [Linear Independence and Orthogonality](/ml-essentials/maths-for-aiml/linear-algebra/linear-independence-orthogonality).
