---
title: Rank, Determinant, and Inverses
description: Rank, determinant, and inverses of matrices in ML
layout: ../../../../layouts/TutorialPage.astro
---

# Rank, Determinant, and Inverses

Understanding the **rank**, **determinant**, and **inverse** of a matrix helps us reason about when systems of equations are solvable, when data is redundant, and when certain machine learning methods can be applied.


## Rank

The **rank** of a matrix is the maximum number of linearly independent rows or columns.

- Rank tells us how much **useful information** a matrix has.  
- If rank = number of columns, the columns are linearly independent.  
- If rank < number of columns, some features are redundant (linear combinations of others).

**Mini Example:**  
$$
A =
\begin{bmatrix}
1 & 2 & 3 \\
4 & 5 & 6 \\
7 & 8 & 9
\end{bmatrix}
$$

Here, rank$(A) = 2$ because one row/column is a linear combination of the others.

::: info Explanation of Rank
Rank tells us whether the matrix is “full information” or if it contains redundancy.  
- In ML: rank-deficient design matrices $X$ cause issues in regression because $X^T X$ becomes non-invertible.  
:::


## Determinant

The **determinant** of a square matrix is a scalar value that summarizes important properties:

- If det$(A) = 0$, the matrix is **singular** (non-invertible).  
- If det$(A) \neq 0$, the matrix is invertible.  
- Geometric meaning: determinant is the **scaling factor** of volume under the transformation.

**Mini Example:**  
$$
A =
\begin{bmatrix}
4 & 7 \\
2 & 6
\end{bmatrix}
$$

det$(A) = 4\cdot 6 - 7\cdot 2 = 10$

::: info Explanation of Determinant
Determinant tells us if a matrix “collapses” space.  
- det = 0 → volume collapses to lower dimension (no inverse).  
- det ≠ 0 → transformation preserves space (matrix invertible).  
:::


## Inverse (Recap)

The **inverse** $A^{-1}$ satisfies:

$$
AA^{-1} = A^{-1}A = I
$$

- Only square, non-singular matrices (det ≠ 0) have inverses.  
- In ML: used in the normal equation for regression, but avoided in practice.

**Mini Example:**  
For the same $A$ above:  
$$
A^{-1} = \frac{1}{10} 
\begin{bmatrix}
6 & -7 \\
-2 & 4
\end{bmatrix}
$$


## Hands-on with Python and Rust

::: code-group

```python [Python]
import numpy as np

A = np.array([[4, 7], [2, 6]])
B = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# Rank
rank_B = np.linalg.matrix_rank(B)

# Determinant
det_A = np.linalg.det(A)

# Inverse
inv_A = np.linalg.inv(A)

print("Rank of B:", rank_B)
print("Determinant of A:", det_A)
print("Inverse of A:\n", inv_A)
```

```rust [Rust]
use ndarray::array;
use ndarray::Array2;
use ndarray_linalg::{Determinant, Inverse, Rank};

fn main() {
    let a: Array2<f64> = array![[4.0, 7.0], [2.0, 6.0]];
    let b: Array2<f64> = array![
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0]
    ];

    // Rank
    let rank_b = b.rank().unwrap();

    // Determinant
    let det_a = a.det().unwrap();

    // Inverse
    let inv_a = a.inv().unwrap();

    println!("Rank of B: {}", rank_b);
    println!("Determinant of A: {}", det_a);
    println!("Inverse of A:\n{:?}", inv_a);
}
```

:::


## Connection to ML

- **Rank** → tells us if features are redundant (multicollinearity).  
- **Determinant** → indicates whether transformations preserve volume (non-singular).  
- **Inverse** → appears in closed-form regression solutions (but we prefer iterative solvers).  


## Next Steps

Continue to [Eigenvalues and Eigenvectors](/ml-essentials/maths-for-aiml/linear-algebra/eigenvalues-eigenvectors).
