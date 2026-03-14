---
title: Matrix Operations - Multiplication, Transpose, and Inverse
description: Matrix operations for ML – multiplication, transpose, and inverse
layout: ../../../../layouts/TutorialPage.astro
---

# Matrix Operations: Multiplication, Transpose, and Inverse

Matrices are at the heart of machine learning. Operations on matrices power linear regression, neural networks, dimensionality reduction, and much more. In this lesson, we'll cover three fundamental operations: **multiplication**, **transpose**, and **inverse**.


## Matrix Multiplication

Given two matrices $A \in \mathbb{R}^{m \times n}$ and $B \in \mathbb{R}^{n \times p}$, their product $C = AB$ is defined as:

$$
C_{ij} = \sum_{k=1}^n A_{ik} B_{kj}
$$

- The number of **columns of A** must equal the number of **rows of B**.  
- The result has shape $m \times p$.

::: info Explanation of Multiplication
Matrix multiplication represents combining **linear transformations**.  
- In ML: multiplying dataset matrix $X$ with parameter vector $w$ computes predictions.  
:::

**Mini Example:**  
$$
A = \begin{bmatrix}1 & 2 \\ 3 & 4\end{bmatrix}, 
B = \begin{bmatrix}5 \\ 6\end{bmatrix}
$$

Then:  
$$
AB = \begin{bmatrix}1\cdot 5 + 2\cdot 6 \\ 3\cdot 5 + 4\cdot 6\end{bmatrix} =
\begin{bmatrix}17 \\ 39\end{bmatrix}
$$


## Matrix Transpose

The **transpose** of a matrix $A$ flips rows and columns:

$$
(A^T)_{ij} = A_{ji}
$$

Example:  
$$
A = \begin{bmatrix}1 & 2 & 3 \\ 4 & 5 & 6\end{bmatrix} 
\quad \Rightarrow \quad
A^T = \begin{bmatrix}1 & 4 \\ 2 & 5 \\ 3 & 6\end{bmatrix}
$$

::: info Explanation of Transpose
Transpose is essential in ML when switching between “row as samples” vs “column as features”.  
- In regression: $X^T X$ appears in the normal equation.  
:::


## Matrix Inverse

The **inverse** of a square matrix $A$ (if it exists) is $A^{-1}$ such that:

$$
AA^{-1} = A^{-1}A = I
$$

where $I$ is the identity matrix.

- Not all matrices are invertible (singular matrices don't have inverses).  
- In ML: matrix inverses appear in the **normal equation** for linear regression:  
  $$w = (X^T X)^{-1} X^T y$$

**Mini Example:**  
$$
A = \begin{bmatrix}4 & 7 \\ 2 & 6\end{bmatrix}
\quad \Rightarrow \quad
A^{-1} = \frac{1}{(4)(6) - (7)(2)} \begin{bmatrix}6 & -7 \\ -2 & 4\end{bmatrix}
= \frac{1}{10} \begin{bmatrix}6 & -7 \\ -2 & 4\end{bmatrix}
$$

::: info Explanation of Inverse
Matrix inverse is like division for matrices.  
- But inversion is computationally expensive.  
- In ML practice, we often avoid explicit inversion, using more stable methods (like QR or gradient descent).  
:::


## Hands-on with Python and Rust

::: code-group

```python [Python]
import numpy as np

# Define matrices
A = np.array([[1, 2], [3, 4]])
B = np.array([[5], [6]])

# Multiplication
C = A.dot(B)

# Transpose
AT = A.T

# Inverse
A_inv = np.linalg.inv(np.array([[4, 7], [2, 6]]))

print("A * B =\n", C)
print("Transpose of A =\n", AT)
print("Inverse of [[4,7],[2,6]] =\n", A_inv)
```

```rust [Rust]
use ndarray::{array, Array2};
use ndarray_linalg::Inverse;

fn main() {
    // Define matrices
    let a: Array2<f64> = array![[1.0, 2.0], [3.0, 4.0]];
    let b: Array2<f64> = array![[5.0], [6.0]];

    // Multiplication
    let c = a.dot(&b);

    // Transpose
    let at = a.t().to_owned();

    // Inverse
    let a2: Array2<f64> = array![[4.0, 7.0], [2.0, 6.0]];
    let a_inv = a2.inv().unwrap();

    println!("A * B =\n{:?}", c);
    println!("Transpose of A =\n{:?}", at);
    println!("Inverse of [[4,7],[2,6]] =\n{:?}", a_inv);
}
```

:::


## Connection to ML

- **Multiplication** → computing predictions, combining transformations.  
- **Transpose** → appears in regression, covariance, gradient formulas.  
- **Inverse** → theoretical solution in linear regression (though avoided in practice).  


## Next Steps

Continue to [Special Matrices: Identity, Diagonal, Orthogonal](/ml-essentials/maths-for-aiml/linear-algebra/special-matrices).
