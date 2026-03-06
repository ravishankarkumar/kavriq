---
title: Projections and Least Squares
description: Understanding projections and least squares in ML
---

# Projections and Least Squares

In this lesson, we connect **orthogonality** with one of the most important tools in machine learning: **least squares regression**.  
The key idea: when fitting models, we are often **projecting data onto a subspace** that best approximates it.

---

## Projection of a Vector onto Another

Given a vector $y$ and a direction $x$, the **projection of $y$ onto $x$** is:

$$
\text{proj}_x(y) = \frac{y \cdot x}{x \cdot x} x
$$

- This is the component of $y$ along $x$.  
- The residual $r = y - \text{proj}_x(y)$ is orthogonal to $x$.  

::: info Why this matters
In ML, when we approximate a target vector $y$ using features $x$, the **error (residual)** must be orthogonal to $x$ for the solution to be optimal.
:::

---

## Projection onto a Subspace

If we have multiple feature vectors (columns of $X$), the projection of $y$ onto the subspace spanned by $X$ is:

$$
\hat{y} = X (X^T X)^{-1} X^T y
$$

- $X$ is the design matrix (features).  
- $\hat{y}$ is the projection of $y$ onto the column space of $X$.  
- The residual $y - \hat{y}$ is orthogonal to all columns of $X$.  

This formula is exactly the **ordinary least squares (OLS)** solution.

---

## Least Squares Problem

We want to solve:

$$
\min_w \| y - Xw \|^2
$$

Expanding and differentiating w.r.t. $w$ gives the **normal equations**:

$$
X^T X w = X^T y
$$

Solving yields:

$$
w = (X^T X)^{-1} X^T y
$$

This is the same closed-form solution we saw in linear regression.

::: info Geometric interpretation
OLS finds the vector $\hat{y} = Xw$ in the column space of $X$ that is **closest** to $y$ (in Euclidean distance).  
The difference $y - \hat{y}$ is the residual, orthogonal to the feature space.
:::

---

## Hands-on with Python and Rust

::: code-group

```python [Python]
import numpy as np

# Feature matrix X and target y
X = np.array([[1], [2], [3]])
y = np.array([2, 2.9, 4.1])

# Closed-form solution (normal equation)
w = np.linalg.inv(X.T @ X) @ X.T @ y

# Predictions (projection of y onto span(X))
y_hat = X @ w

# Residual (orthogonal to X)
residual = y - y_hat

print("Weight:", w)
print("Predictions:", y_hat)
print("Residual (should be orthogonal):", residual)
print("Dot(X[:,0], residual) =", np.dot(X[:,0], residual))
```

```rust [Rust]
use ndarray::{array, Array1, Array2};
use ndarray_linalg::Inverse;

fn main() {
    // Feature matrix X and target y
    let x: Array2<f64> = array![[1.0], [2.0], [3.0]];
    let y: Array1<f64> = array![2.0, 2.9, 4.1];

    // Compute (X^T X)^{-1} X^T y
    let xtx = x.t().dot(&x);
    let xtx_inv = xtx.inv().unwrap();
    let xty = x.t().dot(&y);
    let w = xtx_inv.dot(&xty);

    // Predictions
    let y_hat = x.dot(&w);

    // Residual
    let residual = &y - &y_hat;
    let dot = x.column(0).dot(&residual);

    println!("Weight: {:?}", w);
    println!("Predictions: {:?}", y_hat);
    println!("Residual: {:?}", residual);
    println!("Dot(X[:,0], residual) = {}", dot);
}
```

:::

---

## Connection to Machine Learning

- **Linear regression** is solving a least squares problem.  
- **PCA** can also be seen as a projection (onto directions of maximum variance).  
- Many ML methods boil down to finding the “best projection” of data onto a simpler subspace.  

---

<!-- ## Next Steps -->

<!-- Continue to [Functions and Limits](/maths-for-aiml/calculus/functions-limits). -->

