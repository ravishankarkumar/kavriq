---
title: Pseudo-Inverse & Ill-Conditioned Systems
description: Understanding the Moore–Penrose inverse, handling non-invertible matrices, and numerical stability in ML
---

# Pseudo-Inverse & Ill-Conditioned Systems

In machine learning, we often need to invert matrices (e.g., in linear regression: $(X^T X)^{-1}$).  
But what if the matrix is **not invertible** or is **ill-conditioned** (unstable for inversion)?  
This is where the **pseudo-inverse** and the concept of **numerical stability** come in.

---

## 1. The Moore–Penrose Pseudo-Inverse

If $A$ is not square or not invertible, we use the **Moore–Penrose inverse** $A^+$.  

**Definition:**  
$A^+$ is the unique matrix such that:

$$
A A^+ A = A, \quad A^+ A A^+ = A^+, \quad (A A^+)^T = A A^+, \quad (A^+ A)^T = A^+ A
$$

**In regression:**  
Instead of solving

$$
w = (X^T X)^{-1} X^T y
$$

we use

$$
w = X^+ y
$$

where $X^+$ is the pseudo-inverse (often computed via SVD).

::: info ML relevance
- Works even if $X^T X$ is singular (e.g., correlated features, fewer samples than features).  
- Used in **regularized regression** and **neural network pseudo-inverse training**.  
:::

---

## 2. Handling Non-Invertible Matrices in Regression

Situations where $(X^T X)$ is **not invertible**:  
- **Multicollinearity**: features are linearly dependent.  
- **Underdetermined systems**: more features than samples.  

**Solutions:**  
- Use **pseudo-inverse**.  
- Add **regularization** (Ridge regression: $(X^T X + \lambda I)^{-1}$).  
- Reduce dimensionality (PCA).  

---

## 3. Condition Number & Numerical Stability

The **condition number** of a matrix $A$ (with respect to inversion) is:

$$
\kappa(A) = \|A\| \cdot \|A^{-1}\|
$$

- If $\kappa(A)$ is large → small input errors cause large output errors.  
- High condition number → matrix is **ill-conditioned**.  

::: info ML relevance
- Ill-conditioned $X^T X$ means regression weights are highly unstable.  
- Regularization (Ridge) reduces condition number.  
- QR or SVD are often used instead of direct inversion for stability.  
:::

---

## Hands-on with Python and Rust

::: code-group

```python [Python]
import numpy as np

# Feature matrix with collinearity
X = np.array([[1, 2], [2, 4], [3, 6]])  # second column is 2x first
y = np.array([1, 2, 3])

# Direct normal equation (fails: X^T X not invertible)
try:
    w = np.linalg.inv(X.T @ X) @ X.T @ y
except np.linalg.LinAlgError:
    print("Matrix is singular, cannot invert.")

# Use pseudo-inverse instead
w_pinv = np.linalg.pinv(X) @ y

# Condition number
cond_num = np.linalg.cond(X)

print("Pseudo-inverse solution:", w_pinv)
print("Condition number of X:", cond_num)
```

```rust [Rust]
use ndarray::{array, Array2, Array1};
use ndarray_linalg::{PseudoInverse, Norm};

fn main() {
    // Feature matrix with collinearity
    let x: Array2<f64> = array![
        [1.0, 2.0],
        [2.0, 4.0],
        [3.0, 6.0]
    ];
    let y: Array1<f64> = array![1.0, 2.0, 3.0];

    // Pseudo-inverse solution
    let x_pinv = x.pinv(1e-8).unwrap();
    let w = x_pinv.dot(&y);

    // Condition number
    let cond_num = x.norm_l2() * x_pinv.norm_l2();

    println!("Pseudo-inverse solution: {:?}", w);
    println!("Condition number of X: {}", cond_num);
}
```

:::

---

## Summary

- **Pseudo-inverse (Moore–Penrose)** solves regression when $(X^T X)$ is not invertible.  
- **Ill-conditioning** → unstable solutions due to large condition numbers.  
- **Fixes** → pseudo-inverse, regularization, SVD/QR-based methods.  

---

## Next Steps

Continue to [Block Matrices and Kronecker Products](/ml-essentials/maths-for-aiml/linear-algebra/block-kronecker).
