---
title: Linear Independence and Orthogonality
description: Understanding linear independence and orthogonality in ML
---

# Linear Independence and Orthogonality

Two of the most important concepts in linear algebra are **linear independence** and **orthogonality**. They form the backbone of many machine learning techniques, from feature selection to dimensionality reduction.

---

## Linear Independence

A set of vectors $\{v_1, v_2, ..., v_k\}$ is **linearly independent** if:

$$
a_1 v_1 + a_2 v_2 + ... + a_k v_k = 0 \quad \Rightarrow \quad a_1 = a_2 = ... = a_k = 0
$$

In other words, no vector in the set can be written as a linear combination of the others.

- If vectors are dependent, some information is redundant.  
- The number of independent vectors determines the **dimension** of their span (basis).  

**Mini Example:**  
- $[1, 0]$ and $[0, 1]$ are independent.  
- $[1, 0]$ and $[2, 0]$ are dependent (second is a multiple of the first).  

---

## Orthogonality

Two vectors $u$ and $v$ are **orthogonal** if their dot product is zero:

$$
u \cdot v = 0
$$

- Orthogonal vectors are **perpendicular** in geometry.  
- If orthogonal and unit length → **orthonormal**.  

**Mini Example:**  
- $[1, 0]$ and $[0, 1]$ are orthogonal.  
- $[1, 2]$ and $[2, -1]$ are also orthogonal (dot product = 0).  

---

## Why Are These Important?

- **Linear independence** → ensures features add unique information.  
- **Orthogonality** → simplifies computations (orthogonal bases make projections easy).  
- **Orthonormal basis** → foundation of PCA, Fourier transforms, and many ML algorithms.  

---

## Hands-on with Python and Rust

::: code-group

```python [Python]
import numpy as np

# Independent vectors
v1 = np.array([1, 0])
v2 = np.array([0, 1])

# Dependent vectors
v3 = np.array([2, 0])

# Check independence via rank
A = np.column_stack([v1, v2])
rank_A = np.linalg.matrix_rank(A)

B = np.column_stack([v1, v3])
rank_B = np.linalg.matrix_rank(B)

# Check orthogonality
dot_v1_v2 = np.dot(v1, v2)

print("Rank of [v1,v2]:", rank_A)
print("Rank of [v1,v3]:", rank_B)
print("Dot(v1,v2) =", dot_v1_v2)
```

```rust [Rust]
use ndarray::{array, Array2};
use ndarray_linalg::Rank;

fn main() {
    // Independent vectors
    let v1 = array![1.0, 0.0];
    let v2 = array![0.0, 1.0];

    // Dependent vector
    let v3 = array![2.0, 0.0];

    // Stack vectors as matrix
    let a: Array2<f64> = array![
        [v1[0], v2[0]],
        [v1[1], v2[1]]
    ];
    let b: Array2<f64> = array![
        [v1[0], v3[0]],
        [v1[1], v3[1]]
    ];

    // Rank check
    let rank_a = a.rank().unwrap();
    let rank_b = b.rank().unwrap();

    // Dot product
    let dot_v1_v2 = v1[0]*v2[0] + v1[1]*v2[1];

    println!("Rank of [v1,v2]: {}", rank_a);
    println!("Rank of [v1,v3]: {}", rank_b);
    println!("Dot(v1,v2) = {}", dot_v1_v2);
}
```

:::

---

## Connection to ML

- **Linear independence** → avoids redundant features, improves interpretability.  
- **Orthogonality** → PCA finds orthogonal axes of variance.  
- **Orthonormal bases** → simplify projections, reduce error accumulation.  

---

## Next Steps

Continue to [Projections and Least Squares](/ml-essentials/maths-for-aiml/linear-algebra/projections-least-squares).
