---
title: Linear Transformations and Geometry
description: Understanding linear transformations and their geometric meaning for ML
layout: ../../../../layouts/TutorialPage.astro
---

# Linear Transformations and Geometry

Linear algebra is not just about numbers in tables—it's about **transformations of space**. Matrices can be seen as functions that transform vectors, stretching, rotating, reflecting, or projecting them. Understanding these transformations geometrically is essential for intuition in machine learning.

---

## Linear Transformations

A function $T: \mathbb{R}^n \to \mathbb{R}^m$ is a **linear transformation** if it satisfies:

1. **Additivity**: $T(x + y) = T(x) + T(y)$  
2. **Homogeneity**: $T(cx) = cT(x)$ for scalar $c$  

Every linear transformation can be represented as a **matrix multiplication**:

$$
T(x) = A x
$$

for some matrix $A$.

---

## Geometric Interpretations

1. **Scaling** – Multiply vectors by a scalar.  
   Example: $\begin{bmatrix}2 & 0 \\ 0 & 3\end{bmatrix}$ scales $x$ by 2 and $y$ by 3.  

2. **Rotation** – Preserve length but rotate direction.  
   Example (2D rotation by $\theta$):  
   $$
   R(\theta) =
   \begin{bmatrix}
   \cos \theta & -\sin \theta \\
   \sin \theta & \cos \theta
   \end{bmatrix}
   $$

3. **Reflection** – Flip across a line or plane.  

4. **Projection** – Collapse vectors onto a subspace.  
   Example: Project onto $x$-axis in 2D:  
   $$
   P =
   \begin{bmatrix}
   1 & 0 \\
   0 & 0
   \end{bmatrix}
   $$

---

## Mini Examples

- Stretching:  
  $$
  A =
  \begin{bmatrix}
  2 & 0 \\
  0 & 1
  \end{bmatrix}
  $$  
  doubles the $x$-axis component while leaving $y$ unchanged.  

- Rotation by 90°:  
  $$
  R =
  \begin{bmatrix}
  0 & -1 \\
  1 & 0
  \end{bmatrix}
  $$  
  rotates any vector counterclockwise by 90°.  

- Projection:  
  $$
  P =
  \begin{bmatrix}
  1 & 0 \\
  0 & 0
  \end{bmatrix}
  $$  
  projects vectors onto the $x$-axis.  

---

## Hands-on with Python and Rust

::: code-group

```python [Python]
import numpy as np

v = np.array([1, 2])

# Scaling
A = np.array([[2, 0], [0, 3]])
scaled = A.dot(v)

# Rotation (90 degrees)
R = np.array([[0, -1], [1, 0]])
rotated = R.dot(v)

# Projection onto x-axis
P = np.array([[1, 0], [0, 0]])
projected = P.dot(v)

print("Original vector:", v)
print("Scaled:", scaled)
print("Rotated:", rotated)
print("Projected:", projected)
```

```rust [Rust]
use ndarray::{array, Array1, Array2};

fn main() {
    let v: Array1<f64> = array![1.0, 2.0];

    // Scaling
    let a: Array2<f64> = array![[2.0, 0.0], [0.0, 3.0]];
    let scaled = a.dot(&v);

    // Rotation (90 degrees)
    let r: Array2<f64> = array![[0.0, -1.0], [1.0, 0.0]];
    let rotated = r.dot(&v);

    // Projection onto x-axis
    let p: Array2<f64> = array![[1.0, 0.0], [0.0, 0.0]];
    let projected = p.dot(&v);

    println!("Original vector: {:?}", v);
    println!("Scaled: {:?}", scaled);
    println!("Rotated: {:?}", rotated);
    println!("Projected: {:?}", projected);
}
```

:::

---

## Connection to ML

- **Scaling** → standardization of features.  
- **Rotation** → PCA rotates data into new basis.  
- **Projection** → dimensionality reduction (projecting onto lower-dimensional subspaces).  
- **Reflections & symmetries** → used in data augmentation and transformations.  

---

## Next Steps

Continue to [Subspaces and Basis](/ml-essentials/maths-for-aiml/linear-algebra/subspaces-basis).
