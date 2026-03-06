---
title: Vector Operations - Dot Product, Norms, and Distances
description: Vector operations for ML – dot product, norms, and distances
---

# Vector Operations: Dot Product, Norms, and Distances

Vectors are the building blocks of machine learning. Beyond representing features, we often need to measure **similarity**, **magnitude**, and **distance** between vectors. These concepts form the basis of algorithms like k-Nearest Neighbors, clustering, and gradient computations.


## Dot Product (Inner Product)

The **dot product** of two vectors $\mathbf{a}, \mathbf{b} \in \mathbb{R}^n$ is:

$$
\mathbf{a} \cdot \mathbf{b} = \sum_{i=1}^n a_i b_i
$$

::: info Explanation of Dot Product
- Geometrically: measures how much two vectors point in the same direction.  
- If $\mathbf{a} \cdot \mathbf{b} > 0$: they point roughly the same way.  
- If $\mathbf{a} \cdot \mathbf{b} < 0$: they point in opposite directions.  
- If $\mathbf{a} \cdot \mathbf{b} = 0$: they are orthogonal (perpendicular).  
:::

**Mini Example:**  
$$
[1, 2, 3] \cdot [4, 5, 6] = 1\cdot 4 + 2\cdot 5 + 3\cdot 6 = 32
$$


## Vector Norms (Magnitude / Length)

A **norm** measures the size or length of a vector. The most common is the **Euclidean norm ($L_2$)**:

$$
\|\mathbf{a}\|_2 = \sqrt{a_1^2 + a_2^2 + \dots + a_n^2}
$$

Other norms:
- **$L_1$ norm (Manhattan distance)**: $\|\mathbf{a}\|_1 = \sum |a_i|$  
- **$L_\infty$ norm (Max norm)**: $\|\mathbf{a}\|_\infty = \max |a_i|$

::: info Explanation of Norms
- $\|\mathbf{a}\|_2$ = straight-line length from origin.  
- $\|\mathbf{a}\|_1$ = “grid path” length (like walking city blocks).  
- $\|\mathbf{a}\|_\infty$ = maximum single coordinate size.  
:::

**Mini Example:**  
For $\mathbf{a} = [3, 4]$:  
- $\|\mathbf{a}\|_2 = \sqrt{3^2 + 4^2} = 5$  
- $\|\mathbf{a}\|_1 = 3 + 4 = 7$  
- $\|\mathbf{a}\|_\infty = \max(3, 4) = 4$  


## Distances Between Vectors

The **distance** between two vectors measures how far apart they are.

- **Euclidean Distance ($L_2$)**:  
  $$
  d(\mathbf{a}, \mathbf{b}) = \|\mathbf{a} - \mathbf{b}\|_2
  $$  

- **Manhattan Distance ($L_1$)**:  
  $$
  d(\mathbf{a}, \mathbf{b}) = \|\mathbf{a} - \mathbf{b}\|_1
  $$  

- **Cosine Similarity** (based on dot product):  
  $$
  \cos \theta = \frac{\mathbf{a} \cdot \mathbf{b}}{\|\mathbf{a}\| \|\mathbf{b}\|}
  $$

::: info Explanation of Distances
- Euclidean = straight-line distance.  
- Manhattan = sum of coordinate-wise differences.  
- Cosine similarity = angle between vectors; used in NLP & recommendation systems.  
:::

**Mini Example:**  
For $\mathbf{a} = [1,2]$, $\mathbf{b} = [4,6]$:  
- Euclidean: $\sqrt{(4-1)^2 + (6-2)^2} = \sqrt{25} = 5$  
- Manhattan: $|4-1| + |6-2| = 3 + 4 = 7$  
- Cosine similarity: $\frac{1\cdot 4 + 2\cdot 6}{\sqrt{1^2+2^2}\sqrt{4^2+6^2}} = \frac{16}{\sqrt{5}\sqrt{52}} \approx 0.99$


## Hands-on with Python and Rust

::: code-group

```python [Python]
import numpy as np

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

# Dot product
dot = np.dot(a, b)

# Norms
norm2 = np.linalg.norm(a, 2)
norm1 = np.linalg.norm(a, 1)
norm_inf = np.linalg.norm(a, np.inf)

# Distances
euclidean = np.linalg.norm(a - b)
manhattan = np.sum(np.abs(a - b))
cosine = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

print("Dot product:", dot)
print("L2 norm of a:", norm2)
print("Euclidean distance:", euclidean)
print("Cosine similarity:", cosine)
```

```rust [Rust]
use ndarray::array;
use ndarray::Array1;

fn dot(a: &Array1<f64>, b: &Array1<f64>) -> f64 {
    a.dot(b)
}

fn norm_l2(a: &Array1<f64>) -> f64 {
    a.dot(a).sqrt()
}

fn norm_l1(a: &Array1<f64>) -> f64 {
    a.iter().map(|x| x.abs()).sum()
}

fn euclidean_distance(a: &Array1<f64>, b: &Array1<f64>) -> f64 {
    ((a - b).mapv(|x| x.powi(2))).sum().sqrt()
}

fn cosine_similarity(a: &Array1<f64>, b: &Array1<f64>) -> f64 {
    dot(a, b) / (norm_l2(a) * norm_l2(b))
}

fn main() {
    let a = array![1.0, 2.0, 3.0];
    let b = array![4.0, 5.0, 6.0];

    println!("Dot product: {}", dot(&a, &b));
    println!("L2 norm of a: {}", norm_l2(&a));
    println!("L1 norm of a: {}", norm_l1(&a));
    println!("Euclidean distance: {}", euclidean_distance(&a, &b));
    println!("Cosine similarity: {}", cosine_similarity(&a, &b));
}
```

:::


## Connection to ML

- **Dot product** → similarity, projections, neural network activations.  
- **Norms** → regularization, measuring feature magnitudes.  
- **Distances** → kNN, clustering, recommender systems, embeddings.  


## Next Steps

Continue to [Matrix Operations: Multiplication, Transpose, and Inverse](/ml-essentials/maths-for-aiml/linear-algebra/matrix-operations).
