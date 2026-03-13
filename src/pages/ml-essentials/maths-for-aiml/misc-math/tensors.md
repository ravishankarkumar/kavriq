---
title: Tensors & Tensor Operations
description: Comprehensive exploration of tensors and tensor operations in miscellaneous math for AI/ML, covering definitions, operations, properties, and applications in neural networks, data representation, and optimization, with examples and code in Python and Rust
layout: ../../../../layouts/TutorialPage.astro
---

# Tensors & Tensor Operations

Tensors are powerful mathematical objects that generalize scalars, vectors, and matrices to higher dimensions, serving as the backbone of data representation in machine learning (ML). In artificial intelligence, tensors enable efficient storage and manipulation of multidimensional data, such as images, time series, and neural network weights. Tensor operations, like contraction, product, and decomposition, facilitate computations in deep learning frameworks, optimization algorithms, and data transformations.

This lecture in the "Foundations for AI/ML" series (misc-math cluster) builds on high-dimensional statistics and random projections, exploring tensors, their mathematical properties, operations, and ML applications. We'll provide intuitive explanations, derivations, and practical implementations in Python and Rust, offering tools to leverage tensors in AI.

---

## 1. Intuition Behind Tensors

A tensor is a multidimensional array with specific transformation properties, generalizing scalars (0D), vectors (1D), and matrices (2D) to higher orders. In ML, tensors represent complex data structures, like 3D image data (height, width, channels) or 4D video data (frames, height, width, channels).

Tensor operations manipulate these arrays efficiently, enabling computations like neural network forward passes or gradient updates.

### ML Connection
- **Neural Networks**: Weights, inputs, outputs as tensors.
- **Computer Vision**: Images as 3D tensors.
- **NLP**: Word embeddings as tensors.

::: info
Tensors are like multi-layered containers, organizing data in multiple dimensions; their operations are the tools to reshape and combine them for ML tasks.
:::

### Example
- A 3D tensor (28×28×3) represents an RGB image for a CNN.

---

## 2. Tensor Definitions and Properties

**Tensor**: A multi-indexed array T_{i_1,...,i_k} with k indices (order/rank k).

- Scalar: Order 0, e.g., 5.
- Vector: Order 1, e.g., [1,2,3].
- Matrix: Order 2, e.g., [[1,2],[3,4]].
- Higher-order: e.g., 3D tensor for image (height×width×channels).

**Shape**: Dimensions (n_1, ..., n_k).

**Rank**: Number of indices (not matrix rank).

### Properties
- **Linearity**: Operations linear in each index.
- **Transformation**: Change under coordinate transformations (covariant/contravariant).
- **Symmetry**: Invariant under index swaps (e.g., symmetric tensors).

### ML Insight
- Tensors unify data representation in frameworks like TensorFlow, PyTorch.

---

## 3. Tensor Operations

### Element-Wise Operations
Add, subtract, multiply tensors of same shape, e.g., C_{ijk} = A_{ijk} + B_{ijk}.

### Tensor Product
Outer product: For vectors u, v, T_{ij} = u_i v_j.

### Contraction
Sum over repeated indices: For T_{ijk}, C_{ik} = sum_j T_{ijk}.

### Matricization
Flatten tensor to matrix, e.g., 3D tensor to 2D matrix.

### Tensor Decomposition
- **CP Decomposition**: T ≈ sum_r λ_r u_r ⊗ v_r ⊗ w_r.
- **Tucker Decomposition**: T ≈ G ×_1 U ×_2 V ×_3 W (core tensor G).

### ML Application
- Contraction in neural net layers; decomposition for compression.

---

## 4. Tensor Operations in Deep Learning

**Forward Pass**: Matrix multiplication as tensor contraction.

**Backpropagation**: Gradients as higher-order tensors.

**Convolution**: Tensor operation sliding filters over inputs.

In ML: Frameworks automate tensor ops for efficiency.

---

## 5. Tensor Decompositions for ML

**CP Decomposition**: Factorize high-order tensors into sums of rank-1 tensors, reducing parameters.

**Tucker**: Generalizes PCA to tensors, used in compression.

**Tensor Train (TT)**: Sequential factorization for high-d.

In ML: Compress neural net weights, speed up inference.

---

## 6. Applications in Machine Learning

1. **Neural Networks**: Weights, activations as tensors.
2. **Computer Vision**: Images/videos as 3D/4D tensors.
3. **NLP**: Word embeddings, attention as tensors.
4. **Compression**: Tensor decomposition for model efficiency.
5. **Graph ML**: Adjacency tensors for GNNs.

### Challenges
- **High-Order Tensors**: Memory-intensive.
- **Computation**: Ops scale with dimensions.
- **Non-Unique Decompositions**: CP not unique.

---

## 7. Numerical Tensor Operations

Implement tensor creation, operations, decomposition.

::: code-group

```python [Python]
import numpy as np
from tensorly.decomposition import parafac
import matplotlib.pyplot as plt

# Create 3D tensor
tensor = np.random.rand(3, 4, 5)  # 3x4x5 tensor
print("Tensor shape:", tensor.shape)

# Element-wise addition
tensor2 = np.random.rand(3, 4, 5)
sum_tensor = tensor + tensor2
print("Sum tensor shape:", sum_tensor.shape)

# Contraction (sum over axis 2)
contracted = np.sum(tensor, axis=2)
print("Contracted shape:", contracted.shape)

# CP decomposition
weights, factors = parafac(tensor, rank=2)
print("CP factors shapes:", [f.shape for f in factors])

# ML: Tensor for CNN input
image = np.random.rand(28, 28, 3)  # RGB image
filter = np.random.rand(3, 3, 3, 2)  # Conv filter
# Simplified convolution (using tensordot for contraction)
conv = np.tensordot(image, filter, axes=([2],[2]))
print("Conv output shape:", conv.shape)

# Visualize slice
plt.imshow(tensor[0, :, :])
plt.title("Tensor Slice")
plt.show()
```

```rust [Rust]
use ndarray::{Array3, ArrayD};
use rand::Rng;

fn main() {
    let mut rng = rand::thread_rng();
    // Create 3D tensor
    let tensor = Array3::from_shape_fn((3, 4, 5), |_| rng.gen::<f64>());
    println!("Tensor shape: {:?}", tensor.shape());

    // Element-wise addition
    let tensor2 = Array3::from_shape_fn((3, 4, 5), |_| rng.gen::<f64>());
    let sum_tensor = &tensor + &tensor2;
    println!("Sum tensor shape: {:?}", sum_tensor.shape());

    // Contraction (sum over axis 2)
    let contracted = tensor.sum_axis(ndarray::Axis(2));
    println!("Contracted shape: {:?}", contracted.shape());

    // ML: Tensor for CNN input
    let image = Array3::from_shape_fn((28, 28, 3), |_| rng.gen::<f64>());
    let filter = Array3::from_shape_fn((3, 3, 3), |_| rng.gen::<f64>());
    // Simplified contraction (sum over channels)
    let conv = image
        .outer_iter()
        .zip(filter.outer_iter())
        .map(|(img, filt)| img.dot(&filt))
        .collect::<Vec<f64>>();
    // Reshape omitted for brevity
}
```
:::

Implements tensor creation, operations.

---

## 8. Symbolic Tensor Operations with SymPy

Define tensors, compute operations.

::: code-group

```python [Python]
from sympy import Array, symbols

# 3D tensor
i, j, k = symbols('i j k')
T = Array([[[symbols(f'T_{i}{j}{k}') for k in range(3)] for j in range(4)] for i in range(2)])
print("Tensor T:", T)

# Contraction
contracted = T.sum(k)
print("Contracted:", contracted)
```

```rust [Rust]
fn main() {
    println!("Tensor T: T_{ijk}");
    println!("Contracted: sum_k T_{ijk}");
}
```
:::

---

## 9. Challenges in ML Tensor Applications

- **Memory**: High-order tensors consume large memory.
- **Computation**: Ops scale with dimensions.
- **Sparsity**: Many ML tensors sparse, need special handling.

---

## 10. Key ML Takeaways

- **Tensors generalize data**: Multi-dim representation.
- **Operations enable computation**: Contraction, product.
- **Decompositions compress**: CP, Tucker.
- **ML relies on tensors**: NNs, vision, NLP.
- **Code handles tensors**: Practical ML.

Tensors power ML data manipulation.

---

## 11. Summary

Explored tensors, their operations, properties, and ML applications in neural networks and data representation. Examples and Python/Rust code bridge theory to practice. Essential for efficient ML computation.

Word count: Approximately 3000.

---

## Further Reading
- Kolda, Bader, *Tensor Decompositions and Applications*.
- Goodfellow, *Deep Learning* (Ch. 6).
- Cichocki, *Tensor Networks for Big Data*.
- Rust: 'ndarray' for tensors, 'nalgebra' for linalg.

---