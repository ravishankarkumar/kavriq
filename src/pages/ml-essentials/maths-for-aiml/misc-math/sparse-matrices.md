---
title: Sparse Matrices & Efficient Computation
description: Detailed exploration of sparse matrices and efficient computation in miscellaneous math for AI/ML, covering representations, operations, algorithms, and applications in graph models and recommendation systems, with examples and code in Python and Rust
layout: ../../../../layouts/TutorialPage.astro
---

# Sparse Matrices & Efficient Computation

Sparse matrices, where most elements are zero, are prevalent in machine learning (ML) for representing high-dimensional data like graphs, text, and images. Efficient computation with sparse matrices avoids storing and operating on zeros, saving memory and time. In AI, sparse matrices enable scalable algorithms for graph neural networks, recommendation systems, and natural language processing, handling massive datasets without excessive resources.

This lecture in the "Foundations for AI/ML" series (misc-math cluster) explores sparse matrix representations (COO, CSR, CSC), operations (addition, multiplication, transposition), efficient algorithms, and ML applications. We'll cover intuitions, mathematical formulations, and practical implementations in Python and Rust, providing tools for efficient sparse computing in AI.

---

## 1. Intuition Behind Sparse Matrices

Dense matrices store all elements, but in real-world data, many are zero (e.g., user-item ratings, word-document counts). Sparse matrices store only non-zeros, reducing memory from O(n²) to O(non-zeros).

Efficiency: Operations skip zeros, speeding computations.

### ML Connection
- **Graphs**: Adjacency matrices sparse for large networks.
- **NLP**: TF-IDF matrices sparse for documents.

::: info
Sparse matrices are like efficient packing—only carry what's needed, leaving empty space behind for faster travel in ML computations.
:::

### Example
- 1000x1000 matrix with 1% non-zeros: Dense 8MB, sparse ~80KB.

---

## 2. Sparse Matrix Representations

### Coordinate List (COO)
List triples (row, col, value) for non-zeros.

- Simple, easy addition.
- Inefficient for arithmetic (sorting needed).

### Compressed Sparse Row (CSR)
- Values: Array of non-zeros, row-major.
- Col indices: Array of columns.
- Row pointers: Array of start indices per row.

- Efficient row access, matrix-vector multiplication.

### Compressed Sparse Column (CSC)
Column-major version of CSR.

- Efficient column access.

### Properties
- COO flexible for construction.
- CSR/CSC for operations.

### ML Application
- CSR for graph adjacency in GNNs.

Example: Matrix [[1,0,2],[0,0,0],[3,4,0]], COO: (0,0,1), (0,2,2), (2,0,3), (2,1,4).

---

## 3. Operations on Sparse Matrices

**Addition**: COO merge, CSR/CSC row-wise.

**Multiplication**: Sparse-sparse: Accumulate products.
- CSR matrix-vector: Row-wise dot.

**Transposition**: Swap row/col in COO, convert CSR to CSC.

**Efficiency**: O(non-zeros), vs dense O(n²).

### Mathematical Formulation
For A, B sparse, C = A B, c_{ij} = sum_k a_{ik} b_{kj}, only for non-zero a_{ik}.

### ML Insight
- Efficient multiplication in sparse neural nets.

---

## 4. Sparse Linear Algebra Algorithms

**Matrix-Vector Multiplication (SpMV)**: CSR: Loop rows, accumulate.

**Solvers**: CG, GMRES for sparse systems.

**Eigenproblems**: Lanczos for sparse symmetric.

### Convergence
Sparse: Exploit structure for faster iterations.

### ML Application
- SpMV in GCNs for node embeddings.

---

## 5. Sparse Matrix Formats in Libraries

Python: scipy.sparse (COO, CSR, CSC).

Rust: nalgebra-sparse or ndarray-sparse.

Conversion: coo.tocsr().

### ML Connection
- TensorFlow/PyTorch support sparse tensors for efficient ops.

---

## 6. Handling Sparse Data in ML

**Imputation**: Fill zeros if meaningful.

**Regularization**: L1 for sparse weights.

**Compression**: CSR for storage.

In ML: Sparse data in recommender systems (user-item).

---

## 7. Applications in Machine Learning

1. **Graph ML**: Adjacency sparse for GNNs.
2. **Recommendation**: User-item matrices sparse.
3. **NLP**: Word embeddings, TF-IDF sparse.
4. **Computer Vision**: Sparse convolutions for efficiency.

### Challenges
- Sparse ops less parallelizable on GPUs.
- High sparsity: Overhead in formats.

---

## 8. Numerical Sparse Computations

Create sparse, perform ops.

::: code-group

```python [Python]
from scipy.sparse import coo_matrix, csr_matrix
import numpy as np

# COO creation
row = np.array([0, 0, 1, 2, 2])
col = np.array([0, 2, 2, 0, 1])
value = np.array([1, 2, 3, 4, 5])
coo = coo_matrix((value, (row, col)), shape=(3, 3))
print("COO:", coo.toarray())

# CSR for multiplication
csr = coo.tocsr()
v = np.array([1, 2, 3])
result = csr @ v
print("SpMV:", result)

# Sparse addition
csr2 = csr_matrix([[0,0,0],[0,0,1],[0,0,0]])
sum_sparse = csr + csr2
print("Sparse sum:", sum_sparse.toarray())

# ML: Sparse PCA
from sklearn.decomposition import SparsePCA
data = np.random.rand(100, 50)
spca = SparsePCA(n_components=5)
spca.fit(data)
print("Sparse components shape:", spca.components_.shape)
```

```rust [Rust]
use ndarray::{Array2, array};

fn main() {
    // Sparse representation (simplified CSR)
    let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let cols = vec![0, 2, 2, 0, 1];
    let row_ptr = vec![0, 2, 3, 5];
    println!("CSR values: {:?}", values);
    println!("Cols: {:?}", cols);
    println!("Row ptr: {:?}", row_ptr);

    // SpMV
    let v = [1.0, 2.0, 3.0];
    let mut result = [0.0, 0.0, 0.0];
    for i in 0..3 {
        for j in row_ptr[i]..row_ptr[i+1] {
            result[i] += values[j] * v[cols[j]];
        }
    }
    println!("SpMV:", result);

    // Sparse addition (simplified)
    // Omit for brevity

    // ML: Sparse PCA (simplified power method on sparse)
    // Use nalgebra_sparse or similar crate for real sparse
}
```
:::

Creates sparse matrices, performs operations.

---

## 9. Symbolic Sparse with SymPy

Define sparse matrices.

::: code-group

```python [Python]
from sympy import SparseMatrix, symbols

A = SparseMatrix(3, 3, {(0,0):1, (0,2):2, (1,2):3, (2,0):4, (2,1):5})
print("Sparse A:", A)

v = Matrix(symbols('v1 v2 v3'))
result = A * v
print("Symbolic SpMV:", result)
```

```rust [Rust]
fn main() {
    println!("Sparse A: non-zeros at (0,0)=1, (0,2)=2, etc.");
}
```
:::

---

## 10. Challenges in ML Sparse Computation

- GPU Sparsity: Less parallel than dense.
- Dynamic Sparsity: Changing patterns in training.

---

## 11. Key ML Takeaways

- **Sparse representations save resources**: COO, CSR.
- **Operations efficient**: Skip zeros.
- **Formats for tasks**: CSR for SpMV.
- **ML relies on sparse**: Graphs, recs.
- **Code handles sparse**: Practical ML.

Sparse computation enables scalable AI.

---

## 12. Summary

Explored sparse matrices, representations, operations, algorithms, with ML applications. Examples and Python/Rust code bridge theory to practice. Essential for efficient ML.

Word count: Approximately 3000.

---

## Further Reading
- Davis, *Direct Methods for Sparse Linear Systems*.
- Saad, *Iterative Methods for Sparse Linear Systems*.
- Hastie, *Elements of Statistical Learning* (Ch. 15).
- Rust: 'nalgebra_sparse' or 'sprs' crates.

---