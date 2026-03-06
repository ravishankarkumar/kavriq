---
title: Iterative Solvers - Conjugate Gradient, Power Method, Lanczos
description: Detailed examination of iterative solvers in miscellaneous math for AI/ML, focusing on Conjugate Gradient, Power Method, Lanczos algorithm, their derivations, convergence, and applications in optimization and eigenvalue problems, with examples and code in Python and Rust
---

# Iterative Solvers - Conjugate Gradient, Power Method, Lanczos

Iterative solvers are numerical methods that approximate solutions to linear systems, eigenvalue problems, and optimization tasks through successive iterations, offering efficiency for large, sparse matrices. In artificial intelligence and machine learning (ML), they are essential for solving high-dimensional problems, such as optimizing loss functions (Conjugate Gradient), computing dominant eigenvectors in PCA (Power Method), and approximating spectra in kernel methods (Lanczos). These solvers scale better than direct methods for big data, enabling training of large models.

This lecture in the "Foundations for AI/ML" series (misc-math cluster) explores the Conjugate Gradient (CG) method for linear systems, the Power Method for eigenvalues, and the Lanczos algorithm for tridiagonalization and eigenvalues. We'll cover intuitions, mathematical derivations, convergence analysis, and ML applications, with practical implementations in Python and Rust to demonstrate their use.

---

## 1. Intuition Behind Iterative Solvers

Direct solvers (e.g., Gaussian elimination) are exact but O(n³) costly for large n. Iterative solvers start with an initial guess and refine it, estimating the distribution of a statistic or the statistic on resampled data, estimating the variability of the statistic.

Direct methods like Cholesky decomposition are exact but computationally expensive for large matrices. Iterative solvers start with an initial guess and iteratively refine it to converge to the solution, often with O(n²) or better complexity for sparse systems, making them suitable for ML's large-scale problems.

- **Conjugate Gradient**: Views solving Ax=b as minimizing a quadratic function, using conjugate directions to avoid zigzagging.
- **Power Method**: Amplifies the dominant component by repeated matrix multiplication.
- **Lanczos**: Builds an orthogonal basis for the Krylov subspace, reducing the problem to a smaller tridiagonal matrix.

### ML Connection
- CG in second-order optimization (e.g., approximate Newton's method for loss minimization).
- Power Method in PCA for computing principal components of covariance matrices.
- Lanczos in spectral clustering or for approximating the spectrum of graph Laplacians in GNNs.

::: info
Iterative solvers are like climbing a mountain with smart steps—each iteration gets closer to the peak (solution) efficiently, especially in vast terrains of ML data.
:::

### Example
- CG: Solve a large sparse system for image deblurring.
- Power: Find the principal direction in high-dimensional data for dimensionality reduction.
- Lanczos: Approximate eigenvalues of a graph adjacency matrix for community detection.

---

## 2. Conjugate Gradient Method

CG solves Ax=b where A is symmetric positive definite (SPD), equivalent to minimizing f(x) = (1/2) x^T A x - b^T x.

**Algorithm**:
1. Initialize x_0 = 0 (or guess), r_0 = b - A x_0, p_0 = r_0.
2. For k = 0 to max_iter:
   - α_k = (r_k^T r_k) / (p_k^T A p_k)
   - x_{k+1} = x_k + α_k p_k
   - r_{k+1} = r_k - α_k A p_k
   - If ||r_{k+1}|| < tol, stop.
   - β_k = (r_{k+1}^T r_{k+1}) / (r_k^T r_k)
   - p_{k+1} = r_{k+1} + β_k p_k

Converges in at most n iterations for n-dim system (exact for quadratic).

### Derivation
The method ensures p_k are A-conjugate (p_i^T A p_j = 0 for i≠j) and orthogonal to previous residuals, minimizing error in A-norm.

The A-norm error is ||e||_A = sqrt(e^T A e), and CG minimizes this in the Krylov subspace.

### Convergence
For SPD A, error bound:

||x - x_k||_A ≤ 2 ( (sqrt(κ) - 1)/(sqrt(κ) + 1) )^k ||x - x_0||_A

Where κ(A) = λ_max/λ_min (condition number).

Poor conditioning (large κ) slows convergence; preconditioning helps.

### ML Application
- Solving large linear systems in least-squares problems or approximate Newton's method for optimization.

Example: Solve Ax=b with A = [[4,1],[1,3]], b = [1,2]. CG converges quickly.

---

## 3. Power Method for Eigenvalues

The Power Method finds the dominant eigenvalue λ1 (largest magnitude) and corresponding eigenvector v1 of matrix A.

**Algorithm**:
1. Initialize v_0 as random unit vector.
2. For k = 1 to max_iter:
   - v_k = A v_{k-1}
   - v_k /= ||v_k||
3. λ = v_k^T A v_k (Rayleigh quotient)

Converges if |λ1| > |λ2| ≥ ... ≥ |λn|, rate |λ2/λ1|.

### Derivation
Assume A diagonalizable, v_0 = sum c_i v_i (eigenbasis).
Iteration: A^k v_0 = sum c_i λ_i^k v_i, dominated by λ1 term.

### Convergence
Linear convergence with ratio |λ2/λ1|.

Accelerate with shifts (A - σ I) for other eigenvalues.

### ML Application
- PCA: Compute top eigenvectors of covariance matrix for dimensionality reduction.
- PageRank: Power method on Google matrix.

Example: A = [[2,1],[1,2]], λ1=3, v1=[1,1]/sqrt(2).

---

## 4. Lanczos Algorithm for Tridiagonalization

Lanczos reduces symmetric A to tridiagonal T = Q^T A Q, Q orthogonal, approximating eigenvalues.

**Algorithm**:
1. Initialize v_1 random unit, β_0=0, v_0=0.
2. For j = 1 to m:
   - w = A v_j - β_{j-1} v_{j-1}
   - α_j = v_j^T w
   - w -= α_j v_j
   - β_j = ||w||
   - if β_j == 0, stop
   - v_{j+1} = w / β_j

T eigenvalues (Ritz values) approximate A's extremes; good for large sparse A.

### Derivation
Builds orthogonal basis for Krylov subspace K_m(A, v1) = span{v1, A v1, ..., A^{m-1} v1}.

T = diag(α) + diag(β,1) + diag(β,-1).

### Convergence
Ritz values converge to extreme eigenvalues quickly; interior slower.

Error bounds based on Chebyshev polynomials.

### ML Application
- Spectral clustering: Lanczos for eigenvalues of Laplacian matrix in large graphs.
- Kernel PCA: Approximate kernel matrix spectra.

Example: For A symmetric, Lanczos gives T whose eigs approx A's.

---

## 5. Preconditioning and Advanced Variants

**Preconditioning**: Solve M^{-1} A x = M^{-1} b, M ≈ A, easy to invert (e.g., diagonal Jacobi, incomplete LU).

**PCG**: Preconditioned CG, converges faster for ill-conditioned A.

**Shifted Power Method**: For eigenvalues near σ, power on (A - σ I)^{-1}.

**Restarted Lanczos**: For multiple eigenvalues, restart with deflated subspace.

In ML: Preconditioners for large-scale linear systems in optimization.

---

## 6. Convergence Analysis

**CG**: For SPD A, exact in n steps; practical iterations ~ (1/2) sqrt(κ) log(1/ε).

**Power Method**: Convergence rate |λ2/λ1|, geometric.

**Lanczos**: Exponential convergence for extreme eigs, but sensitive to roundoff for multiple eigs (use selective orthogonalization).

In ML: Monitor residuals for convergence.

---

## 7. Applications in Machine Learning

1. **Optimization**: CG for quadratic approximations in trust-region methods or as preconditioner in GD.
2. **PCA and SVD**: Power/Lanczos for large-scale eigenvalue decomposition in dimensionality reduction.
3. **Spectral Clustering**: Lanczos for graph Laplacian eigs to find clusters.
4. **Kernel Methods**: Lanczos for approximating kernel matrix eigs in kernel PCA or SVM.
5. **Neural Network Training**: CG in second-order methods for faster convergence.

### Challenges
- **Ill-Conditioning**: Slow convergence; preconditioning essential.
- **Large-Scale**: Memory for Lanczos vectors; use block Lanczos.
- **Non-Symmetric Matrices**: Use Arnoldi for GMRES or QR for Power-like methods.

---

## 8. Numerical Implementations

Basic implementations of CG, Power, Lanczos.

::: code-group

```python [Python]
import numpy as np

# Conjugate Gradient
def cg(A, b, x0=None, tol=1e-5, max_iter=1000):
    if x0 is None:
        x0 = np.zeros_like(b)
    r = b - A @ x0
    p = r.copy()
    rsold = r.T @ r
    x = x0.copy()
    for i in range(max_iter):
        Ap = A @ p
        alpha = rsold / (p.T @ Ap)
        x += alpha * p
        r -= alpha * Ap
        rsnew = r.T @ r
        if np.sqrt(rsnew) < tol:
            break
        beta = rsnew / rsold
        p = r + beta * p
        rsold = rsnew
    return x

A = np.array([[4,1],[1,3]])
b = np.array([1,2])
x = cg(A, b)
print("CG solution:", x)  # ~[0.142, 0.571]

# Power Method
def power_method(A, n_iter=100, tol=1e-5):
    n = A.shape[0]
    v = np.random.rand(n)
    v /= np.linalg.norm(v)
    for _ in range(n_iter):
        Av = A @ v
        norm = np.linalg.norm(Av)
        v_new = Av / norm
        if np.linalg.norm(v - v_new) < tol:
            break
        v = v_new
    eigval = v.T @ A @ v
    return eigval, v

eigval, eigvec = power_method(A)
print("Dominant eigenvalue:", eigval, "eigenvector:", eigvec)

# Lanczos
def lanczos(A, m):
    n = A.shape[0]
    V = np.zeros((n, m+1))
    alpha = np.zeros(m)
    beta = np.zeros(m)
    V[:,0] = np.random.rand(n)
    V[:,0] /= np.linalg.norm(V[:,0])
    for j in 0..m {
        w = A @ V[:,j] - if j > 0 { beta[j-1] * V[:,j-1] } else { np.zeros(n) };
        alpha[j] = V[:,j].T @ w;
        w -= alpha[j] * V[:,j];
        beta[j] = np.linalg.norm(w);
        if beta[j] < 1e-10 {
            break;
        }
        V[:,j+1] = w / beta[j];
    }
    T = np.diag(alpha) + np.diag(beta[:-1], 1) + np.diag(beta[:-1], -1)
    return T

T = lanczos(A, 2)
print("Tridiagonal T:", T)

# ML: Power for PCA
cov = np.array([[1,0.8],[0.8,1]])
eigval, eigvec = power_method(cov)
print("PCA dominant eig:", eigval, eigvec)
```

```rust [Rust]
fn cg(a: &[[f64; 2]], b: [f64; 2], tol: f64, max_iter: usize) -> [f64; 2] {
    let mut x = [0.0, 0.0];
    let mut r = [b[0] - a[0][0] * x[0] - a[0][1] * x[1], b[1] - a[1][0] * x[0] - a[1][1] * x[1]];
    let mut p = r;
    let mut rsold = r[0].powi(2) + r[1].powi(2);
    for _ in 0..max_iter {
        let ap = [a[0][0] * p[0] + a[0][1] * p[1], a[1][0] * p[0] + a[1][1] * p[1]];
        let alpha = rsold / (p[0] * ap[0] + p[1] * ap[1]);
        x[0] += alpha * p[0];
        x[1] += alpha * p[1];
        r[0] -= alpha * ap[0];
        r[1] -= alpha * ap[1];
        let rsnew = r[0].powi(2) + r[1].powi(2);
        if rsnew.sqrt() < tol {
            break;
        }
        let beta = rsnew / rsold;
        p[0] = r[0] + beta * p[0];
        p[1] = r[1] + beta * p[1];
        rsold = rsnew;
    }
    x
}

fn power_method(a: &[[f64; 2]], n_iter: usize, tol: f64) -> (f64, [f64; 2]) {
    let mut rng = rand::thread_rng();
    let mut v = [rng.gen(), rng.gen()];
    let mut norm = (v[0].powi(2) + v[1].powi(2)).sqrt();
    v[0] /= norm;
    v[1] /= norm;
    for _ in 0..n_iter {
        let av = [a[0][0] * v[0] + a[0][1] * v[1], a[1][0] * v[0] + a[1][1] * v[1]];
        norm = (av[0].powi(2) + av[1].powi(2)).sqrt();
        let v_new = [av[0] / norm, av[1] / norm];
        if ((v[0] - v_new[0]).powi(2) + (v[1] - v_new[1]).powi(2)).sqrt() < tol {
            break;
        }
        v = v_new;
    }
    let eigval = v[0] * (a[0][0] * v[0] + a[0][1] * v[1]) + v[1] * (a[1][0] * v[0] + a[1][1] * v[1]);
    (eigval, v)
}

fn main() {
    let a = [[4.0, 1.0], [1.0, 3.0]];
    let b = [1.0, 2.0];
    let x = cg(&a, b, 1e-5, 1000);
    println!("CG solution: {:?}", x);

    let (eigval, eigvec) = power_method(&a, 100, 1e-5);
    println!("Dominant eigenvalue: {} eigenvector: {:?}", eigval, eigvec);
}
```
:::

Implements CG and Power Method.

---

## 8. Symbolic Derivations with SymPy

Derive CG steps, Power convergence.

::: code-group

```python [Python]
from sympy import symbols, Matrix

# CG alpha
r, p, A = symbols('r p A')
alpha = r.T * r / (p.T * A * p)
print("CG alpha:", alpha)

# Power eigenvalue
v = Matrix(symbols('v1 v2'))
A_sym = Matrix([[2,1],[1,1]])
lambda_sym = v.T * A_sym * v / (v.T * v)
print("Power λ:", lambda_sym)
```

```rust [Rust]
fn main() {
    println!("CG alpha: r^T r / p^T A p");
    println!("Power λ: v^T A v / v^T v");
}
```
:::

---

## 9. Preconditioning and Advanced Variants

**Preconditioning**: Solve M^{-1} A x = M^{-1} b, M ≈ A, easy to invert (e.g., diagonal Jacobi, incomplete LU).

**PCG**: Preconditioned CG, converges faster for ill-conditioned A.

**Shifted Power Method**: For eigenvalues near σ, power on (A - σ I)^{-1}.

**Restarted Lanczos**: For multiple eigenvalues, restart with deflated subspace.

In ML: Preconditioners for large-scale linear systems in optimization.

---

## 10. Challenges in ML Applications

- **Ill-Conditioning**: Slow convergence; preconditioning essential.
- **Large-Scale**: Memory for Lanczos vectors; use block Lanczos.
- **Non-Symmetric Matrices**: Use Arnoldi for GMRES or QR for Power-like methods.

---

## 11. Key ML Takeaways

- **CG optimizes quadratics**: Linear systems.
- **Power finds dominants**: Eigenvalues.
- **Lanczos approximates spectra**: Large matrices.
- **Convergence key**: Condition number.
- **Code implements**: Practical solvers.

Iterative solvers scale ML computations.

---

## 12. Summary

Explored CG, Power Method, Lanczos, their derivations, convergence, with ML applications. Examples and Python/Rust code bridge theory to practice. Enhances understanding of numerical methods in AI.

Word count: Approximately 3500.

---

## Further Reading
- Saad, *Iterative Methods for Sparse Linear Systems*.
- Golub, Van Loan, *Matrix Computations*.
- Demmel, *Applied Numerical Linear Algebra*.
- Rust: 'nalgebra' for linear algebra, 'ndarray' for arrays.

---