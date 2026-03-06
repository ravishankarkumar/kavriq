---
title: Scalars, Vectors, and Matrices - The Language of Data
description: Introduction to scalars, vectors, and matrices for ML
---

# Scalars, Vectors, and Matrices: The Language of Data

Machine Learning (ML) is built on data, and the language of data is **linear algebra**. Almost every ML algorithm represents information using **scalars**, **vectors**, or **matrices**. Understanding these basic building blocks is essential before moving to advanced concepts.


## Scalars

A **scalar** is a single number. Scalars often represent:
- A single feature value (e.g., height = 170).
- A model parameter (e.g., learning rate $\eta = 0.01$).

Formally, scalars are just real numbers:  
$$
a \in \mathbb{R}
$$


## Vectors

A **vector** is an ordered list of numbers, often representing a data point or a set of features.

$$
\mathbf{x} = 
\begin{bmatrix}
x_1 \\
x_2 \\
x_3
\end{bmatrix}
$$

- Each entry $x_i$ is a **feature value**.
- A vector with $n$ entries lives in $n$-dimensional space: $\mathbf{x} \in \mathbb{R}^n$.

::: info Explanation of Vectors
Think of a vector as a **row of values** in your dataset.  
- If you have 3 features (height, weight, age), one person's data = a vector of 3 numbers.  
- In ML, feature vectors are the input to models.
:::

**Mini example:**  
If we describe a student with height = 170 cm, weight = 65 kg, and age = 20:  
$$
\mathbf{x} = [170, 65, 20]^T
$$


## Matrices

A **matrix** is a 2D array of numbers. In ML, matrices usually represent **datasets**.

$$
X =
\begin{bmatrix}
x_{11} & x_{12} & x_{13} \\
x_{21} & x_{22} & x_{23} \\
x_{31} & x_{32} & x_{33}
\end{bmatrix}
$$

- Each **row** = one data point (a feature vector).  
- Each **column** = values of one feature across all data points.  

::: info Explanation of Matrices
A dataset with $m$ samples and $n$ features is represented as an $m \times n$ matrix.  
- $m$ = number of rows = number of examples.  
- $n$ = number of columns = number of features.  
So: $X \in \mathbb{R}^{m \times n}$.
:::

**Mini example:**  
Suppose we record 3 students with features [height, weight, age]:  

$$
X =
\begin{bmatrix}
170 & 65 & 20 \\
180 & 75 & 22 \\
160 & 55 & 19
\end{bmatrix}
$$

- 3 rows = 3 students (examples).  
- 3 columns = 3 features (height, weight, age).  


## Hands-on with Python and Rust

::: code-group

```python [Python]
import numpy as np

# A scalar
a = 3.14

# A vector (student features: height, weight, age)
x = np.array([170, 65, 20])

# A matrix (3 students × 3 features)
X = np.array([
    [170, 65, 20],
    [180, 75, 22],
    [160, 55, 19]
])

print("Scalar:", a)
print("Vector:", x)
print("Matrix:\n", X)
```

```rust [Rust]
use ndarray::array;

fn main() {
    // A scalar
    let a: f64 = 3.14;

    // A vector (student features: height, weight, age)
    let x = array![170.0, 65.0, 20.0];

    // A matrix (3 students × 3 features)
    let X = array![
        [170.0, 65.0, 20.0],
        [180.0, 75.0, 22.0],
        [160.0, 55.0, 19.0]
    ];

    println!("Scalar: {}", a);
    println!("Vector: {:?}", x);
    println!("Matrix:\n{:?}", X);
}
```

:::


## Connection to ML

- **Scalars** → individual feature values or hyperparameters.  
- **Vectors** → single data points (feature vectors).  
- **Matrices** → entire datasets.  

Every ML algorithm (from linear regression to neural networks) starts by manipulating these structures. Mastering them is the first step toward understanding ML.


## Next Steps

Continue to [Vector Operations: Dot Product, Norms, and Distances](/ml-essentials/maths-for-aiml/linear-algebra/vector-operations).
