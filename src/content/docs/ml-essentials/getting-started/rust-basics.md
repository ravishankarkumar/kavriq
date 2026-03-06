---
title: Rust Basics
description: Introduction to Rust programming for ML
---
# Rust Basics

This section covers essential Rust programming concepts for machine learning (ML) tasks. You'll learn syntax, ownership, and data structures, preparing for ML labs with `linfa` and `tch-rs`. A basic familiarity with Rust is needed. Deepen your skills with *The Rust Programming Language* (The Book) and *Programming Rust*. 
<!-- Write the code yourself to learn Rust, but you can refer to examples in our GitHub repository: [https://github.com/ravishankarkumar/aiunderthehood-sample-code](https://github.com/ravishankarkumar/aiunderthehood-sample-code). -->

## Why Rust for ML?

Rust offers performance, memory safety, and a growing ML ecosystem, ideal for AI/ML. Its ownership model ensures robust code, and its speed rivals C++ for neural network training.

## Basic Syntax

Rust's syntax is clear. This program sums a vector, showing variables, loops, and functions:

```rust
fn main() {
    let numbers = vec![1, 2, 3, 4, 5];
    let mut sum = 0;
    for num in numbers {
        sum += num;
    }
    println!("Sum: {}", sum);
}
```

- **Variables**: `let` declares immutable variables; `mut` allows mutation (e.g., `let mut sum`).
- **Vectors**: `vec![]` stores data, used in ML for datasets.
- **Loops**: `for` iterates over collections.
- **Functions**: `fn` defines functions, with `main` as the entry point.

Run with `cargo run` to see “Sum: 15”.

## Ownership and Borrowing

Rust's ownership ensures memory safety without a garbage collector:

- **Ownership**: Each value has one owner; it's dropped when out of scope.
- **Borrowing**: Use `&` (immutable) or `&mut` (mutable) to borrow values.

Example:

```rust
fn main() {
    let data = vec![1.0, 2.0, 3.0];
    let mean = compute_mean(&data); // Borrow immutably
    println!("Mean: {}", mean);
}

fn compute_mean(values: &Vec<f64>) -> f64 {
    let sum: f64 = values.iter().sum();
    sum / values.len() as f64
}
```

This computes a vector's mean, borrowing `data` safely, crucial for ML data handling.

## Data Structures for ML

ML uses arrays, matrices, and datasets. Rust provides:

- **Vectors (**`Vec<T>`**)**: Dynamic arrays for feature vectors.
- **Arrays (**`[T; N]`**)**: Fixed-size arrays for static data.
- **Structs**: Custom types for ML models.

Example dataset struct:

```rust
struct Dataset {
    features: Vec<Vec<f64>>,
    labels: Vec<f64>,
}

fn main() {
    let dataset = Dataset {
        features: vec![vec![1.0, 2.0], vec![3.0, 4.0]],
        labels: vec![0.0, 1.0],
    };
    println!("Dataset size: {}", dataset.features.len());
}
```

This mimics ML datasets, used with `linfa` and `ndarray`.

## Lab: Vector Operations

Practice Rust by computing the Euclidean distance between two vectors, a key ML operation.

1. **Create** `examples/vector_distance/main.rs` in your `rust_ml_tutorial` project:
    ```rust
    use ndarray::{Array1, array};

    fn main() {
        let v1 = array![1.0, 2.0, 3.0];
        let v2 = array![4.0, 5.0, 6.0];
        let distance = euclidean_distance(&v1, &v2).expect("Vectors must have equal length");
        println!("Euclidean Distance: {:.3}", distance);
    }

    fn euclidean_distance(v1: &Array1<f64>, v2: &Array1<f64>) -> Result<f64, &'static str> {
        if v1.len() != v2.len() {
            return Err("Vectors must have equal length");
        }
        let sum: f64 = v1.iter().zip(v2.iter()).map(|(&a, &b)| (a - b).powi(2)).sum();
        Ok(sum.sqrt())
    }
    ```

2. **Dependencies**:
    Add the following to `Cargo.toml` under `[dependencies]`:
    <LibraryVersions />

3. **Run**:
    Write the code above manually to understand vector operations. 
    <!-- Alternatively, run the example from the repository:
    ```bash
    git clone https://github.com/ravishankarkumar/aiunderthehood-sample-code.git
    cd aiunderthehood-sample-code/rust
    cargo run --example rust-basics
    ``` -->
   **Expected Output**:
    ```
    Euclidean Distance: 5.196
    ```

This lab builds skills for ML computations, using `ndarray` for efficient vector operations.

## Learning from Official Resources

Deepen your Rust knowledge with:

- **The Rust Programming Language (The Book)**: Free official guide covering syntax, ownership, and more. [doc.rust-lang.org/book](https://doc.rust-lang.org/book)
- **Programming Rust**: Comprehensive book by Blandy, Orendorff, and Tindall, great for ML applications. Available for purchase or through libraries.

## Next Steps

Move to [First ML Lab](/ml-essentials/getting-started/first-lab) to build your first ML model, or revisit [Setup](/ml-essentials/getting-started/setup).

## Further Reading

- *The Rust Programming Language*: [doc.rust-lang.org/book](https://doc.rust-lang.org/book)
- *Programming Rust* by Blandy, Orendorff, and Tindall
- Rust Documentation: [doc.rust-lang.org](https://doc.rust-lang.org)