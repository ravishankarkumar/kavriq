---
title: Python Basics
description: Introduction to Python programming for ML
layout: ../../../layouts/TutorialPage.astro
---
# Python Basics

This section covers essential Python programming concepts for machine learning (ML) tasks. You'll learn syntax, data types, and libraries, preparing for ML labs with `numpy`, `pandas`, and `scikit-learn`. Basic familiarity with Python helps, but no ML experience is required. 
<!-- Write the code yourself to learn Python, but you can refer to examples in our GitHub repository: [https://github.com/ravishankarkumar/aiunderthehood-sample-code](https://github.com/ravishankarkumar/aiunderthehood-sample-code). -->

## Why Python for ML?

Python dominates AI/ML because of its simplicity, readability, and massive ecosystem. Its high-level syntax enables rapid prototyping, while libraries like `numpy`, `pandas`, and `pytorch` handle performance-critical operations in optimized C/C++ under the hood.

## Basic Syntax

Python's syntax is simple and beginner-friendly. This program sums a list, showing variables, loops, and functions:

```python
def main():
    numbers = [1, 2, 3, 4, 5]
    total = 0
    for num in numbers:
        total += num
    print("Sum:", total)

if __name__ == "__main__":
    main()
```

- **Variables**: Declared dynamically, no type annotations needed (`total = 0`).
- **Lists**: `[]` stores ordered data, commonly used for datasets.
- **Loops**: `for` iterates directly over items.
- **Functions**: `def` defines functions, with `main()` as convention.

Run with `python main.py` to see “Sum: 15”.

## Data Structures for ML

ML tasks require arrays, matrices, and datasets. Python provides:

- **Lists (`list`)**: Flexible collections for data.
- **Tuples (`tuple`)**: Immutable sequences.
- **Dictionaries (`dict`)**: Key-value pairs for metadata.
- **NumPy Arrays**: Efficient multidimensional arrays (preferred for ML).

Example dataset with Python structures:

```python
dataset = {
    "features": [[1.0, 2.0], [3.0, 4.0]],
    "labels": [0, 1]
}

print("Dataset size:", len(dataset["features"]))
```

This mimics ML datasets, later used with `numpy`, `pandas`, or `scikit-learn`.

## Lab: Vector Operations with NumPy

Practice Python by computing the Euclidean distance between two vectors, a key ML operation.

1. **Create** `vector_distance.py` in your project:
    ```python
    import numpy as np

    def euclidean_distance(v1, v2):
        v1, v2 = np.array(v1), np.array(v2)
        if v1.shape != v2.shape:
            raise ValueError("Vectors must have equal length")
        return np.sqrt(np.sum((v1 - v2) ** 2))

    if __name__ == "__main__":
        v1 = [1.0, 2.0, 3.0]
        v2 = [4.0, 5.0, 6.0]
        distance = euclidean_distance(v1, v2)
        print(f"Euclidean Distance: {distance:.3f}")
    ```

2. **Dependencies**:
    Install NumPy if not already installed:
    ```bash
    pip install numpy
    ```

3. **Run**:
    ```bash
    python vector_distance.py
    ```
   **Expected Output**:
    ```
    Euclidean Distance: 5.196
    ```

This builds skills for ML computations, using NumPy for efficient vector operations.

## Learning from Official Resources

Deepen your Python knowledge with:

- **Python Official Tutorial**: [docs.python.org/tutorial](https://docs.python.org/3/ml-essentials/)
- **Automate the Boring Stuff with Python** by Al Sweigart: Beginner-friendly, practical examples.
- **Effective Python** by Brett Slatkin: Best practices for writing clean, efficient code.

## Next Steps

Move to [First ML Lab](/ml-essentials/getting-started/first-lab) to build your first ML model, or revisit [Setup](/ml-essentials/getting-started/setup).

## Further Reading

- Python Documentation: [docs.python.org](https://docs.python.org/3/)
- *Automate the Boring Stuff with Python* by Al Sweigart
- *Effective Python* by Brett Slatkin
