---
title: Functions and Limits - The Language of Change
description: Deep dive into functions and limits for AI/ML, with intuition, examples, and code in Python and Rust
---

# Functions and Limits: The Language of Change

Calculus is the mathematics of change, and at its foundation lie **functions and limits**. In machine learning, models are functions that transform inputs into outputs, and limits give us the language to describe how these transformations behave as inputs vary. Understanding these ideas is crucial for grasping derivatives, optimization, and the smoothness of learning algorithms.

---

## 1. Functions: The Building Blocks of ML

A **function** maps inputs to outputs. Formally:  
$$
f: X \to Y, \quad y = f(x)
$$

where $X$ is the input domain, $Y$ is the output range.

### ML Connection
- In **linear regression**, $f(x) = w^T x + b$.  
- In **neural networks**, $f(x)$ is a composition of affine transformations and nonlinear activations.  
- In **classification**, $f(x)$ may output probabilities via a softmax function.

::: info
Think of every ML model as just a **fancy function**: it takes feature vectors in and produces predictions out.
:::

### Example
- $f(x) = x^2$ maps real numbers to their squares.  
- $f(x) = \text{ReLU}(x) = \max(0, x)$ is a key activation in deep learning.

---

## 2. Limits: Understanding Behavior at Boundaries

A **limit** describes what a function approaches as input approaches a point.

Formally,  
$$
\lim_{x \to a} f(x) = L
$$  
means that as $x$ gets arbitrarily close to $a$, $f(x)$ gets arbitrarily close to $L$.

### Why Important?
- **Gradients** (derivatives) are defined using limits.  
- **Continuity** relies on limits.  
- ML algorithms assume smooth loss functions for optimization.

---

## 3. Types of Limits

1. **Left-hand limit**:  
$$
\lim_{x \to a^-} f(x)
$$  
2. **Right-hand limit**:  
$$
\lim_{x \to a^+} f(x)
$$  
3. **At infinity**:  
$$
\lim_{x \to \infty} f(x)
$$  

### ML Example
- Sigmoid function:  
  $$
  \sigma(x) = \frac{1}{1+e^{-x}}  
  $$  
  - As $x \to \infty$, $\sigma(x) \to 1$.  
  - As $x \to -\infty$, $\sigma(x) \to 0$.

---

## 4. Continuity and Discontinuities

A function $f(x)$ is **continuous** at $x=a$ if:
1. $f(a)$ is defined,  
2. $\lim_{x \to a} f(x)$ exists,  
3. $\lim_{x \to a} f(x) = f(a)$.  

### Types of Discontinuities
- **Removable**: hole in the graph.  
- **Jump**: left and right limits differ.  
- **Infinite**: vertical asymptote.

### ML Connection
- **ReLU** is continuous but not differentiable at 0.  
- Smoothness of loss landscapes determines training stability.

---

## 5. Exploring Limits Numerically

We can approximate limits by evaluating function values near a point.

::: code-group

```python [Python]
import numpy as np

def relu(x):
    return np.maximum(0, x)

xs = np.linspace(-0.01, 0.01, 5)
ys = relu(xs)
print("Inputs:", xs)
print("Outputs:", ys)

# Sigmoid at extremes
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

print("sigmoid(100) =", sigmoid(100))
print("sigmoid(-100) =", sigmoid(-100))
```

```rust [Rust]
use ndarray::array;

fn relu(x: f64) -> f64 {
    if x > 0.0 { x } else { 0.0 }
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

fn main() {
    let xs = array![-0.01, -0.005, 0.0, 0.005, 0.01];
    let ys: Vec<f64> = xs.iter().map(|&v| relu(v)).collect();
    println!("Inputs: {:?}", xs);
    println!("Outputs: {:?}", ys);

    println!("sigmoid(100) = {}", sigmoid(100.0));
    println!("sigmoid(-100) = {}", sigmoid(-100.0));
}
```
:::

---

## 6. Visualization (Conceptual)

- Graph of $f(x) = 1/x$ near 0 shows infinite discontinuity.  
- Graph of sigmoid shows asymptotes at 0 and 1.  
- Graph of ReLU shows a sharp corner at 0.

These visuals help connect the formal math with intuition.

---

## 7. Key ML Takeaways

- **Functions** = models, loss functions, activations.  
- **Limits** = foundation of derivatives, continuity, and optimization.  
- **Smoothness** matters: smoother functions → easier optimization.  
- **Asymptotic behavior** (like sigmoid saturation) influences learning.

---

## 8. Summary

In this lesson, we built intuition for **functions and limits**, grounding them in ML context.  
We saw how to compute limits numerically, explored discontinuities, and connected continuity to model training.  

This foundation prepares us for the next step: **Derivatives: Measuring Change**.

---

## Further Reading
- Stewart, *Calculus: Early Transcendentals* (Chapter 2)  
- Goodfellow, Bengio, Courville, *Deep Learning* (Chapter 6: Deep Feedforward Networks)  
- MIT OpenCourseWare: *Single Variable Calculus*  

---
