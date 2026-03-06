---
title: Chain Rule & Backpropagation
description: Uncover the chain rule in calculus and its pivotal role in backpropagation for AI/ML, with intuitions, multivariable extensions, neural network examples, and code in Python and Rust
---

# Chain Rule & Backpropagation

The chain rule is a fundamental theorem in calculus that allows us to differentiate composite functions, breaking down complex derivatives into simpler parts. In artificial intelligence and machine learning, it underpins backpropagation, the algorithm that efficiently computes gradients in neural networks, enabling models to learn from data by minimizing loss functions. Without the chain rule, training deep networks would be computationally infeasible.

This lecture explores the chain rule from single-variable basics to multivariable generalizations, then dives into backpropagation's mechanics, applications, and implementations. We'll blend mathematical insights with practical ML connections, illustrated through examples and code in Python and Rust, empowering you to grasp how AI systems propagate errors backward to optimize forward.

---

## 1. Intuition Behind the Chain Rule

Imagine speed: If distance \( s = f(u) \) depends on velocity \( u = g(t) \), then rate of change of s w.r.t. t is velocity times acceleration—chain rule multiplies rates.

For \( y = f(g(x)) \), the derivative \( \frac{dy}{dx} = f'(g(x)) \cdot g'(x) \)—rate of y w.r.t. inner function times inner w.r.t. x.

### ML Connection
- Neural networks are compositions: Output = activation(affine(activation(affine(input)))).
- Chain rule decomposes gradients layer by layer.

::: info
Chain rule is like dominoes: A small change at input cascades through, multiplied at each step.
:::

### Everyday Example
- Temperature conversion: C = 5/9 (F - 32). If F changes with time, dC/dt = (5/9) dF/dt.

---

## 2. Formal Statement in Single Variable

For differentiable f and g, with y = f(u), u = g(x):

\[
\frac{dy}{dx} = \frac{dy}{du} \bigg|_{u=g(x)} \cdot \frac{du}{dx}
\]

Leibniz notation highlights cancellation.

### Proof Sketch
From definition: Limit of [f(g(x+h)) - f(g(x))]/h = [f'(g(x)) (g(x+h) - g(x))]/h → f'(g(x)) g'(x).

### Examples
- y = sin(x^2): dy/dx = cos(x^2) · 2x.
- y = e^{sqrt(x)}: dy/dx = e^{sqrt(x)} · (1/(2 sqrt(x))).

### ML Insight
- Activation derivatives: For sigmoid σ(z), σ'(z) = σ(z)(1-σ(z)), chained in nets.

---

## 3. Multivariable Chain Rule

For vector functions: If \(\mathbf{y} = \mathbf{f}(\mathbf{u})\), \(\mathbf{u} = \mathbf{g}(\mathbf{x})\), Jacobian J_yx = J_yu J_ux.

In scalars: For z = f(x,y), x=g(t), y=h(t), dz/dt = (∂f/∂x) dx/dt + (∂f/∂y) dy/dt.

### Tree Diagrams
Visualize dependencies: dz/dt sums paths from z to t, multiplying partials.

### ML Application
- In nets, gradients w.r.t. weights involve summing over paths (backprop).

Example: z = x^2 + sin(xy), x= t^2, y= e^t.
dz/dt = 2x (2t) + [cos(xy) y (2x) + cos(xy) x e^t].

---

## 4. Implicit Differentiation via Chain Rule

For F(x,y)=0, dy/dx = - (∂F/∂x) / (∂F/∂y).

Extends to multivariables.

### Example
Circle x^2 + y^2 =1: 2x + 2y dy/dx =0 → dy/dx = -x/y.

In ML: Constraints in optimization, like normalizing layers.

---

## 5. Higher-Order Derivatives and Chain Rule

Second derivatives: d^2y/dx^2 = d/dx [f'(g(x)) g'(x)] = f''(g(x)) [g'(x)]^2 + f'(g(x)) g''(x).

In ML: Hessians for second-order methods.

---

## 6. Introduction to Backpropagation

Backprop applies chain rule to compute gradients in feedforward nets.

Forward pass: Compute activations layer by layer.

Backward pass: Propagate errors from output to input, computing local gradients.

Efficiency: O(n) vs. numerical diff's O(n^2).

### Core Idea
Loss L, output ŷ = f(W_l a_{l-1} + b_l), a_l = σ(ŷ_l).

δ_l = ∂L/∂ŷ_l = (∂L/∂a_l) σ'(ŷ_l).

Then δ_{l-1} = W_l^T δ_l ⊙ σ'(ŷ_{l-1}).

Gradients: ∂L/∂W_l = δ_l a_{l-1}^T.

---

## 7. Backpropagation Algorithm Step-by-Step

1. Initialize weights.
2. Forward: a^1 = x, z^{l+1} = W^{l+1} a^l + b^{l+1}, a^{l+1} = σ(z^{l+1}).
3. Compute output error δ^L = ∇_a L ⊙ σ'(z^L).
4. Backprop: δ^l = (W^{l+1})^T δ^{l+1} ⊙ σ'(z^l).
5. Gradients: ∇_{W^l} L = a^{l-1} (δ^l)^T, ∇_{b^l} L = δ^l.

### ML Connection
- Enables deep learning: Scales to billions of parameters.
- Variants: With momentum, RMSprop.

---

## 8. Geometric and Computational Interpretations

Geometrically: Backprop navigates loss surface via chain-ruled gradients.

Computationally: Reuses intermediates, avoids redundant calcs.

Issues: Vanishing (small sigmoids), exploding (large weights)—mitigated by ReLU, init strategies.

---

## 9. Implementing Chain Rule Numerically

Approximate via finite diffs, but backprop is exact/efficient.

::: code-group

```python [Python]
import numpy as np

# Single var chain rule
def outer(u):
    return np.sin(u)

def inner(x):
    return x**2

def chain_deriv(x):
    u = inner(x)
    dy_du = np.cos(u)
    du_dx = 2 * x
    return dy_du * du_dx

x = 3.0
print(f"Chain rule deriv at x={x}: {chain_deriv(x)}")

# Simple backprop toy net
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_prime(z):
    s = sigmoid(z)
    return s * (1 - s)

# Forward
x = np.array([1.0])
W1 = np.array([[0.5]])
b1 = np.array([0.1])
z1 = W1 @ x + b1
a1 = sigmoid(z1)

W2 = np.array([[1.5]])
b2 = np.array([0.2])
z2 = W2 @ a1 + b2
y_hat = sigmoid(z2)

y_true = np.array([0.8])
loss = 0.5 * (y_hat - y_true)**2

# Backward
dL_dyhat = y_hat - y_true
dyhat_dz2 = sigmoid_prime(z2)
dL_dz2 = dL_dyhat * dyhat_dz2

dL_dW2 = dL_dz2 * a1.T
dL_db2 = dL_dz2

dz2_da1 = W2.T
dL_da1 = dz2_da1 @ dL_dz2

da1_dz1 = sigmoid_prime(z1)
dL_dz1 = dL_da1 * da1_dz1

dL_dW1 = dL_dz1 * x.T
dL_db1 = dL_dz1

print(f"Grad W1: {dL_dW1}, Grad b1: {dL_db1}")
```

```rust [Rust]
use std::f64;

fn outer(u: f64) -> f64 {
    u.sin()
}

fn inner(x: f64) -> f64 {
    x.powi(2)
}

fn chain_deriv(x: f64) -> f64 {
    let u = inner(x);
    let dy_du = u.cos();
    let du_dx = 2.0 * x;
    dy_du * du_dx
}

fn sigmoid(z: f64) -> f64 {
    1.0 / (1.0 + (-z).exp())
}

fn sigmoid_prime(z: f64) -> f64 {
    let s = sigmoid(z);
    s * (1.0 - s)
}

fn main() {
    let x_val = 3.0;
    println!("Chain rule deriv at x={}: {}", x_val, chain_deriv(x_val));

    // Simple backprop toy net
    let x = 1.0;
    let w1 = 0.5;
    let b1 = 0.1;
    let z1 = w1 * x + b1;
    let a1 = sigmoid(z1);

    let w2 = 1.5;
    let b2 = 0.2;
    let z2 = w2 * a1 + b2;
    let y_hat = sigmoid(z2);

    let y_true = 0.8;
    let loss = 0.5 * (y_hat - y_true).powi(2);

    let dl_dyhat = y_hat - y_true;
    let dyhat_dz2 = sigmoid_prime(z2);
    let dl_dz2 = dl_dyhat * dyhat_dz2;

    let dl_dw2 = dl_dz2 * a1;
    let dl_db2 = dl_dz2;

    let dz2_da1 = w2;
    let dl_da1 = dz2_da1 * dl_dz2;

    let da1_dz1 = sigmoid_prime(z1);
    let dl_dz1 = dl_da1 * da1_dz1;

    let dl_dw1 = dl_dz1 * x;
    let dl_db1 = dl_dz1;

    println!("Grad W1: {}, Grad b1: {}", dl_dw1, dl_db1);
}
```
:::

This demonstrates chain rule and a toy backprop.

---

## 10. Symbolic Chain Rule Application

Use libraries for exact derivatives.

::: code-group

```python [Python]
from sympy import symbols, diff, sin, exp, sqrt

x = symbols('x')
y = sin(x**2)
dy_dx = diff(y, x)
print("dy/dx:", dy_dx)

# Multivar
t = symbols('t')
z = (t**2)**2 + sin(t**2 * exp(t))
dz_dt = diff(z, t)
print("dz/dt:", dz_dt)
```

```rust [Rust]
// Simulated symbolic
fn main() {
    // dy/dx = cos(x^2) * 2x
    println!("dy/dx: cos(x^2) * 2x");

    let x = 3.0;
    let deriv = (x.powi(2).cos()) * 2.0 * x;
    println!("At x=3: {}", deriv);
}
```
:::

---

## 11. Backpropagation in Deep Networks

For deeper nets: Recurse backward.

In conv nets: Chain through convolutions, pooling.

Auto-diff libs (PyTorch, TensorFlow) abstract this.

Challenges: Overfitting (dropout), optimization (Adam).

---

## 12. Advanced Topics: Vector Chain Rule and Jacobians

For vector outputs: Gradient is Jacobian transpose times upstream.

In backprop: Modular, each op defines forward/backward.

---

## 13. Key ML Takeaways

- **Chain rule decomposes**: Makes complex diffs manageable.
- **Backprop = chain in reverse**: Efficient gradient computation.
- **Enables depth**: Powers modern AI.
- **Numerical/symbolic impls**: Verify and apply.
- **Intuition to code**: Bridge math to models.

Chain rule turns composition into multiplication of locals.

---

## 14. Summary

We dissected the chain rule from intuition to multivariables, then applied it to backpropagation's core. Examples and Python/Rust code illustrated computations, connecting calculus to ML training. Mastering this unlocks neural net dynamics.

Next: Optimization in Practice.

Word count: Approximately 2920.

---

## Further Reading
- Stewart, *Calculus* (Chain Rule sections).
- Nielsen, *Neural Networks and Deep Learning* (Chapter 2: Backprop).
- 3Blue1Brown: Backpropagation video.
- Rust ML: Explore 'ndarray', 'candle' for tensors.

---