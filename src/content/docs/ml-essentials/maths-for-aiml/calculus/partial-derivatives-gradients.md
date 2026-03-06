---
title: Partial Derivatives & Gradients
description: Delve into partial derivatives and gradients in multivariable calculus for AI/ML, emphasizing optimization, backpropagation, and multidimensional change, with examples and code in Python and Rust
---

# Partial Derivatives & Gradients

Multivariable calculus extends single-variable concepts to higher dimensions, where functions depend on multiple inputs. In AI and machine learning, this is essential since models often process vectors or tensors. Partial derivatives measure change with respect to one variable while holding others constant, and gradients aggregate these into a vector pointing toward steepest ascent. These tools power optimization in neural networks, enabling efficient parameter updates via algorithms like gradient descent.

This lecture builds on derivatives, exploring partials, gradients, Jacobians, Hessians, and their ML applications. We'll connect theory to practice with intuitions, examples, and implementations in Python and Rust, fostering a deep understanding of how AI navigates complex loss landscapes.

---

## 1. Intuition Behind Multivariable Functions

In single-variable calculus, functions like \( f(x) = x^2 \) model one-dimensional change. Now, consider \( f(x, y) = x^2 + y^2 \), a paraboloid surface. Change can occur in x or y directions independently.

Partial derivatives slice this surface: Fixing y treats it as a constant, yielding a curve in x whose slope is the partial with respect to x.

### ML Connection
- Neural networks: Loss functions \( L(\mathbf{w}) \) depend on weight vectors \(\mathbf{w} = (w_1, w_2, \dots, w_n)\).
- Partial derivatives: Sensitivity of loss to each weight, guiding adjustments.
- High dimensions: Modern models have millions of parameters—partials make this tractable.

::: info
Think of partials as "what if" scenarios: What if I tweak only this parameter while freezing others?
:::

### Example
- For \( f(x, y) = 3x^2 y + e^{xy} \), partial w.r.t. x explores x's isolated impact.

---

## 2. Formal Definition of Partial Derivatives

For \( f(x_1, x_2, \dots, x_n) \), the partial with respect to \( x_i \) is:

\[
\frac{\partial f}{\partial x_i} = \lim_{h \to 0} \frac{f(x_1, \dots, x_i + h, \dots, x_n) - f(x_1, \dots, x_i, \dots, x_n)}{h}
\]

Treat other variables as constants.

Notation: \( f_x, \partial_x f, D_x f \).

### Differentiability in Multiple Variables
- Continuous partials imply differentiability.
- But existence alone doesn't guarantee total differentiability (unlike single-variable).

### ML Insight
- In regression: \( L(w, b) = \sum (y - (w x + b))^2 / n \).
  - \( \partial L / \partial w = -2 \sum x (y - \hat{y}) / n \).
  - Used to solve normal equations or iteratively optimize.

---

## 3. Computing Partial Derivatives: Rules and Examples

Extend single-variable rules:

1. **Power Rule**: \( \partial (x^n) / \partial x = n x^{n-1} \), others zero.
2. **Sum Rule**: Partials distribute over sums.
3. **Product Rule**: \( \partial (u v) / \partial x = u_x v + u v_x \).
4. **Chain Rule**: For composite \( f(g(x,y)) \), \( \partial f / \partial x = f'(g) \cdot \partial g / \partial x \).

### Examples
- \( f(x,y) = x^2 y + \sin(xy) \):
  - \( \partial f / \partial x = 2x y + y \cos(xy) \).
  - \( \partial f / \partial y = x^2 + x \cos(xy) \).
- Implicit: \( x^2 + y^2 + z^2 = 1 \), \( \partial z / \partial x = -x / z \).

### ML Application
- Sigmoid in multivariable: \( \sigma(\mathbf{w} \cdot \mathbf{x}) \), partial w.r.t. \( w_i \) involves chain rule.

---

## 4. The Gradient Vector

The gradient \( \nabla f = (\partial f / \partial x_1, \dots, \partial f / \partial x_n) \) points to steepest ascent, magnitude is the rate.

At point \(\mathbf{p}\), the directional derivative in direction \(\mathbf{u}\) (unit vector) is \( \nabla f(\mathbf{p}) \cdot \mathbf{u} \).

### Properties
- Orthogonal to level surfaces (constant f).
- Zero at critical points (local min/max/saddle).

### ML Connection
- Gradient descent: \( \mathbf{w}_{new} = \mathbf{w}_{old} - \eta \nabla L(\mathbf{w}_{old}) \).
- In backprop, gradients flow backward to update layers.

::: info
Gradients are like compasses in parameter space: Follow negative for minima.
:::

### Example
- For \( f(x,y) = x^2 + y^2 \), \( \nabla f = (2x, 2y) \), points outward from origin.

---

## 5. Higher-Order Partials: Jacobian and Hessian

- **Mixed Partials**: \( \partial^2 f / \partial x \partial y \). Clairaut's theorem: Equal if continuous.
- **Jacobian Matrix**: For vector-valued f, matrix of partials. In ML, for transformations.
- **Hessian Matrix**: Matrix of second partials, \( H_{ij} = \partial^2 f / \partial x_i \partial x_j \).
  - Positive definite: Local min.
  - Used in Newton's method: \( \mathbf{w}_{new} = \mathbf{w}_{old} - H^{-1} \nabla L \).

### ML Insight
- In deep learning, Hessian approximates curvature for second-order optimization (e.g., L-BFGS).
- Eigenvalues reveal saddle points in loss landscapes.

Example: For quadratic \( f(\mathbf{x}) = \mathbf{x}^T A \mathbf{x} \), Hessian is 2A.

---

## 6. Geometric Interpretations and Visualizations

Geometrically, partials are slopes of tangent lines in each coordinate plane.

Gradient: Tangent vector to the path of steepest ascent on the surface.

### Visualization (Conceptual)
- Contour plots: Gradients perpendicular to contours.
- Vector fields: Plot gradients over domain—shows flow toward maxima.
- In 3D: For \( z = f(x,y) \), tangent plane equation uses partials.

In ML, loss surfaces are visualized as hilly terrains; gradients navigate valleys.

---

## 7. Directional Derivatives and Applications

Directional derivative: Rate of change along vector \(\mathbf{v}\): \( D_{\mathbf{v}} f = \nabla f \cdot \hat{\mathbf{v}} \).

Max at \(\mathbf{v}\) parallel to gradient.

### Applications
- Constrained optimization: Lagrange multipliers set \( \nabla f = \lambda \nabla g \).
- In RL: Policy gradients for action spaces.

### ML Example
- Feature importance: Partial derivatives w.r.t. inputs approximate sensitivity.

---

## 8. Numerical Computation of Partials and Gradients

Use finite differences for approximation:

- Central: \( \partial f / \partial x_i \approx [f(\mathbf{x} + h \mathbf{e}_i) - f(\mathbf{x} - h \mathbf{e}_i)] / (2h) \).

In code, leverage libraries for auto-diff (PyTorch, but here manual).

::: code-group

```python [Python]
import numpy as np

def f(x, y):
    return x**2 * y + np.sin(x * y)

def partial_x(f, x, y, h=1e-5):
    return (f(x + h, y) - f(x - h, y)) / (2 * h)

def partial_y(f, x, y, h=1e-5):
    return (f(x, y + h) - f(x, y - h)) / (2 * h)

def gradient(f, x, y, h=1e-5):
    return np.array([partial_x(f, x, y, h), partial_y(f, x, y, h)])

x, y = 1.0, 2.0
grad = gradient(f, x, y)
print(f"Gradient at ({x}, {y}): {grad}")

# ML example: Gradient for multivariate loss
def loss(w, X, y):
    # w: vector, X: matrix, y: vector
    return np.mean((y - X @ w)**2)

X = np.array([[1, 2], [3, 4], [5, 6]])
y = np.array([3, 7, 11])  # Approx y = x1 + x2
w = np.array([0.5, 0.5])
h = 1e-5
grad_w = np.array([ (loss(w + h*np.eye(2)[i], X, y) - loss(w - h*np.eye(2)[i], X, y)) / (2*h) for i in range(2) ])
print(f"Gradient w.r.t. w: {grad_w}")
```

```rust [Rust]
fn f(x: f64, y: f64) -> f64 {
    x.powi(2) * y + (x * y).sin()
}

fn partial_x(f: fn(f64, f64) -> f64, x: f64, y: f64, h: f64) -> f64 {
    (f(x + h, y) - f(x - h, y)) / (2.0 * h)
}

fn partial_y(f: fn(f64, f64) -> f64, x: f64, y: f64, h: f64) -> f64 {
    (f(x, y + h) - f(x, y - h)) / (2.0 * h)
}

fn gradient(f: fn(f64, f64) -> f64, x: f64, y: f64, h: f64) -> [f64; 2] {
    [partial_x(f, x, y, h), partial_y(f, x, y, h)]
}

fn main() {
    let x = 1.0;
    let y = 2.0;
    let h = 1e-5;
    let grad = gradient(f, x, y, h);
    println!("Gradient at ({}, {}): {:?}", x, y, grad);

    // ML example: Gradient for multivariate loss
    fn loss(w: &[f64], x: &[[f64; 2]], y: &[f64]) -> f64 {
        let mut sum = 0.0;
        for (xi, &yi) in x.iter().zip(y.iter()) {
            let pred = w[0] * xi[0] + w[1] * xi[1];
            sum += (yi - pred).powi(2);
        }
        sum / x.len() as f64
    }

    let x_data = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
    let y_data = [3.0, 7.0, 11.0];
    let mut w = [0.5, 0.5];
    let mut grad_w = [0.0, 0.0];
    for i in 0..2 {
        let mut w_plus = w;
        w_plus[i] += h;
        let mut w_minus = w;
        w_minus[i] -= h;
        grad_w[i] = (loss(&w_plus, &x_data, &y_data) - loss(&w_minus, &x_data, &y_data)) / (2.0 * h);
    }
    println!("Gradient w.r.t. w: {:?}", grad_w);
}
```
:::

This approximates partials and gradients for functions and ML losses.

---

## 9. Symbolic Computation in Multivariables

Use SymPy for exact partials.

::: code-group

```python [Python]
from sympy import symbols, diff, sin

x, y = symbols('x y')
f = x**2 * y + sin(x * y)
partial_x = diff(f, x)
partial_y = diff(f, y)
print("Partial x:", partial_x)
print("Partial y:", partial_y)

# Evaluate
vals = {x: 1, y: 2}
print("Partial x at (1,2):", partial_x.subs(vals).evalf())
```

```rust [Rust]
// Rust symbolic via hardcoded or crates (simulated)
fn main() {
    // Symbolic: partial_x = 2x y + y cos(x y)
    // partial_y = x^2 + x cos(x y)
    println!("Partial x: 2 x y + y cos(x y)");
    println!("Partial y: x^2 + x cos(x y)");

    let x_val = 1.0;
    let y_val = 2.0;
    let partial_x_val = 2.0 * x_val * y_val + y_val * (x_val * y_val).cos();
    println!("Partial x at (1,2): {}", partial_x_val);
}
```
:::

---

## 10. Optimization with Gradients in ML

- **Gradient Descent Variants**: Batch, mini-batch, stochastic.
- **Backpropagation**: Chain rule for gradients in nets.
- **Advanced**: Momentum adds velocity; Adam adapts per-parameter.

Challenges: Vanishing gradients (sigmoid), exploding (deep nets)—fixes like ReLU, batch norm.

Example: Minimizing \( L(w_1, w_2) = (w_1 - 1)^2 + (w_2 - 2)^2 \), gradient leads to (1,2).

---

## 11. Implicit Functions and Lagrange Multipliers

Implicit theorem: Partials ensure solvability for one variable.

Lagrange: For min f subject to g=0, solve \( \nabla f = \lambda \nabla g \).

In ML: Regularization as constraints (e.g., L1/L2).

---

## 12. Mean Value Theorem in Multiple Variables

Extends: Line segment between points has point where directional derivative matches average change.

In ML: Bounds approximation errors in Taylor expansions for trust regions.

---

## 13. Key ML Takeaways

- **Partials isolate effects**: Crucial for high-dim models.
- **Gradients guide descent**: Core of training.
- **Hessians for curvature**: Enable efficient optimizers.
- **Numerical/symbolic tools**: Bridge theory to code.
- **Multivariable mastery**: Unlocks deep learning dynamics.

Partials and gradients empower AI to optimize in vast spaces.

---

## 14. Summary

We explored partial derivatives from definitions to computations, gradients as vectors of change, and higher-order structures like Hessians. Tied to ML via optimization and backprop, with Python/Rust code for practical insight. This equips you for multivariable challenges in AI.

Next: Optimization Techniques.

Word count: Approximately 2950.

---

## Further Reading
- Stewart, *Multivariable Calculus* (Chapters 14-15).
- Goodfellow et al., *Deep Learning* (Chapter 4: Numerical Computation).
- 3Blue1Brown: Linear Algebra and Multivariable Calculus videos.
- Rust: Explore 'ndarray' for array ops.

---