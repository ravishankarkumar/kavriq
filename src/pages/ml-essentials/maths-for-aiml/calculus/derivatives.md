---
title: Derivatives - Measuring Change
description: Explore derivatives in calculus for AI/ML, focusing on rates of change, gradients, and optimization, with examples and code in Python and Rust
layout: ../../../../layouts/TutorialPage.astro
---

# Derivatives: Measuring Change

Calculus empowers us to quantify change, and derivatives are the cornerstone for measuring instantaneous rates of change. In artificial intelligence and machine learning, derivatives manifest as gradients that drive optimization algorithms like gradient descent. They tell us how to adjust model parameters to minimize loss functions, enabling models to learn from data. This lecture delves into the intuition behind derivatives, their formal definitions, computational rules, and practical applications in ML, all while bridging mathematical rigor with code implementations in Python and Rust.

Whether you're tuning neural networks or analyzing algorithmic efficiency, mastering derivatives unlocks deeper insights into how AI systems evolve and improve.

## 1. What is a Derivative? Intuition First

Imagine driving a car: your speedometer shows the instantaneous speed, not the average over the trip. A derivative captures this "instantaneous rate of change" for any function. For a position function \( s(t) \), the derivative \( s'(t) \) gives velocity at time \( t \).

Formally, the derivative of \( f(x) \) at \( x = a \) is the slope of the tangent line to the graph at that point. It's the limit of the average rate of change over shrinking intervals.

### ML Connection
- In **gradient descent**, the derivative of the loss function with respect to weights tells us the direction to update parameters: \( w_{new} = w_{old} - \eta \frac{\partial L}{\partial w} \).
- **Backpropagation** computes derivatives layer by layer in neural networks.
- Derivatives help diagnose issues like vanishing or exploding gradients in deep learning.

::: info
Derivatives aren't just slopes—they're the DNA of optimization, revealing how small tweaks in inputs ripple through outputs.
:::

### Everyday Example
- For \( f(x) = x^2 \), the derivative at \( x = 2 \) is 4, meaning the function increases by about 4 units for every small unit increase in \( x \) near 2.
- In economics (analogous to ML cost functions), marginal cost is the derivative of total cost.

## 2. Formal Definition Using Limits

The derivative is defined via limits:

\[
f'(x) = \lim_{h \to 0} \frac{f(x + h) - f(x)}{h}
\]

This is the **difference quotient** approaching the instantaneous rate as \( h \) shrinks.

If the limit exists, \( f \) is **differentiable** at \( x \). Differentiability implies continuity, but not vice versa (e.g., \( |x| \) at 0 is continuous but not differentiable).

### Conditions for Differentiability
- Left and right limits must agree.
- No sharp corners, cusps, or vertical tangents.

### ML Insight
- Loss functions like mean squared error (MSE) are differentiable, allowing smooth optimization.
- Non-differentiable activations like ReLU use subgradients in practice, but derivatives guide most training.

## 3. Geometric and Physical Interpretations

Geometrically: The derivative is the tangent slope. For \( f(x) = \sin(x) \), at \( x = 0 \), \( f'(0) = 1 \), a gentle upward slope.

Physically: Acceleration is the derivative of velocity (second derivative of position). In robotics or reinforcement learning, this models dynamic systems.

### Visualization (Conceptual)
- Plot \( f(x) = x^3 - x \): Tangents at local max/min have zero slope (critical points).
- In ML, the gradient vector field over a loss surface shows paths of steepest descent—valleys lead to minima.

These graphs illustrate how derivatives pinpoint where functions flatten or accelerate.

## 4. Basic Differentiation Rules

No need to compute limits every time—rules simplify:

1. **Constant Rule**: \( \frac{d}{dx} c = 0 \).
2. **Power Rule**: \( \frac{d}{dx} x^n = n x^{n-1} \).
3. **Sum Rule**: \( \frac{d}{dx} [f(x) + g(x)] = f'(x) + g'(x) \).
4. **Product Rule**: \( \frac{d}{dx} [f(x) g(x)] = f'(x)g(x) + f(x)g'(x) \).
5. **Quotient Rule**: \( \frac{d}{dx} \left[ \frac{f(x)}{g(x)} \right] = \frac{f'(x)g(x) - f(x)g'(x)}{[g(x)]^2} \).
6. **Chain Rule**: \( \frac{d}{dx} f(g(x)) = f'(g(x)) \cdot g'(x) \) — crucial for composite functions like neural nets.

### Examples
- \( f(x) = 3x^4 \): \( f'(x) = 12x^3 \).
- \( f(x) = e^x \sin(x) \): Use product rule → \( f'(x) = e^x \sin(x) + e^x \cos(x) \).
- Chain rule in action: For \( f(x) = (x^2 + 1)^3 \), let \( u = x^2 + 1 \), \( f = u^3 \), so \( f'(x) = 3(x^2 + 1)^2 \cdot 2x \).

### ML Application
- Chain rule enables backpropagation: Derivatives propagate backward through layers.
- In logistic regression, derivative of sigmoid: \( \sigma'(x) = \sigma(x)(1 - \sigma(x)) \).

## 5. Higher-Order Derivatives

Derivatives of derivatives reveal concavity and acceleration.

- Second derivative: \( f''(x) = \frac{d}{dx} f'(x) \), measures curvature.
  - \( f''(x) > 0 \): Concave up (local min possible).
  - \( f''(x) < 0 \): Concave down (local max).
- Third and higher: Used in Taylor expansions for approximations.

### ML Connection
- Hessian matrix (second partials) in Newton's method for faster convergence.
- In optimization, positive definite Hessian ensures a minimum.

Example: For \( f(x) = x^4 \), \( f'(x) = 4x^3 \), \( f''(x) = 12x^2 \), always non-negative—global minimum at 0.

## 6. Partial Derivatives in Multivariable Calculus

ML deals with high-dimensional spaces, so extend to functions of multiple variables.

For \( f(x, y) \), partial with respect to \( x \): Treat \( y \) as constant.

\[
\frac{\partial f}{\partial x} = \lim_{h \to 0} \frac{f(x + h, y) - f(x, y)}{h}
\]

Gradient: \( \nabla f = \left( \frac{\partial f}{\partial x}, \frac{\partial f}{\partial y} \right) \), direction of steepest ascent.

### ML Example
- In linear regression: Loss \( L(w, b) = \frac{1}{n} \sum (y_i - (w x_i + b))^2 \).
  - \( \frac{\partial L}{\partial w} = -\frac{2}{n} \sum x_i (y_i - \hat{y_i}) \).
  - Used in batch gradient descent.

## 7. Numerical Approximation of Derivatives

Analytically impossible sometimes? Use finite differences:

- Forward: \( f'(x) \approx \frac{f(x + h) - f(x)}{h} \).
- Central: \( f'(x) \approx \frac{f(x + h) - f(x - h)}{2h} \) — more accurate.

Choose small \( h \), but watch for floating-point errors.

::: code-group

```python [Python]
import numpy as np

def f(x):
    return x**2 + np.sin(x)

def numerical_derivative(f, x, h=1e-5):
    return (f(x + h) - f(x - h)) / (2 * h)

x = 1.0
analytical = 2*x + np.cos(x)  # Exact derivative
numerical = numerical_derivative(f, x)

print(f"Analytical derivative at x={x}: {analytical}")
print(f"Numerical derivative at x={x}: {numerical}")

# ML example: Gradient for simple loss
def loss(w, x_data, y_data):
    return np.mean((y_data - w * x_data)**2)

x_data = np.array([1, 2, 3])
y_data = np.array([2, 4, 6])  # Linear: y=2x
w = 1.5
grad_w = numerical_derivative(lambda w: loss(w, x_data, y_data), w)
print(f"Gradient w.r.t. w at {w}: {grad_w}")
```

```rust [Rust]
use std::f64;

fn f(x: f64) -> f64 {
    x.powi(2) + x.sin()
}

fn numerical_derivative(f: fn(f64) -> f64, x: f64, h: f64) -> f64 {
    (f(x + h) - f(x - h)) / (2.0 * h)
}

fn main() {
    let x = 1.0;
    let h = 1e-5;
    let analytical = 2.0 * x + x.cos();
    let numerical = numerical_derivative(f, x, h);

    println!("Analytical derivative at x={}: {}", x, analytical);
    println!("Numerical derivative at x={}: {}", x, numerical);

    // ML example: Gradient for simple loss
    fn loss(w: f64, x_data: &[f64], y_data: &[f64]) -> f64 {
        let mut sum = 0.0;
        for (&xi, &yi) in x_data.iter().zip(y_data.iter()) {
            sum += (yi - w * xi).powi(2);
        }
        sum / x_data.len() as f64
    }

    let x_data = [1.0, 2.0, 3.0];
    let y_data = [2.0, 4.0, 6.0];  // Linear: y=2x
    let w = 1.5;
    let grad_w = numerical_derivative(|w| loss(w, &x_data, &y_data), w, h);
    println!("Gradient w.r.t. w at {}: {}", w, grad_w);
}
```

:::

This code demonstrates approximating derivatives and applying them to a toy ML loss function.

## 8. Symbolic Differentiation in Code

For exact derivatives, use symbolic libraries. In Python, SymPy; in Rust, libraries like symbolic-rs (simulated here).

::: code-group

```python [Python]
from sympy import symbols, diff, sin

x = symbols('x')
f = x**2 + sin(x)
f_prime = diff(f, x)
print("Symbolic derivative:", f_prime)

# Evaluate at x=1
value = f_prime.subs(x, 1)
print("At x=1:", value.evalf())
```

```rust [Rust]
fn symbolic_derivative() {
    // Note: Rust lacks built-in symbolic differentiation; simulate or use crates.
    // Hardcoding derivative: d/dx (x^2 + sin(x)) = 2x + cos(x)
    println!("Symbolic derivative: 2x + cos(x)");
    
    let x = 1.0;
    let value = 2.0 * x + x.cos();
    println!("At x=1: {}", value);
}

fn main() {
    symbolic_derivative();
}
```

:::

In practice, for Rust, consider crates like `nalgebra` or implement manually for simple cases.

## 9. Derivatives in Optimization Algorithms

Gradient descent variants rely on derivatives:

- **Vanilla GD**: Step proportional to negative gradient.
- **Stochastic GD (SGD)**: Noisy but efficient for large data.
- **Adam**: Adaptive learning rates using first and second moments.

### Challenges
- **Local minima**: Derivatives zero, but not global optimum.
- **Saddle points**: Hessian has mixed signs.
- ML fix: Momentum, learning rate scheduling.

Example: Optimizing \( L(\theta) = \theta^2 \) converges to 0 via \( \theta \leftarrow \theta - \eta \cdot 2\theta \).

## 10. Implicit Differentiation and Related Rates

For equations like \( x^2 + y^2 = 1 \) (circle), differentiate implicitly: \( 2x + 2y \frac{dy}{dx} = 0 \Rightarrow \frac{dy}{dx} = -\frac{x}{y} \).

In ML: Used in constrained optimization or auto-differentiation frameworks like PyTorch.

Related rates: If volume \( V = \frac{4}{3}\pi r^3 \), \( \frac{dV}{dt} = 4\pi r^2 \frac{dr}{dt} \)—analogous to chain rule in dynamic models.

## 11. Mean Value Theorem and Applications

**MVT**: If continuous on [a,b] and differentiable on (a,b), there exists c where \( f'(c) = \frac{f(b) - f(a)}{b - a} \).

Implications: Connects average and instantaneous rates. In ML, bounds error in approximations.

**Rolle's Theorem**: Special case where f(a)=f(b), so f'(c)=0—guarantees critical points.

## 12. Key ML Takeaways

- **Gradients drive learning**: Derivatives quantify parameter sensitivity.
- **Chain rule = backprop**: Enables deep networks.
- **Higher derivatives**: Inform curvature-aware optimizers.
- **Numerical vs. Analytical**: Trade accuracy for computability in code.
- **Multivariable**: Handles real-world high-dim data.

Derivatives transform static functions into dynamic tools for change detection and prediction.

## 13. Summary

This lecture unpacked derivatives from intuitive slopes to rigorous limits, rules, and multivariable extensions. We connected them to ML optimization, illustrated with examples, and implemented numerical/symbolic computations in Python and Rust. Understanding derivatives equips you to dissect how AI models adapt and improve.

Next up: Integrals—reversing derivatives to accumulate change.

## Further Reading
- Khan Academy: Derivatives section.
- Bishop, *Pattern Recognition and Machine Learning* (Chapter 5: Neural Networks).
- 3Blue1Brown: Essence of Calculus series (visual intuitions).
- Rust crates: For advanced symbolic math, explore `nalgebra` or `symbolic`.

---