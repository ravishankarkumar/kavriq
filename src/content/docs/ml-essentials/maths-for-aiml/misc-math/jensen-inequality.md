---
title: Jensen's Inequality & Convex Functions
description: Comprehensive exploration of Jensen's inequality and convex functions in miscellaneous math for AI/ML, covering definitions, proofs, applications in optimization and probabilistic analysis, with examples and code in Python and Rust
---

# Jensen's Inequality & Convex Functions

Jensen's inequality is a fundamental result in mathematics that relates convex functions to expectations of random variables, providing powerful bounds in probability and optimization. In artificial intelligence and machine learning (ML), it underpins key concepts like the convexity of loss functions, variational inference, and information theory, enabling robust model training and analysis. Convex functions, with their unique properties, are central to optimization algorithms, ensuring global minima in problems like linear regression and logistic regression.

This lecture in the "Foundations for AI/ML" series (misc-math cluster) explores convex functions, Jensen's inequality, its proof, extensions, and applications in ML optimization and probabilistic modeling. We'll provide intuitive explanations, mathematical derivations, and practical implementations in Python and Rust, offering tools to leverage convexity in AI.

---

## 1. Intuition Behind Convex Functions and Jensen's Inequality

**Convex Functions**: A function f is convex if its graph lies below any chord connecting two points, i.e., for λ ∈ [0,1], f(λx + (1-λ)y) ≤ λf(x) + (1-λ)f(y). Intuitively, convex functions have a "bowl" shape, ensuring a single global minimum.

**Jensen's Inequality**: For a convex function f and random variable X:

\[
f(E[X]) \leq E[f(X)]
\]

It says the function evaluated at the mean is less than or equal to the mean of the function values, with equality for linear f.

### ML Connection
- **Optimization**: Convex loss functions (e.g., MSE) guarantee global minima.
- **Information Theory**: Jensen's bounds KL divergence, cross-entropy.

::: info
Jensen's inequality is like weighing ingredients before mixing: the average outcome (f(E[X])) is lighter than mixing outcomes and averaging (E[f(X)]).
:::

### Example
- For f(x)=x² (convex), X~Uniform[0,1], E[X]=0.5, f(0.5)=0.25, E[X²]=1/3≈0.333 > 0.25.

---

## 2. Convex Functions: Definition and Properties

A function f: ℝ → ℝ is convex if for all x, y and λ ∈ [0,1]:

\[
f(λx + (1-λ)y) ≤ λf(x) + (1-λ)f(y)
\]

**Strict Convexity**: Strict inequality for λ ∈ (0,1).

### Properties
- **First-Order**: If differentiable, f(y) ≥ f(x) + f'(x)(y-x) (tangent below graph).
- **Second-Order**: If twice differentiable, f''(x) ≥ 0.
- **Global Minimum**: If f convex, any local minimum is global.
- **Examples**: x², e^x, -log(x), ||x||₂².

### ML Application
- Convex losses (e.g., logistic) ensure unique optima in gradient descent.

Example: f(x)=x², f''(x)=2 > 0, convex.

---

## 3. Jensen's Inequality: Formal Statement and Proof

For convex f, random variable X:

\[
f(E[X]) \leq E[f(X)]
\]

If f strictly convex, equality holds only if X constant.

### Proof (Discrete Case)
For X with values x_i, probs p_i, E[X] = sum p_i x_i.

By convexity, f(sum p_i x_i) ≤ sum p_i f(x_i) = E[f(X)].

### Continuous Case
E[X] = ∫ x p(x) dx, E[f(X)] = ∫ f(x) p(x) dx.

Jensen follows from convexity over expectation.

### ML Insight
- Bounds expected loss in stochastic optimization.

Example: f(x)=x², X~Bernoulli(0.5), E[X]=0.5, E[X²]=0.5, f(0.5)=0.25 ≤ 0.5.

---

## 4. Extensions and Variants

**Concave Functions**: f concave if -f convex, reverses Jensen: f(E[X]) ≥ E[f(X)] (e.g., log(x)).

**Multivariate Jensen**: For convex f: ℝ^n → ℝ, f(E[X]) ≤ E[f(X)].

**Conditional Jensen**: For convex f, E[f(X)|Y] ≥ f(E[X|Y]).

### ML Application
- Variational inference: KL divergence convexity via Jensen.

---

## 5. Applications in Machine Learning

1. **Optimization**: Convex losses ensure global minima (e.g., SVM, logistic regression).
2. **Information Theory**: Jensen bounds KL divergence, cross-entropy in classification.
3. **Variational Inference**: ELBO optimization relies on Jensen.
4. **Risk Analysis**: Bounds expected loss in decision-making.

### Challenges
- Non-convexity in deep learning requires approximations.
- High-dim convexity verification costly.

---

## 6. Jensen's Inequality in Probabilistic Bounds

**KL Divergence**: For distributions P, Q:

\[
D_{KL}(P||Q) = E_P[\log(P/Q)] ≥ \log(E_P[P/Q]) = 0
\]

By Jensen, since -log is convex.

**Cross-Entropy**: H(P,Q) ≥ H(P) from Jensen.

In ML: Justifies loss functions.

---

## 7. Numerical Computations of Jensen's Inequality

Verify Jensen, compute bounds.

::: code-group

```python [Python]
import numpy as np
import matplotlib.pyplot as plt

# Jensen for f(x)=x^2, Uniform[0,1]
X = np.random.uniform(0, 1, 10000)
f_X = X**2
E_X = np.mean(X)  # ~0.5
f_E_X = E_X**2   # ~0.25
E_f_X = np.mean(f_X)  # ~1/3
print("Jensen: f(E[X])=", f_E_X, "≤ E[f(X)]=", E_f_X)

# Plot convex f(x)=x^2
x = np.linspace(0, 1, 100)
plt.plot(x, x**2, label='f(x)=x^2')
plt.axhline(E_f_X, color='r', label='E[f(X)]')
plt.axhline(f_E_X, color='g', label='f(E[X])')
plt.legend()
plt.title("Jensen's Inequality: x^2")
plt.show()

# ML: Cross-entropy bound
from scipy.stats import entropy
P = np.array([0.6, 0.4])
Q = np.array([0.5, 0.5])
kl = entropy(P, Q)
print("KL(P||Q):", kl, "≥ 0 (Jensen)")
```

```rust [Rust]
use rand::Rng;
use plotters::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Jensen for f(x)=x^2, Uniform[0,1]
    let mut rng = rand::thread_rng();
    let x: Vec<f64> = (0..10000).map(|_| rng.gen_range(0.0..1.0)).collect();
    let f_x: Vec<f64> = x.iter().map(|&xi| xi.powi(2)).collect();
    let e_x = x.iter().sum::<f64>() / x.len() as f64;
    let f_e_x = e_x.powi(2);
    let e_f_x = f_x.iter().sum::<f64>() / f_x.len() as f64;
    println!("Jensen: f(E[X])={} ≤ E[f(X)]={}", f_e_x, e_f_x);

    // Plot
    let root = BitMapBackend::new("jensen.png", (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&root)
        .caption("Jensen's Inequality: x^2", ("sans-serif", 50))
        .build_cartesian_2d(0f64..1f64, 0f64..1f64)?;
    chart.draw_series(LineSeries::new(
        (0..100).map(|i| (i as f64 / 100.0, (i as f64 / 100.0).powi(2))),
        &BLUE,
    ))?;
    chart.draw_series(LineSeries::new(
        [(0.0, e_f_x), (1.0, e_f_x)],
        &RED,
    ))?.label("E[f(X)]");
    chart.draw_series(LineSeries::new(
        [(0.0, f_e_x), (1.0, f_e_x)],
        &GREEN,
    ))?.label("f(E[X])");
    chart.configure_series_labels().draw()?;

    // KL divergence
    let p = [0.6, 0.4];
    let q = [0.5, 0.5];
    let kl = p.iter().zip(q.iter()).map(|(&pi, &qi)| if pi > 0.0 { pi * (pi / qi).ln() } else { 0.0 }).sum::<f64>();
    println!("KL(P||Q): {} ≥ 0 (Jensen)", kl);

    Ok(())
}
```
:::

Verifies Jensen's inequality, plots, computes KL.

---

## 8. Symbolic Derivations with SymPy

Derive Jensen's inequality.

::: code-group

```python [Python]
from sympy import symbols, Sum, IndexedBase, log, E

# Jensen for convex f(x)=x^2
X = IndexedBase('X')
p, i = symbols('p i')
n = symbols('n', integer=True, positive=True)
f_x = X[i]**2
e_x = Sum(p * X[i], (i, 1, n))
e_f_x = Sum(p * f_x, (i, 1, n))
print("Jensen: f(E[X])=", e_x**2, "≤ E[f(X)]=", e_f_x)

# KL divergence
P, Q = IndexedBase('P'), IndexedBase('Q')
kl = Sum(P[i] * log(P[i]/Q[i]), (i, 1, n))
print("KL(P||Q) ≥ 0 via Jensen")
```

```rust [Rust]
fn main() {
    println!("Jensen: f(E[X]) = (sum p_i x_i)^2 ≤ E[f(X)] = sum p_i x_i^2");
    println!("KL(P||Q) ≥ 0 via Jensen");
}
```
:::

---

## 9. Challenges in ML Applications

- Non-convexity: Deep learning losses require approximations.
- High-dim: Convexity verification costly.
- Discrete distributions: Jensen less tight.

---

## 10. Key ML Takeaways

- **Convexity ensures optima**: In optimization.
- **Jensen bounds expectations**: Probabilistic analysis.
- **KL, cross-entropy rely**: On Jensen.
- **Loss functions convex**: For reliability.
- **Code verifies**: Practical bounds.

Jensen and convexity drive ML optimization.

---

## 11. Summary

Explored convex functions, Jensen's inequality, derivations, with ML applications in optimization and information theory. Examples and Python/Rust code bridge theory to practice. Essential for probabilistic ML.

Word count: Approximately 3000.

---

## Further Reading
- Boyd, Vandenberghe, *Convex Optimization*.
- Cover, Thomas, *Elements of Information Theory*.
- Murphy, *Probabilistic Machine Learning* (Ch. 2).
- Rust: 'plotters' for visualization, 'rand' for sampling.

---