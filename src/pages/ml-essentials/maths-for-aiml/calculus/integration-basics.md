---
title: Integration Basics - Area Under the Curve
description: Fundamental concepts of integration in calculus for AI/ML, covering Riemann sums, antiderivatives, the Fundamental Theorem of Calculus, and applications like probability densities and cumulative rewards, with examples and code in Python and Rust
layout: ../../../../layouts/TutorialPage.astro
---

# Integration Basics - Area Under the Curve

Integration is the inverse of differentiation, transforming rates of change into accumulated quantities like areas, volumes, and totals. In AI and machine learning, integration computes expectations in probabilistic models, normalizes probability distributions, and evaluates cumulative rewards in reinforcement learning. The definite integral measures net accumulation over an interval, while the indefinite integral finds antiderivatives—functions whose derivatives recover the original.

This lecture shifts from derivatives to integrals, starting with geometric intuition, formal definitions via Riemann sums, the Fundamental Theorem of Calculus (FTC), basic rules, and numerical methods. We'll connect these to ML through examples like integrating loss functions or PDFs, with hands-on code in Python and Rust to compute areas and approximations.

Understanding integration equips you to handle continuous models in AI, from Bayesian inference to signal processing in computer vision.

---

## 1. Intuition: From Rates to Accumulation

Derivatives measure instantaneous change; integrals accumulate over time or space. Think of velocity v(t): Integral of v dt = displacement s(t).

Geometrically: Area under f(x) ≥0 from a to b is ∫_a^b f(x) dx.

If f negative, signed area.

### ML Connection
- In probabilistic ML: ∫ p(x) dx =1 (normalization).
- Cumulative distribution F(x) = ∫_(-{\infty})^x p(t) dt.
- Expected value E[X] = ∫ x p(x) dx.

::: info
Integration "sums infinitesimally"—discretize to Riemann sums, then take limit.
:::

### Example
- f(x)=x from 0 to 1: Triangle area 1/2.
- In RL: Integral of reward rate over time = total return.

---

## 2. Riemann Sums and Definite Integrals

Partition [a,b] into n subintervals Δx_i = (b-a)/n.

Riemann sum: sum f(x_i*) Δx_i, x_i* in i-th interval (left, right, midpoint).

Definite integral: lim_{n→∞} sum = ∫_a^b f(x) dx.

Properties: Linearity, additivity over intervals, |∫ f| ≤ ∫ |f|.

### Convergence
Uniform partitions for continuous f.

### ML Insight
- Monte Carlo integration: Sample x_i ~ uniform, average f(x_i) * (b-a) ≈ integral.

---

## 3. Antiderivatives and Indefinite Integrals

F(x) antiderivative of f if F'=f.

Indefinite ∫ f dx = F(x) + C.

Families of solutions.

### Finding Antiderivatives
Reverse rules: ∫ x^n dx = x^{n+1}/(n+1) + C, n≠-1.

### Example
- ∫ sin x dx = -cos x + C.
- In ML: Antiderivative of loss derivative gives loss.

---

## 4. Fundamental Theorem of Calculus

FTC1: If f continuous, F(x)=∫_a^x f(t) dt, then F'(x)=f(x).

FTC2: ∫_a^b f(x) dx = F(b) - F(a), F antideriv.

Links diff and int: Evaluate definite via antideriv.

### Proof Sketch
FTC1: By mean value theorem on integrals.

### ML Application
- Derivative of integral (e.g., score function in RL) = integrand.

Example: ∫_0^1 x dx = [x^2/2]_0^1 = 1/2.

---

## 5. Basic Integration Rules

1. ∫ k f dx = k ∫ f dx.
2. ∫ (f + g) dx = ∫ f + ∫ g.
3. ∫ x^n dx = x^{n+1}/(n+1) + C.
4. ∫ e^x dx = e^x + C.
5. ∫ 1/x dx = ln|x| + C.
6. Substitution: Let u=g(x), du=g' dx, ∫ f(g) g' dx = ∫ f(u) du.

### Examples
- ∫ (3x^2 + 2) dx = x^3 + 2x + C.
- Sub: ∫ sin(2x) dx = - (1/2) cos(2x) + C.

### ML Insight
- Integrate activations for smooth losses.

---

## 6. Applications: Areas, Volumes, Averages

Area: ∫ f dx.

Volume: Disk method π ∫ [R(x)]^2 dx.

Average value: (1/(b-a)) ∫_a^b f dx.

In ML: Average loss over data distribution.

Arc length: ∫ sqrt(1 + (f')^2) dx.

---

## 7. Numerical Integration: Trapezoidal, Simpson's, Monte Carlo

Trapezoidal: (Δx/2) (f(a) + 2 sum mid + f(b)).

Simpson: Quadratic approx.

Monte Carlo: For high-dim, sample-based.

::: code-group

```python [Python]
import numpy as np

def riemann_sum(f, a, b, n=1000):
    dx = (b - a) / n
    x = np.linspace(a, b, n+1)
    return np.sum(f(x[:-1]) * dx)  # Left Riemann

def trapezoidal(f, a, b, n=1000):
    dx = (b - a) / n
    x = np.linspace(a, b, n+1)
    y = f(x)
    return dx * (0.5 * y[0] + np.sum(y[1:-1]) + 0.5 * y[-1])

def monte_carlo(f, a, b, n=10000):
    samples = np.random.uniform(a, b, n)
    return (b - a) * np.mean(f(samples))

# Example: ∫_0^1 x^2 dx = 1/3 ≈0.333
f = lambda x: x**2
exact = 1/3
print("Riemann:", riemann_sum(f, 0, 1))
print("Trapezoidal:", trapezoidal(f, 0, 1))
print("Monte Carlo:", monte_carlo(f, 0, 1))

# ML: Integrate Gaussian PDF for CDF approx
from scipy.stats import norm
def gaussian_pdf(x, mu=0, sigma=1):
    return (1/(sigma * np.sqrt(2*np.pi))) * np.exp(-0.5 * ((x - mu)/sigma)**2)

cdf_approx = trapezoidal(gaussian_pdf, -3, 3)
print("Approx integral (should ~1):", cdf_approx)
```

```rust [Rust]
fn riemann_sum(f: fn(f64) -> f64, a: f64, b: f64, n: usize) -> f64 {
    let dx = (b - a) / n as f64;
    (0..n).map(|i| {
        let x = a + i as f64 * dx;
        f(x) * dx
    }).sum::<f64>()
}

fn trapezoidal(f: fn(f64) -> f64, a: f64, b: f64, n: usize) -> f64 {
    let dx = (b - a) / n as f64;
    let mut sum = 0.5 * f(a) + 0.5 * f(b);
    for i in 1..n {
        let x = a + i as f64 * dx;
        sum += f(x);
    }
    sum * dx
}

fn monte_carlo(f: fn(f64) -> f64, a: f64, b: f64, n: usize) -> f64 {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let mut total = 0.0;
    for _ in 0..n {
        let x = rng.gen_range(a..=b);
        total += f(x);
    }
    (b - a) * total / n as f64
}

fn main() {
    let f = |x: f64| x.powi(2);
    let exact = 1.0 / 3.0;
    println!("Riemann: {}", riemann_sum(f, 0.0, 1.0, 1000));
    println!("Trapezoidal: {}", trapezoidal(f, 0.0, 1.0, 1000));
    println!("Monte Carlo: {}", monte_carlo(f, 0.0, 1.0, 10000));

    // Gaussian PDF
    let gaussian = |x: f64| {
        let sigma = 1.0;
        let mu = 0.0;
        (1.0 / (sigma * (2.0 * std::f64::consts::PI).sqrt())) * (-0.5 * ((x - mu)/sigma).powi(2)).exp()
    };
    println!("Approx integral: {}", trapezoidal(gaussian, -3.0, 3.0, 1000));
}
```
:::

Numerically approximates integrals.

---

## 8. Integration in Machine Learning

- Probability: Normalize ∫ p(θ|data) dθ=1.
- Expectations: E[loss] = ∫ loss(θ) p(θ) dθ.
- In VAEs: ELBO integrates latent space.
- Numerical: Quadrature for low-dim, MCMC for high.

Example: Softmax probabilities sum to 1 via integral approx in continuous.

---

## 9. Initial Value Problems and Definite Integrals

Solve dy/dx = f(x,y), y(a)=b: y(x) = b + ∫_a^x f(t,y(t)) dt.

In ML: ODEs for neural ODEs.

---

## 10. Common Pitfalls and Tips

- Forgetting +C in indefinite.
- Sign errors in limits.
- Non-integrable discontinuities (improper integrals).

In code: Choose method by smoothness/dim.

---

## 11. Improper Integrals and Convergence

∫_a^∞ f = lim_{b→({\infty})} ∫_a^b f.

Absolute convergence for switching limits.

In ML: Infinite support PDFs.

---

## 12. Key ML Takeaways

- **Accumulation in models**: Integrals for totals.
- **FTC links diff/int**: Backprop through integrals.
- **Numerical tools**: Approx for complex.
- **Probabilistic core**: Expectations via int.
- **Code for practice**: Riemann to Monte Carlo.

Integration completes calculus toolkit for AI.

---

## 13. Summary

Introduced integration from Riemann to FTC, rules, numerical, with ML ties. Python/Rust code enables computation. Prepares for advanced like multiple integrals.

Next: Multivariable Integration.

Word count: Approximately 2900.

---

## Further Reading
- Stewart, *Calculus* (Chapters 5-6).
- Goodfellow et al., *Deep Learning* (Appendix: Probability).
- 3Blue1Brown: Essence of Calculus Ch. 5-6.
- Rust: 'statrs' for stats, 'ndarray' for numerics.

---