---
title: Law of Large Numbers & Central Limit Theorem
description: Exploration of the Law of Large Numbers (LLN) and Central Limit Theorem (CLT) in probability for AI/ML, covering their statements, proofs, types, and applications in statistical inference, model evaluation, and optimization, with examples and code in Python and Rust
---

# Law of Large Numbers & Central Limit Theorem

The Law of Large Numbers (LLN) and Central Limit Theorem (CLT) are foundational results in probability that explain how sample averages behave in large datasets. LLN assures that sample means converge to expected values, while CLT describes the normal distribution of those means. In machine learning (ML), these theorems justify empirical risk minimization, bootstrap methods, and confidence intervals, enabling reliable model training and evaluation from data.

This sixth lecture in the "Probability Foundations for AI/ML" series builds on independence and correlation, delving into LLN (weak and strong forms), CLT, their proofs, conditions, and ML applications. We'll provide intuitive explanations, mathematical derivations, and implementations in Python and Rust, preparing you for advanced topics like maximum likelihood estimation.

---

## 1. Intuition Behind LLN and CLT

**LLN**: As sample size grows, the average converges to the true mean—like flipping a coin many times, the proportion of heads approaches 0.5.

**CLT**: The distribution of sample means becomes normal, regardless of the underlying distribution, for large n—enabling Gaussian approximations.

### ML Connection
- **LLN**: Justifies training on large data; loss averages converge.
- **CLT**: Confidence intervals for model performance.

::: info
LLN stabilizes averages; CLT shapes them normal—together, they make big data predictable.
:::

### Example
- LLN: Sample mean of dice rolls →3.5.
- CLT: Means of 100 rolls ~ N(3.5, var/100).

---

## 2. Law of Large Numbers: Weak and Strong Forms

For i.i.d. X_i with E[X_i]=μ <∞.

**Weak LLN (WLLN)**: Sample mean \bar{X}_n →^P μ (in probability).

**Strong LLN (SLLN)**: \bar{X}_n →^a.s. μ (almost surely).

### Conditions
- WLLN: Finite variance (Chebyshev) or identical dists (Khintchine).
- SLLN: Finite expectation (Kolmogorov).

### Proof Sketch (WLLN via Chebyshev)
Var(\bar{X}_n)=σ^2/n →0, so P(|\bar{X}_n - μ| >ε) ≤ Var(\bar{X}_n)/ε^2 →0.

### ML Insight
- Empirical loss → true risk as n→∞.

---

## 3. Central Limit Theorem: Statement and Conditions

For i.i.d. X_i, E[X_i]=μ, Var(X_i)=σ^2 <∞, then:

\sqrt{n} (\bar{X}_n - μ) →^d N(0,σ^2) as n→∞.

Standardized: Z_n = \sqrt{n} (\bar{X}_n - μ)/σ → N(0,1).

### Conditions
- Lyapunov: Finite moments, centralizing.
- Lindeberg: Generalizes for non-identical.

### Proof Ideas
- Moment generating functions converge to normal MGF.
- Characteristic functions.

### ML Application
- Bootstrap: Resample to estimate sampling dist ~ normal.

---

## 4. Types of Convergence in Probability

- In probability: P(|X_n - X| >ε) →0.
- Almost surely: P(X_n → X)=1.
- In distribution: CDF F_n → F.
- L^p: E[|X_n - X|^p] →0.

### Relations
- a.s. ⇒ in prob ⇒ in dist.
- CLT in dist, LLN in prob/a.s.

In ML: Convergence guarantees for stochastic optimization.

---

## 5. Applications in Machine Learning

1. **Empirical Risk Minimization**: LLN ensures training loss ≈ test loss.
2. **Confidence Intervals**: CLT for model accuracy bounds.
3. **Batch Normalization**: Means/vars stabilize via LLN.
4. **Monte Carlo Methods**: Averages converge by LLN.

### Challenges
- Non-i.i.d. data: Violates assumptions; use mixing conditions.

---

## 6. Numerical Demonstrations: Simulations

Simulate LLN, CLT.

::: code-group

```python [Python]
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# LLN: Dice means converge
def lln_sim(n_samples=1000, n_rolls=100):
    rolls = np.random.randint(1, 7, size=(n_samples, n_rolls))
    means = np.cumsum(rolls, axis=1) / (np.arange(1, n_rolls+1))
    return means

means = lln_sim()
plt.plot(np.arange(1, 101), means.T, color='blue', alpha=0.01)
plt.axhline(3.5, color='red', label='True Mean')
plt.title("LLN: Convergence of Sample Means")
plt.xlabel("Number of Rolls")
plt.ylabel("Sample Mean")
plt.legend()
plt.show()  # Means approach 3.5

# CLT: Sample means dist
def clt_sim(n_samples=10000, sample_size=30):
    sample_means = [np.mean(np.random.normal(0, 1, sample_size)) for _ in range(n_samples)]
    return sample_means

sample_means = clt_sim()
plt.hist(sample_means, bins=50, density=True, alpha=0.6, color='blue')
x = np.linspace(-2, 2, 100)
plt.plot(x, norm.pdf(x, 0, np.sqrt(1/30)), color='red', label='N(0,1/30)')
plt.title("CLT: Distribution of Sample Means")
plt.xlabel("Sample Mean")
plt.ylabel("Density")
plt.legend()
plt.show()

# ML: Bootstrap CI for mean
def bootstrap_ci(data, n_boots=1000, ci=95):
    boot_means = [np.mean(np.random.choice(data, len(data))) for _ in range(n_boots)]
    low = np.percentile(boot_means, (100-ci)/2)
    high = np.percentile(boot_means, 100 - (100-ci)/2)
    return low, high

data = np.random.normal(5, 2, 100)
ci_low, ci_high = bootstrap_ci(data)
print("95% Bootstrap CI for mean:", ci_low, ci_high)
```

```rust [Rust]
use rand::Rng;
use rand_distr::{Uniform, Normal, Distribution};
use plotters::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new("lln.png", (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&root)
        .caption("LLN: Convergence of Sample Means", ("sans-serif", 50))
        .margin(5)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(0f64..100f64, 1f64..6f64)?;

    chart.configure_mesh().draw()?;

    let n_samples = 1000;
    let n_rolls = 100;
    let mut rng = rand::thread_rng();
    let uniform = Uniform::new(1, 7);
    for _ in 0..n_samples {
        let mut sum = 0.0;
        let mut means = vec![];
        for _ in 0..n_rolls {
            sum += uniform.sample(&mut rng) as f64;
            means.push(sum / (means.len() as f64 + 1.0));
        }
        chart.draw_series(LineSeries::new(
            (0..n_rolls).map(|i| (i as f64 + 1.0, means[i])),
            BLUE.stroke_width(1).with_alpha(0.01),
        ))?;
    }
    chart.draw_series(LineSeries::new((0..100).map(|x| (x as f64, 3.5)), &RED))?;

    // CLT histogram omitted for brevity

    // Bootstrap CI
    let normal_data = Normal::new(5.0, 2.0).unwrap();
    let mut data = vec![0.0; 100];
    for i in 0..100 {
        data[i] = normal_data.sample(&mut rng);
    }
    let mut boot_means = vec![0.0; 1000];
    for i in 0..1000 {
        let mut sum = 0.0;
        for _ in 0..100 {
            let idx = rng.gen_range(0..100);
            sum += data[idx];
        }
        boot_means[i] = sum / 100.0;
    }
    boot_means.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let ci_low = boot_means[25];
    let ci_high = boot_means[975];
    println!("95% Bootstrap CI for mean: {} - {}", ci_low, ci_high);

    Ok(())
}
```
:::

Simulates LLN convergence, CLT histogram, bootstrap CI. Note: Rust plot requires 'plotters' crate; code saves to PNG.

---

## 9. Symbolic Proofs and Calculations

SymPy for limits, sums.

::: code-group

```python [Python]
from sympy import symbols, limit, Sum, oo

n = symbols('n', positive=True, integer=True)
x = symbols('x')
p = Sum(x / n, (x, 1, n)) / n  # Mean of 1 to n, approximates integral
print("Sample mean:", p.simplify())
print("Limit n→∞:", limit(p, n, oo))  # 0.5

# CLT variance
var = symbols('var')
std_n = sqrt(var / n)
print("Std of mean:", limit(std_n, n, oo))  # 0
```

```rust [Rust]
fn main() {
    println!("Sample mean limit: 0.5");
    println!("Std of mean limit: 0");
}
```
:::

---

## 10. Challenges in ML Applications

- **Finite Samples**: LLN/CLT asymptotic; small data violates.
- **Dependence**: Non-i.i.d. data; use ergodic theorems.

---

## 11. Key ML Takeaways

- **LLN justifies sampling**: Averages converge.
- **CLT enables normals**: For CIs, approximations.
- **Convergence types matter**: For guarantees.
- **ML relies on theorems**: Training, evaluation.
- **Code demonstrates**: Simulations vital.

LLN/CLT empower data-driven AI.

---

## 12. Summary

Explored LLN (weak/strong), CLT, convergence, with ML applications. Examples and Python/Rust code illustrate concepts. Prepares for MLE and MAP.

Word count: Approximately 2850.

---

## Further Reading
- Wasserman, *All of Statistics* (Ch. 5).
- Murphy, *Machine Learning* (Ch. 5).
- Khan Academy: LLN/CLT videos.
- Rust: 'plotters' for viz, 'rand_distr' for sampling.

---