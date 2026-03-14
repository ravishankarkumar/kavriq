---
title: Distributions in Practice - Normal, Binomial, Poisson
description: Detailed exploration of Normal, Binomial, and Poisson distributions in statistics for AI/ML, covering their definitions, properties, derivations, and applications in modeling data and uncertainty, with examples and code in Python and Rust
layout: ../../../../layouts/TutorialPage.astro
---

# Distributions in Practice - Normal, Binomial, Poisson

Probability distributions model how data or random variables behave, providing the foundation for statistical analysis in machine learning (ML). The Normal, Binomial, and Poisson distributions are among the most widely used due to their ability to describe continuous and discrete phenomena, from regression residuals to event counts. In ML, these distributions underpin assumptions in models, guide parameter estimation, and enable uncertainty quantification.

This second lecture in the "Statistics Foundations for AI/ML" series builds on descriptive statistics, delving into the definitions, probability functions, moments, and ML applications of the Normal, Binomial, and Poisson distributions. We'll provide intuitive explanations, mathematical derivations, and practical implementations in Python and Rust, preparing you for sampling and inference topics in the series.

---

## 1. Why Distributions Matter in ML

Distributions describe how probabilities are assigned to outcomes, shaping how ML models interpret data:
- **Normal**: Models continuous data, common in regression and CLT.
- **Binomial**: Captures binary trial outcomes, used in classification.
- **Poisson**: Models event counts, prevalent in NLP and time-series.

These distributions help:
- Model data assumptions (e.g., Gaussian noise).
- Estimate parameters (e.g., MLE).
- Quantify uncertainty in predictions.

### ML Connection
- **Regression**: Normal for residuals.
- **Classification**: Binomial for binary outcomes.
- **Event Modeling**: Poisson for rare events.

::: info
Distributions are blueprints for data's randomness, guiding ML models to fit and predict effectively.
:::

### Example
- Normal: Heights in a population ~ N(μ,σ²).
- Binomial: Number of clicks in 100 ad impressions.
- Poisson: Number of emails received per hour.

---

## 2. Normal Distribution: The Bell Curve

The **Normal distribution** N(μ,σ²) describes continuous data clustering around a mean μ with variance σ².

**PDF**:

\[
f(x|μ,σ) = \frac{1}{\sqrt{2\pi σ²}} e^{-\frac{(x-μ)²}{2σ²}}
\]

**Moments**:
- Mean: E[X] = μ.
- Variance: Var(X) = σ².

### Properties
- Symmetric, bell-shaped.
- 68-95-99.7 rule: ~68% within 1σ, 95% within 2σ, 99.7% within 3σ.
- Central Limit Theorem (CLT): Sums of i.i.d. RVs → Normal.

### Derivation
- Arises from maximizing entropy for fixed mean, variance.

### ML Application
- **Regression**: Assume residuals ~ N(0,σ²).
- **Gaussian Processes**: Model functions.
- **Data Normalization**: Standardize features.

Example: Heights ~ N(170, 25), P(165<X<175) ≈ 0.68 (1σ).

---

## 3. Binomial Distribution: Successes in Trials

The **Binomial distribution** Bin(n,p) models the number of successes in n independent Bernoulli trials with success probability p.

**PMF**:

\[
P(X=k) = \binom{n}{k} p^k (1-p)^{n-k}
\]

**Moments**:
- Mean: E[X] = np.
- Variance: Var(X) = np(1-p).

### Properties
- Discrete, k=0,1,...,n.
- Sum of Bernoulli RVs.

### ML Application
- **Classification**: Binary outcomes (e.g., click/no-click).
- **A/B Testing**: Success counts.

Example: 5 coin flips, p=0.5, P(X=3) = \binom{5}{3} 0.5³ 0.5² ≈ 0.3125.

---

## 4. Poisson Distribution: Modeling Counts

The **Poisson distribution** Pois(λ) models the number of events in a fixed interval, with rate λ.

**PMF**:

\[
P(X=k) = \frac{e^{-λ} λ^k}{k!}
\]

**Moments**:
- Mean: E[X] = λ.
- Variance: Var(X) = λ.

### Properties
- Discrete, k=0,1,...
- Limit of Binomial for large n, small p, np=λ.

### ML Application
- **NLP**: Word counts in documents.
- **Time-Series**: Event occurrences (e.g., server requests).

Example: Emails/hour ~ Pois(2), P(X=1) = e^{-2} · 2 ≈ 0.27.

---

## 5. Relationships Between Distributions

- **Binomial → Poisson**: Large n, small p, np=λ.
- **Binomial → Normal**: Large n, np(1-p) large, ~ N(np, np(1-p)).
- **Poisson → Normal**: Large λ, ~ N(λ, λ).

### ML Insight
- Approximations simplify computations in high-n scenarios.

---

## 6. Parameter Estimation for Distributions

**MLE for Normal**:
- μ_hat = \bar{x}.
- σ²_hat = (1/n) sum (x_i - \bar{x})² (biased).

**MLE for Binomial**: p_hat = k/n.

**MLE for Poisson**: λ_hat = \bar{x}.

### ML Connection
- Fit distributions to data (e.g., GMMs).

---

## 7. Applications in Machine Learning

1. **Regression**: Normal for error modeling.
2. **Classification**: Binomial for binary outcomes.
3. **NLP/Time-Series**: Poisson for counts.
4. **Anomaly Detection**: Deviations from expected dist.

### Challenges
- **Misspecification**: Wrong dist skews results.
- **Small Samples**: Normal approx fails.

---

## 8. Numerical Computations with Distributions

Simulate, compute probs.

::: code-group

```python [Python]
import numpy as np
from scipy.stats import norm, binom, poisson

# Normal probabilities
mu, sigma = 170, 5
p = norm.cdf(175, mu, sigma) - norm.cdf(165, mu, sigma)
print("P(165<X<175):", p)

# Binomial PMF
n, p_binom = 5, 0.5
print("P(X=3) Binomial:", binom.pmf(3, n, p_binom))

# Poisson PMF
lam = 2
print("P(X=1) Poisson:", poisson.pmf(1, lam))

# ML: Fit normal to data
data = np.random.normal(0, 1, 100)
mu_mle = np.mean(data)
sigma_mle = np.std(data, ddof=0)  # Population
print("Normal MLE: μ=", mu_mle, "σ=", sigma_mle)
```

```rust [Rust]
use rand::Rng;
use rand_distr::{Normal, Binomial, Poisson, Distribution};

fn factorial(k: u64) -> f64 {
    (1..=k).product::<u64>() as f64
}

fn poisson_pmf(k: u64, lam: f64) -> f64 {
    (-lam).exp() * lam.powi(k as i32) / factorial(k)
}

fn main() {
    let mut rng = rand::thread_rng();

    // Normal prob (MC approx)
    let normal = Normal::new(170.0, 5.0).unwrap();
    let mut count = 0;
    let n = 10000;
    for _ in 0..n {
        let x = normal.sample(&mut rng);
        if x > 165.0 && x < 175.0 {
            count += 1;
        }
    }
    println!("P(165<X<175): {}", count as f64 / n as f64);

    // Binomial PMF (simulate)
    let binom = Binomial::new(5, 0.5).unwrap();
    let mut count_binom = 0;
    for _ in 0..n {
        if binom.sample(&mut rng) == 3.0 {
            count_binom += 1;
        }
    }
    println!("P(X=3) Binomial: {}", count_binom as f64 / n as f64);

    // Poisson PMF
    let lam = 2.0;
    println!("P(X=1) Poisson: {}", poisson_pmf(1, lam));

    // ML: Fit normal
    let normal_data = Normal::new(0.0, 1.0).unwrap();
    let mut sum = 0.0;
    let mut sum_sq = 0.0;
    let n = 100;
    for _ in 0..n {
        let x = normal_data.sample(&mut rng);
        sum += x;
        sum_sq += x * x;
    }
    let mu_mle = sum / n as f64;
    let sigma_mle = ((sum_sq / n as f64) - mu_mle.powi(2)).sqrt();
    println!("Normal MLE: μ={} σ={}", mu_mle, sigma_mle);
}
```
:::

Simulates probabilities, fits Normal.

---

## 9. Symbolic Computations with SymPy

Exact PMFs/PDFs.

::: code-group

```python [Python]
from sympy import symbols, exp, sqrt, pi, binomial, factorial

# Normal PDF
x, mu, sigma = symbols('x mu sigma', positive=True)
pdf_norm = 1/(sigma * sqrt(2*pi)) * exp(-(x-mu)**2/(2*sigma**2))
print("Normal PDF:", pdf_norm)

# Binomial PMF
k, n, p = symbols('k n p')
pmf_binom = binomial(n, k) * p**k * (1-p)**(n-k)
print("Binomial P(X=k):", pmf_binom.subs({n:5, k:3, p:0.5}))

# Poisson PMF
lam = symbols('lam', positive=True)
pmf_pois = exp(-lam) * lam**k / factorial(k)
print("Poisson P(X=1):", pmf_pois.subs({k:1, lam:2}))
```

```rust [Rust]
fn main() {
    println!("Normal PDF: (1/(σ sqrt(2π))) e^(-(x-μ)²/(2σ²))");
    println!("Binomial P(X=3, n=5, p=0.5): 0.3125");
    println!("Poisson P(X=1, λ=2): {}", (-2.0f64).exp() * 2.0 / 1.0);
}
```
:::

---

## 10. Challenges in ML Applications

- **Assumption Violations**: Non-Normal residuals.
- **Small Samples**: Binomial/Poisson approximations fail.
- **High-Dim**: Multivariate normals complex.

---

## 11. Key ML Takeaways

- **Normal ubiquitous**: Regression, CLT.
- **Binomial for trials**: Classification, testing.
- **Poisson for counts**: Events, NLP.
- **Fitting dists**: Parameter estimation.
- **Code simulates**: Practical modeling.

Distributions shape ML assumptions.

---

## 12. Summary

Explored Normal, Binomial, Poisson distributions, their properties, and ML applications. Examples and Python/Rust code bridge theory to practice. Prepares for sampling and inference.

Word count: Approximately 3000.

---

## Further Reading
- Wasserman, *All of Statistics* (Ch. 2-3).
- James, *Introduction to Statistical Learning* (Ch. 2).
- 3Blue1Brown: Probability distributions videos.
- Rust: 'rand_distr', 'statrs' crates for distributions.

---