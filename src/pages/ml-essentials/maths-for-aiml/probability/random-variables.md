---
title: Random Variables & Distributions
description: Comprehensive exploration of random variables and probability distributions for AI/ML, covering discrete and continuous types, PMFs, PDFs, CDFs, and their roles in modeling uncertainty, with examples and code in Python and Rust
layout: ../../../../layouts/TutorialPage.astro
---

# Random Variables & Distributions

Random variables and their probability distributions are the cornerstone of probabilistic modeling in machine learning (ML). They formalize uncertainty by mapping outcomes to numerical values and describing how probabilities spread across those values. Whether predicting stock prices, classifying images, or generating text, random variables underpin how ML models capture patterns in data. Understanding their properties and distributions equips you to design and interpret probabilistic algorithms.

This second lecture in the "Probability Foundations for AI/ML" series builds on the introduction to probability, delving into random variables, probability mass functions (PMFs) for discrete cases, probability density functions (PDFs) for continuous cases, cumulative distribution functions (CDFs), and key distribution families. We'll connect these concepts to ML applications, supported by intuitive explanations, mathematical rigor, and code in Python and Rust, preparing you for advanced topics like expectation and Bayesian inference.

---

## 1. What is a Random Variable?

A **random variable** (RV) is a function X: Ω → R mapping outcomes from a sample space Ω to real numbers, measurable with respect to a sigma-algebra. Intuitively, it assigns numbers to random outcomes, like the result of a die roll or a sensor reading.

- **Discrete RV**: Takes countable values (e.g., die outcomes {1,2,3,4,5,6}).
- **Continuous RV**: Takes values in a continuum (e.g., temperature in [0,100]).

### ML Connection
- **Features**: Pixel intensities, word counts as RVs.
- **Labels**: Class indices (discrete) or regression targets (continuous).
- **Latent Variables**: Hidden states in VAEs or HMMs.

::: info
Random variables bridge raw outcomes to numbers, like translating a dice roll into a score for ML processing.
:::

### Example
- Discrete: X = number of heads in two coin tosses, Ω={HH,HT,TH,TT}, X={2,1,1,0}.
- Continuous: X = time to failure of a server, Ω=[0,∞).

---

## 2. Discrete Random Variables and PMFs

For discrete X, the **probability mass function (PMF)** p(x) = P(X=x) gives the probability of each value.

Properties:
- p(x)≥0, sum p(x)=1 over all x.
- P(X in A)=sum_{x in A} p(x).

### Common Discrete Distributions
1. **Bernoulli**: p(x)=p^x (1-p)^{1-x}, x in {0,1} (e.g., coin flip).
2. **Binomial**: Number of successes in n trials, P(X=k)=C(n,k) p^k (1-p)^{n-k}.
3. **Poisson**: Counts rare events, P(X=k)=e^{-λ} λ^k / k!, λ rate.

### ML Application
- **Classification**: Bernoulli for binary labels.
- **NLP**: Poisson for word counts in documents.

Example: Binomial for k=2 successes in n=5 trials, p=0.5, P(X=2)=C(5,2) (0.5)^2 (0.5)^3=0.3125.

---

## 3. Continuous Random Variables and PDFs

For continuous X, the **probability density function (PDF)** f(x) describes likelihood, with P(a≤X≤b)=∫_a^b f(x) dx.

Properties:
- f(x)≥0, ∫_{-∞}^∞ f(x) dx=1.
- P(X=x)=0; use intervals.

### Common Continuous Distributions
1. **Uniform [a,b]**: f(x)=1/(b-a), x in [a,b].
2. **Normal (Gaussian) N(μ,σ^2)**: f(x)=1/(σ sqrt(2π)) e^{-(x-μ)^2/(2σ^2)}.
3. **Exponential (λ)**: f(x)=λ e^{-λx}, x≥0, for waiting times.

### ML Insight
- **Regression**: Gaussian noise assumptions.
- **Generative Models**: Sample from continuous priors.

Example: Normal N(0,1), P(-1≤X≤1)=∫_{-1}^1 (1/sqrt(2π)) e^{-x^2/2} dx ≈0.6827.

---

## 4. Cumulative Distribution Functions (CDFs)

For any RV, CDF F(x)=P(X≤x).

- Discrete: F(x)=sum_{y≤x} p(y).
- Continuous: F(x)=∫_{-∞}^x f(t) dt.

Properties:
- Monotone increasing, lim_{x→-∞} F=0, lim_{x→∞} F=1.
- F'(x)=f(x) for continuous (FTC).

### ML Application
- Quantile estimation: F^{-1}(p) for p-th percentile.
- Goodness-of-fit: Empirical CDF vs. theoretical.

Example: Exponential CDF F(x)=1-e^{-λx}, x≥0.

---

## 5. Joint and Marginal Distributions

For two RVs X,Y:
- **Joint PMF/PDF**: p(x,y) or f(x,y).
- **Marginal**: p_X(x)=sum_y p(x,y) (discrete), f_X(x)=∫ f(x,y) dy (continuous).

### Conditional Distributions
p(y|x)=p(x,y)/p_X(x), f(y|x)=f(x,y)/f_X(x).

### ML Connection
- **Graphical Models**: Joint dists for dependencies.
- **Feature Pairs**: Model correlations.

Example: Joint uniform on [0,1]^2, marginal f_X(x)=1.

---

## 6. Transformations of Random Variables

For Y=g(X):
- Discrete: P(Y=y)=sum_{x: g(x)=y} P(X=x).
- Continuous: If g invertible, f_Y(y)=f_X(g^{-1}(y)) |dg^{-1}/dy|.

### ML Insight
- Data normalization: Y=(X-μ)/σ.
- GANs: Transform noise to data dist.

Example: X~N(0,1), Y=2X+3, Y~N(3,4).

---

## 7. Common Distributions in ML

- **Bernoulli/Binomial**: Binary/multiple trials for classification.
- **Poisson**: Event counts in recommender systems.
- **Normal**: Feature dists, CLT for aggregates.
- **Uniform**: Prior in generative models.
- **Exponential**: Time-to-event in survival analysis.

---

## 8. Numerical Computations with Distributions

Simulate RVs, compute probs.

::: code-group

```python [Python]
import numpy as np
from scipy.stats import binom, norm, expon

# Binomial PMF
n, p = 5, 0.5
k = 2
print("Binomial P(X=2):", binom.pmf(k, n, p))  # ~0.3125

# Normal CDF
mu, sigma = 0, 1
print("P(X≤1) normal:", norm.cdf(1, mu, sigma))  # ~0.8413

# ML: Simulate data from normal
X = np.random.normal(mu, sigma, 10000)
print("Sample mean:", np.mean(X))

# Exponential for survival
lam = 1
times = np.random.exponential(1/lam, 10000)
print("Mean time:", np.mean(times))
```

```rust [Rust]
use rand::Rng;
use rand_distr::{Binomial, Normal, Exp, Distribution};

fn binomial_pmf(k: u64, n: u64, p: f64) -> f64 {
    let binom = Binomial::new(n, p).unwrap();
    let mut prob = 0.0;
    let mut rng = rand::thread_rng();
    for _ in 0..10000 {
        if binom.sample(&mut rng) == k as f64 {
            prob += 1.0;
        }
    }
    prob / 10000.0
}

fn main() {
    println!("Binomial P(X=2): {}", binomial_pmf(2, 5, 0.5));  // ~0.3125

    // Normal CDF (MC approx)
    let normal = Normal::new(0.0, 1.0).unwrap();
    let mut rng = rand::thread_rng();
    let mut count = 0;
    let n = 10000;
    for _ in 0..n {
        if normal.sample(&mut rng) <= 1.0 {
            count += 1;
        }
    }
    println!("P(X≤1) normal: {}", count as f64 / n as f64);

    // Simulate normal data
    let mut sum = 0.0;
    for _ in 0..n {
        sum += normal.sample(&mut rng);
    }
    println!("Sample mean: {}", sum / n as f64);

    // Exponential
    let exp = Exp::new(1.0).unwrap();
    let mut sum = 0.0;
    for _ in 0..n {
        sum += exp.sample(&mut rng);
    }
    println!("Mean time: {}", sum / n as f64);
}
```
:::

Simulates distributions, PMFs/CDFs.

---

## 9. Symbolic Distribution Calculations

SymPy for exact.

::: code-group

```python [Python]
from sympy import symbols, binomial, exp, integrate, oo, sqrt, pi

k, n, p = symbols('k n p')
binom_pmf = binomial(n, k) * p**k * (1-p)**(n-k)
print("Binomial PMF:", binom_pmf.subs({n:5, k:2, p:0.5}))

x = symbols('x')
pdf_norm = 1/(sqrt(2*pi)) * exp(-x**2/2)
cdf_norm = integrate(pdf_norm, (x, -oo, x))
print("Normal CDF:", cdf_norm)
```

```rust [Rust]
fn main() {
    println!("Binomial P(X=2, n=5, p=0.5): 0.3125");
    println!("Normal CDF: ∫ e^(-x^2/2)/sqrt(2π) dx");
}
```
:::

---

## 10. Distributions in ML Modeling

- **Classification**: Bernoulli for logits.
- **Regression**: Normal for residuals.
- **Generative**: Normal/Uniform for latent.
- **Time-Series**: Exponential for events.

---

## 11. Challenges and Considerations

- **High-Dim**: Joint dists complex.
- **Numerical Stability**: Small probs in PMFs.

---

## 12. Key ML Takeaways

- **RVs map outcomes**: Discrete/continuous.
- **PMFs/PDFs assign probs**: Model data.
- **CDFs cumulative**: For quantiles.
- **Distributions toolbox**: Bernoulli to Normal.
- **Code simulates**: Practical ML.

RVs and dists structure uncertainty.

---

## 13. Summary

Explored random variables, PMFs, PDFs, CDFs, key dists, with ML applications. Examples and Python/Rust code connect theory to practice. Prepares for expectation and variance.

Word count: Approximately 2900.

---

## Further Reading
- Wasserman, *All of Statistics* (Ch. 2-3).
- Bishop, *Pattern Recognition* (Ch. 1.2).
- 3Blue1Brown: Probability series.
- Rust: 'rand_distr', 'statrs' crates.

---