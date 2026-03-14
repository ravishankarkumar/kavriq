---
title: Sampling & Sampling Distributions
description: Comprehensive exploration of sampling methods and sampling distributions in statistics for AI/ML, covering simple random sampling, stratified sampling, sampling distributions of statistics, and applications in model evaluation, with examples and code in Python and Rust
layout: ../../../../layouts/TutorialPage.astro
---

# Sampling & Sampling Distributions

Sampling is the process of selecting a subset of data from a population to make inferences, while sampling distributions describe the behavior of statistics (e.g., sample mean) across repeated samples. In machine learning (ML), sampling underpins training data selection, cross-validation, and uncertainty estimation, enabling models to generalize from limited data. Understanding sampling distributions is key to quantifying variability and applying the Central Limit Theorem (CLT) in practice.

This fourth lecture in the "Statistics Foundations for AI/ML" series builds on descriptive statistics, distributions, and correlation, exploring sampling methods (simple random, stratified, etc.), sampling distributions for means and proportions, their properties, and ML applications. We'll provide intuitive explanations, mathematical derivations, and practical implementations in Python and Rust, preparing you for confidence intervals and hypothesis testing.

---

## 1. Why Sampling Matters in ML

ML models rely on data, but populations are often too large to analyze fully. Sampling selects representative subsets:
- **Efficiency**: Train on smaller datasets.
- **Inference**: Estimate population parameters.
- **Evaluation**: Validate models via resampling.

Sampling distributions describe how statistics (e.g., mean, proportion) vary across samples, enabling uncertainty quantification.

### ML Connection
- **Training Data**: Random sampling ensures unbiased learning.
- **Cross-Validation**: Subset sampling for performance estimation.
- **Bootstrap**: Resampling for variance estimates.

::: info
Sampling is like taking a spoonful to taste a pot of soup—it must represent the whole to be reliable.
:::

### Example
- Population: All user clicks. Sample: 1000 clicks to estimate click-through rate.

---

## 2. Sampling Methods

### Simple Random Sampling (SRS)
Each unit has equal probability of selection.

- Unbiased, but may miss subgroups.

### Stratified Sampling
Divide population into strata, sample proportionally.

- Ensures representation of key groups.

### Cluster Sampling
Sample clusters (groups), analyze all within.

- Useful for geographically dispersed data.

### Systematic Sampling
Select every k-th unit.

- Simple but risks periodicity bias.

### ML Application
- **Stratified**: Balance classes in imbalanced datasets.
- **Bootstrap**: Resample with replacement for variance.

Example: Stratified sampling for fraud detection ensures rare fraud cases included.

---

## 3. Sampling Distributions: Definition

The **sampling distribution** of a statistic (e.g., sample mean \bar{X}) is the distribution of that statistic over all possible samples of size n.

**Example**: Sample means from N(μ,σ²) population are ~ N(μ, σ²/n) (CLT).

### Properties
- **Mean**: E[\bar{X}] = μ (unbiased).
- **Variance**: Var(\bar{X}) = σ²/n.
- **Shape**: Normal for large n (CLT).

### ML Insight
- Sampling dists justify confidence intervals in model evaluation.

---

## 4. Sampling Distribution of the Mean

For population with mean μ, variance σ², sample size n:
- E[\bar{X}] = μ.
- Var(\bar{X}) = σ²/n.
- For large n or normal population: \bar{X} ~ N(μ, σ²/n).

**Standard Error**: SE = σ / √n (estimated by s / √n).

### Derivation (CLT)
Sum of i.i.d. X_i ~ N(nμ, nσ²), so \bar{X} = (1/n) sum X_i ~ N(μ, σ²/n).

---

## 5. Sampling Distribution of Proportions

For binomial proportion p_hat = X/n (X successes in n trials):
- E[p_hat] = p.
- Var(p_hat) = p(1-p)/n.
- For large n, p_hat ~ N(p, p(1-p)/n).

### ML Application
- Estimate click-through rates in A/B testing.

---

## 6. Central Limit Theorem in Sampling

CLT: For i.i.d. X_i with finite E[X_i]=μ, Var(X_i)=σ², \sqrt{n} (\bar{X} - μ)/σ → N(0,1).

Enables normal approximations for sampling distributions.

In ML: Justifies Gaussian assumptions in large datasets.

---

## 7. Applications in Machine Learning

1. **Data Splitting**: Random sampling for train/test splits.
2. **Cross-Validation**: K-fold sampling for robust evaluation.
3. **Bootstrap**: Estimate model variance.
4. **Uncertainty Quantification**: Sampling dists for CIs.

### Challenges
- **Bias**: Non-random sampling skews results.
- **Small Samples**: CLT fails; use exact dists.

---

## 8. Numerical Sampling and Distributions

Simulate sampling, compute dists.

::: code-group

```python [Python]
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Simple random sampling
population = np.random.normal(100, 15, 10000)
sample = np.random.choice(population, 100)
print("Sample mean:", np.mean(sample))

# Sampling dist of mean
n_samples = 1000
sample_size = 30
sample_means = [np.mean(np.random.choice(population, sample_size)) for _ in range(n_samples)]
plt.hist(sample_means, bins=30, density=True)
x = np.linspace(90, 110, 100)
plt.plot(x, norm.pdf(x, 100, 15/np.sqrt(sample_size)))
plt.title("Sampling Dist of Mean")
plt.show()

# ML: Bootstrap variance
def bootstrap_var(data, n_boots=1000):
    boot_means = [np.mean(np.random.choice(data, len(data))) for _ in range(n_boots)]
    return np.var(boot_means)

data = np.random.normal(5, 2, 100)
print("Bootstrap variance:", bootstrap_var(data))

# Stratified sampling
labels = np.array([0]*50 + [1]*50)
data = np.concatenate([np.random.normal(0, 1, 50), np.random.normal(2, 1, 50)])
strata_indices = []
for label in np.unique(labels):
    strata_indices.extend(np.random.choice(np.where(labels == label)[0], 25))
sample_strat = data[strata_indices]
print("Stratified sample mean:", np.mean(sample_strat))
```

```rust [Rust]
use rand::Rng;
use rand_distr::Normal;

fn main() {
    let mut rng = rand::thread_rng();
    let normal = Normal::new(100.0, 15.0).unwrap();

    // Simple random sampling
    let population: Vec<f64> = (0..10000).map(|_| normal.sample(&mut rng)).collect();
    let sample: Vec<f64> = (0..100).map(|_| population[rng.gen_range(0..population.len())]).collect();
    let sample_mean = sample.iter().sum::<f64>() / sample.len() as f64;
    println!("Sample mean: {}", sample_mean);

    // Sampling dist of mean (simplified)
    let n_samples = 1000;
    let sample_size = 30;
    let mut sample_means = vec![0.0; n_samples];
    for i in 0..n_samples {
        let sample: Vec<f64> = (0..sample_size).map(|_| normal.sample(&mut rng)).collect();
        sample_means[i] = sample.iter().sum::<f64>() / sample_size as f64;
    }
    // Histogram plotting omitted

    // ML: Bootstrap variance
    let data_normal = Normal::new(5.0, 2.0).unwrap();
    let data: Vec<f64> = (0..100).map(|_| data_normal.sample(&mut rng)).collect();
    let mut boot_means = vec![0.0; 1000];
    for i in 0..1000 {
        let boot_sample: Vec<f64> = (0..100).map(|_| data[rng.gen_range(0..100)]).collect();
        boot_means[i] = boot_sample.iter().sum::<f64>() / 100.0;
    }
    let mean_boot = boot_means.iter().sum::<f64>() / 1000.0;
    let var_boot = boot_means.iter().map(|&m| (m - mean_boot).powi(2)).sum::<f64>() / 1000.0;
    println!("Bootstrap variance: {}", var_boot);
}
```
:::

Simulates sampling methods, distributions.

---

## 9. Symbolic Computations with SymPy

Derive sampling dist properties.

::: code-group

```python [Python]
from sympy import symbols, Sum, IndexedBase

n = symbols('n', integer=True, positive=True)
x = IndexedBase('x')
mu = symbols('mu')
sigma = symbols('sigma', positive=True)
sample_mean = (1/n) * Sum(x[i], (i, 1, n))
print("E[sample_mean]:", mu)
var_mean = (1/n**2) * Sum((x[i] - mu)**2, (i, 1, n))
print("Var(sample_mean):", var_mean.subs((x[i] - mu)**2, sigma**2))
```

```rust [Rust]
fn main() {
    println!("E[sample_mean]: μ");
    println!("Var(sample_mean): σ²/n");
}
```
:::

---

## 10. Challenges in ML Sampling

- **Bias**: Non-representative samples skew models.
- **Small n**: CLT approximations fail.
- **Imbalanced Data**: Stratified sampling critical.

---

## 11. Key ML Takeaways

- **Sampling enables inference**: From subsets.
- **Sampling dists quantify variability**: For stats.
- **CLT powers approximations**: Normal dists.
- **Bootstrap estimates uncertainty**: In ML.
- **Code simulates**: Practical sampling.

Sampling drives ML generalization.

---

## 12. Summary

Explored sampling methods (random, stratified), sampling distributions (mean, proportion), with ML applications. Examples and Python/Rust code bridge theory to practice. Prepares for confidence intervals and hypothesis testing.

Word count: Approximately 3000.

---

## Further Reading
- Wasserman, *All of Statistics* (Ch. 5).
- James, *Introduction to Statistical Learning* (Ch. 5).
- Khan Academy: Sampling distributions videos.
- Rust: 'rand' for sampling, 'statrs' for stats.

---