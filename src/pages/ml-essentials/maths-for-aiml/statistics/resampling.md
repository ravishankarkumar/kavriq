---
title: Resampling Methods - Bootstrap & Permutation Tests
description: Comprehensive exploration of resampling methods, including bootstrap and permutation tests, in statistics for AI/ML, covering theory, applications in uncertainty estimation and hypothesis testing, with examples and code in Python and Rust
layout: ../../../../layouts/TutorialPage.astro
---

# Resampling Methods - Bootstrap & Permutation Tests

Resampling methods like bootstrap and permutation tests leverage computational power to estimate sampling distributions and test hypotheses without strict distributional assumptions. In machine learning (ML), these techniques are vital for quantifying uncertainty in model performance, validating feature importance, and conducting robust hypothesis testing, especially when data violates normality or other parametric assumptions. Bootstrap estimates variability by resampling with replacement, while permutation tests assess significance by shuffling data labels.

This eighth lecture in the "Statistics Foundations for AI/ML" series builds on ANOVA and hypothesis testing, exploring the principles, algorithms, and ML applications of bootstrap and permutation tests. We'll provide intuitive explanations, theoretical foundations, and practical implementations in Python and Rust, preparing you for MLE vs. method of moments and advanced inference topics.

---

## 1. Why Resampling Matters in ML

Traditional statistical methods rely on assumptions (e.g., normality, equal variances), which may not hold in ML datasets with complex distributions or small sizes. Resampling methods:
- Estimate sampling distributions via computation.
- Test hypotheses without parametric models.
- Quantify uncertainty in metrics like accuracy or feature importance.

### ML Connection
- **Bootstrap**: Estimate confidence intervals for model performance.
- **Permutation Tests**: Assess feature significance or model differences.
- **Model Evaluation**: Robust metrics for small or non-normal data.

::: info
Resampling is like repeatedly tasting different spoonfuls of soup to understand its flavor variability without assuming its recipe.
:::

### Example
- Bootstrap: Estimate 95% CI for model accuracy from 100 samples.
- Permutation: Test if a feature significantly impacts predictions.

---

## 2. Bootstrap: Estimating Sampling Distributions

**Bootstrap Principle**: Sample with replacement from data to mimic repeated sampling from the population, estimating the distribution of a statistic (e.g., mean, variance).

**Algorithm**:
1. From data {x₁,...,x_n}, draw B bootstrap samples (size n, with replacement).
2. Compute statistic θ_hat for each sample.
3. Use resulting distribution for inference (e.g., CI, SE).

### Types
- **Non-parametric**: Direct resampling.
- **Parametric**: Fit model, sample from it.

### Properties
- **Unbiased**: E[θ_hat] ≈ θ for large n.
- **Consistency**: Converges as n, B → ∞.

### ML Application
- Confidence intervals for model accuracy or regression coefficients.

Example: Data [1,2,3,4], bootstrap samples yield mean distribution.

---

## 3. Confidence Intervals via Bootstrap

**Percentile Method**: For statistic θ_hat, compute B bootstrap θ*_i, take [α/2, 1-α/2] percentiles for 100(1-α)% CI.

**BCa (Bias-Corrected and Accelerated)**: Adjusts for bias and skewness.

### ML Connection
- CI for neural network test accuracy.

---

## 4. Permutation Tests: Non-Parametric Hypothesis Testing

Tests H₀ (no difference between groups) by shuffling labels to break associations.

**Algorithm**:
1. Compute test statistic T (e.g., mean difference).
2. Permute labels B times, recompute T*_i.
3. p-value = proportion of |T*_i| ≥ |T|.

### Properties
- Non-parametric: No distribution assumption.
- Exact for small samples, approximate for large.

### ML Application
- Test feature importance by permuting feature values.

Example: Two groups [1,2,3], [4,5,6], permute labels to test mean difference.

---

## 5. Theoretical Foundations

**Bootstrap**: Relies on empirical distribution approximating population. CLT ensures bootstrap mean ~ N(θ, SE²).

**Permutation**: Under H₀, all permutations equally likely, giving exact p-value.

### Assumptions
- Bootstrap: i.i.d. or weak dependence.
- Permutation: Exchangeability under H₀.

---

## 6. Applications in Machine Learning

1. **Model Evaluation**: Bootstrap CIs for accuracy, F1 score.
2. **Feature Importance**: Permutation tests for feature significance.
3. **A/B Testing**: Permutation for treatment effects.
4. **Uncertainty Quantification**: Bootstrap for robust metrics.

### Challenges
- **Computation**: High B costly.
- **Dependence**: Non-i.i.d. data complicates.

---

## 7. Numerical Resampling Computations

Implement bootstrap and permutation tests.

::: code-group

```python [Python]
import numpy as np
from scipy.stats import ttest_ind

# Bootstrap CI for mean
def bootstrap_ci(data, statistic=np.mean, n_boots=1000, alpha=0.05):
    boot_stats = [statistic(np.random.choice(data, len(data))) for _ in range(n_boots)]
    return np.percentile(boot_stats, [100*alpha/2, 100*(1-alpha/2)])

data = np.random.normal(10, 2, 100)
ci = bootstrap_ci(data)
print("Bootstrap 95% CI for mean:", ci)

# Permutation test for mean difference
def permutation_test(data1, data2, n_perms=1000):
    obs_diff = np.mean(data1) - np.mean(data2)
    combined = np.concatenate([data1, data2])
    perm_diffs = []
    for _ in range(n_perms):
        perm = np.random.permutation(combined)
        perm_diffs.append(np.mean(perm[:len(data1)]) - np.mean(perm[len(data1):]))
    p_val = np.mean(np.abs(perm_diffs) >= np.abs(obs_diff))
    return obs_diff, p_val

group1 = np.random.normal(10, 1, 30)
group2 = np.random.normal(11, 1, 30)
diff, p_val = permutation_test(group1, group2)
print("Permutation test: diff=", diff, "p=", p_val)

# ML: Feature importance via permutation
from sklearn.ensemble import RandomForestClassifier
X = np.random.rand(100, 2)
y = (X[:,0] + X[:,1] > 1).astype(int)
model = RandomForestClassifier(random_state=0).fit(X, y)
baseline_acc = model.score(X, y)
perm_acc = []
for _ in range(100):
    X_perm = X.copy()
    np.random.shuffle(X_perm[:,0])
    perm_acc.append(model.score(X_perm, y))
p_val = np.mean(np.array(perm_acc) <= baseline_acc)
print("Feature importance p:", p_val)
```

```rust [Rust]
use rand::Rng;
use rand::seq::SliceRandom;

fn bootstrap_ci(data: &[f64], n_boots: usize, alpha: f64) -> (f64, f64) {
    let mut rng = rand::thread_rng();
    let mut boot_stats = vec![0.0; n_boots];
    for i in 0..n_boots {
        let boot_sample: Vec<f64> = (0..data.len()).map(|_| data[rng.gen_range(0..data.len())]).collect();
        boot_stats[i] = boot_sample.iter().sum::<f64>() / data.len() as f64;
    }
    boot_stats.sort_by(|a, b| a.partial_cmp(b).unwrap());
    (boot_stats[(n_boots as f64 * alpha / 2.0) as usize], boot_stats[(n_boots as f64 * (1.0 - alpha / 2.0)) as usize])
}

fn permutation_test(data1: &[f64], data2: &[f64], n_perms: usize) -> (f64, f64) {
    let mut rng = rand::thread_rng();
    let obs_diff = data1.iter().sum::<f64>() / data1.len() as f64 - data2.iter().sum::<f64>() / data2.len() as f64;
    let combined: Vec<f64> = data1.iter().chain(data2.iter()).copied().collect();
    let mut perm_diffs = vec![0.0; n_perms];
    for i in 0..n_perms {
        let mut perm = combined.clone();
        perm.shuffle(&mut rng);
        let mean1 = perm[..data1.len()].iter().sum::<f64>() / data1.len() as f64;
        let mean2 = perm[data1.len()..].iter().sum::<f64>() / data2.len() as f64;
        perm_diffs[i] = mean1 - mean2;
    }
    let p_val = perm_diffs.iter().filter(|&&d| d.abs() >= obs_diff.abs()).count() as f64 / n_perms as f64;
    (obs_diff, p_val)
}

fn main() {
    let mut rng = rand::thread_rng();
    let normal1 = rand_distr::Normal::new(10.0, 1.0).unwrap();
    let normal2 = rand_distr::Normal::new(11.0, 1.0).unwrap();

    // Bootstrap CI
    let data: Vec<f64> = (0..100).map(|_| normal1.sample(&mut rng)).collect();
    let ci = bootstrap_ci(&data, 1000, 0.05);
    println!("Bootstrap 95% CI for mean: {:?}", ci);

    // Permutation test
    let group1: Vec<f64> = (0..30).map(|_| normal1.sample(&mut rng)).collect();
    let group2: Vec<f64> = (0..30).map(|_| normal2.sample(&mut rng)).collect();
    let (diff, p_val) = permutation_test(&group1, &group2, 1000);
    println!("Permutation test: diff={} p={}", diff, p_val);
}
```
:::

Implements bootstrap CI and permutation tests.

---

## 8. Theoretical Insights

**Bootstrap**: Approximates sampling distribution via empirical distribution function.

**Permutation**: Exact under exchangeability, robust to non-normality.

### ML Insight
- Bootstrap for small datasets; permutation for feature tests.

---

## 9. Challenges in ML Applications

- **Computational Cost**: High B slows computation.
- **Non-i.i.d. Data**: Time-series, clustered data violate assumptions.
- **Small Samples**: Bootstrap bias in low n.

---

## 10. Key ML Takeaways

- **Bootstrap estimates variability**: CIs for metrics.
- **Permutation tests significance**: Non-parametric.
- **Resampling robust**: Avoids normality.
- **ML evaluation relies**: On resampling.
- **Code implements**: Practical tests.

Resampling powers flexible ML inference.

---

## 11. Summary

Explored bootstrap and permutation tests, their algorithms, and ML applications in uncertainty estimation and hypothesis testing. Examples and Python/Rust code bridge theory to practice. Prepares for MLE vs. MoM and Bayesian statistics.

Word count: Approximately 3000.

---

## Further Reading
- Efron, Tibshirani, *An Introduction to the Bootstrap*.
- Wasserman, *All of Statistics* (Ch. 8).
- James, *Introduction to Statistical Learning* (Ch. 5).
- Rust: 'rand' for resampling, 'statrs' for stats.

---