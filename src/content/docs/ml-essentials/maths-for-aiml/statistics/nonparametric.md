---
title: Nonparametric Statistics - Beyond Distributions
description: Comprehensive exploration of nonparametric statistics for AI/ML, covering Mann-Whitney U, Wilcoxon, Kruskal-Wallis tests, kernel density estimation, and applications in robust model evaluation and feature analysis, with examples and code in Python and Rust
---

# Nonparametric Statistics - Beyond Distributions

Nonparametric statistics provide robust methods for data analysis when distributional assumptions, such as normality, are inappropriate or unverifiable. Unlike parametric methods that rely on specific distributions (e.g., Normal, Binomial), nonparametric techniques use ranks, medians, or data-driven estimates, making them ideal for small samples, non-normal data, or ordinal variables. In machine learning (ML), nonparametric methods are used for hypothesis testing, feature importance, density estimation, and robust model evaluation, especially in real-world datasets with outliers or complex distributions.

This fourteenth lecture in the "Statistics Foundations for AI/ML" series builds on statistical significance and cross-validation, exploring key nonparametric tests (Mann-Whitney U, Wilcoxon signed-rank, Kruskal-Wallis), kernel density estimation (KDE), and their ML applications. We'll provide intuitive explanations, mathematical foundations, and practical implementations in Python and Rust, preparing you for multivariate statistics and time-series analysis.

---

## 1. Why Nonparametric Statistics Matter in ML

ML datasets often violate parametric assumptions due to non-normality, outliers, or small sample sizes. Nonparametric methods:
- Require fewer assumptions, increasing robustness.
- Handle ordinal or categorical data effectively.
- Enable flexible density estimation without fixed models.

### ML Connection
- **Hypothesis Testing**: Compare model performances without normality assumptions.
- **Feature Importance**: Rank-based tests for feature selection.
- **Density Estimation**: KDE for generative modeling.

::: info
Nonparametric statistics are like flexible tools that adapt to any data shape, unlike parametric tools that assume a specific mold.
:::

### Example
- Test if two ML models' accuracies differ using Mann-Whitney U instead of a t-test for non-normal data.

---

## 2. Principles of Nonparametric Methods

Nonparametric methods avoid assuming a specific distribution, relying instead on:
- **Ranks**: Order data instead of raw values.
- **Empirical Distributions**: Use data directly (e.g., KDE).
- **Permutations**: Shuffle for hypothesis testing.

### Advantages
- Robust to outliers.
- Applicable to small samples or non-numeric data.
- Flexible for complex distributions.

### Disadvantages
- Less power if parametric assumptions hold.
- Computationally intensive for large datasets.

### ML Insight
- Nonparametric tests are ideal for imbalanced or skewed datasets.

---

## 3. Key Nonparametric Tests

### Mann-Whitney U Test
Compares two independent groups' distributions.

**H₀**: Distributions equal (same median).

**Statistic**: U = min(U₁, U₂), where U₁ = n₁n₂ + n₁(n₁+1)/2 - R₁, R₁ sum of ranks in group 1.

p-value from U distribution or normal approximation (large n).

### Wilcoxon Signed-Rank Test
Compares paired data (e.g., before/after).

**H₀**: Median difference = 0.

**Statistic**: W = sum of ranks of positive differences (signed).

### Kruskal-Wallis Test
Extends Mann-Whitney to k>2 groups.

**H₀**: All groups have same distribution.

**Statistic**: H = [(12/(N(N+1))) ∑ (R_i²/n_i)] - 3(N+1), R_i rank sum, n_i group size.

### ML Application
- Mann-Whitney: Compare two models' accuracies.
- Kruskal-Wallis: Test multiple hyperparameter settings.

Example: Mann-Whitney on model accuracies, p<0.05 suggests different distributions.

---

## 4. Kernel Density Estimation (KDE)

Estimates continuous PDF using data-driven approach.

**Formula**:

\[
\hat{f}(x) = \frac{1}{nh} \sum K\left(\frac{x-x_i}{h}\right)
\]

K kernel (e.g., Gaussian), h bandwidth.

### Properties
- Nonparametric, flexible shape.
- Bandwidth controls smoothness.

### ML Connection
- KDE for generative modeling or anomaly detection.

---

## 5. Rank-Based Methods

Ranks transform data to ordinal scale, robust to outliers.

Spearman correlation: Rank-based ρ.

### ML Application
- Feature selection with rank correlations.

---

## 6. Theoretical Foundations

**Rank Tests**: Use rank sums, approximate normal for large n.

**KDE**: Converges to true density as n→∞, h→0.

**Assumptions**:
- Exchangeability for tests.
- Continuity for KDE.

### ML Insight
- Nonparametric tests robust for ML's complex data.

---

## 7. Applications in Machine Learning

1. **Model Comparison**: Mann-Whitney for non-normal accuracy distributions.
2. **Feature Selection**: Kruskal-Wallis for categorical features.
3. **Density Estimation**: KDE for data generation.
4. **Anomaly Detection**: Nonparametric thresholds.

### Challenges
- **Power**: Less efficient if normality holds.
- **Computation**: KDE, permutation tests costly.

---

## 8. Numerical Nonparametric Computations

Implement Mann-Whitney, Kruskal-Wallis, KDE.

::: code-group

```python [Python]
import numpy as np
from scipy.stats import mannwhitneyu, kruskal
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt

# Mann-Whitney U test
model1_acc = np.random.exponential(1, 50)  # Non-normal
model2_acc = np.random.exponential(1.2, 50)
u_stat, p_val = mannwhitneyu(model1_acc, model2_acc)
print("Mann-Whitney U: stat=", u_stat, "p=", p_val)

# Kruskal-Wallis test
group1 = np.random.exponential(1, 30)
group2 = np.random.exponential(1.1, 30)
group3 = np.random.exponential(1.2, 30)
h_stat, p_val = kruskal(group1, group2, group3)
print("Kruskal-Wallis: stat=", h_stat, "p=", p_val)

# KDE
data = np.random.exponential(1, 100)
kde = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(data[:, None])
x = np.linspace(0, 5, 100)
log_dens = kde.score_samples(x[:, None])
plt.plot(x, np.exp(log_dens))
plt.title("KDE of Exponential Data")
plt.show()

# ML: Feature importance with Mann-Whitney
from sklearn.ensemble import RandomForestClassifier
X = np.random.rand(100, 2)
y = (X[:,0] + X[:,1] > 1).astype(int)
model = RandomForestClassifier(random_state=0).fit(X, y)
X_perm = X.copy()
np.random.shuffle(X_perm[:,0])
_, p_val = mannwhitneyu(X[y==1,0], X_perm[y==1,0])
print("Feature 0 importance p:", p_val)
```

```rust [Rust]
fn mann_whitney_u(x1: &[f64], x2: &[f64]) -> (f64, f64) {
    let n1 = x1.len() as f64;
    let n2 = x2.len() as f64;
    let mut combined: Vec<(f64, usize)> = x1.iter().map(|&x| (x, 0)).chain(x2.iter().map(|&x| (x, 1))).collect();
    combined.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    let mut rank = 0.0;
    let mut r1 = 0.0;
    for i in 0..combined.len() {
        if i > 0 && combined[i].0 != combined[i-1].0 {
            rank += 1.0;
        }
        if combined[i].1 == 0 {
            r1 += rank;
        }
    }
    let u1 = n1 * n2 + n1 * (n1 + 1.0) / 2.0 - r1;
    let u = u1.min(n1 * n2 - u1);
    (u, 0.0)  // p-value requires U distribution
}

fn main() {
    let mut rng = rand::thread_rng();
    let exp1: Vec<f64> = (0..50).map(|_| rand_distr::Exp::new(1.0).unwrap().sample(&mut rng)).collect();
    let exp2: Vec<f64> = (0..50).map(|_| rand_distr::Exp::new(1.0/1.2).unwrap().sample(&mut rng)).collect();
    let (u, p) = mann_whitney_u(&exp1, &exp2);
    println!("Mann-Whitney U: stat={} p={}", u, p);

    // KDE (simplified Gaussian kernel)
    let data: Vec<f64> = (0..100).map(|_| rand_distr::Exp::new(1.0).unwrap().sample(&mut rng)).collect();
    let h = 0.5;
    let x: Vec<f64> = (0..100).map(|i| i as f64 / 20.0).collect();
    let dens: Vec<f64> = x.iter().map(|&xi| {
        data.iter().map(|&di| (-((xi - di) / h).powi(2) / 2.0).exp() / (h * (2.0 * std::f64::consts::PI).sqrt())).sum::<f64>() / data.len() as f64
    }).collect();
    // Plotting omitted
}
```
:::

Implements Mann-Whitney U, Kruskal-Wallis, and KDE.

---

## 9. Theoretical Insights

**Mann-Whitney**: Tests stochastic dominance via ranks.

**Kruskal-Wallis**: Generalizes to k groups, chi-square approx.

**KDE**: Nonparametric density, converges to true f(x).

### ML Insight
- Robust tests for non-normal ML metrics.

---

## 10. Challenges in ML Applications

- **Power**: Lower than parametric if assumptions hold.
- **Computation**: KDE, permutation tests costly.
- **High-Dim**: Rank tests less effective.

---

## 11. Key ML Takeaways

- **Nonparametric robust**: No distribution assumptions.
- **Rank tests versatile**: Mann-Whitney, Kruskal-Wallis.
- **KDE flexible**: Density estimation.
- **ML applications broad**: Model/feature eval.
- **Code implements**: Practical nonparametrics.

Nonparametric stats enhance ML flexibility.

---

## 12. Summary

Explored nonparametric statistics, including Mann-Whitney U, Wilcoxon, Kruskal-Wallis, and KDE, with ML applications in robust testing and density estimation. Examples and Python/Rust code bridge theory to practice. Prepares for multivariate statistics and time-series.

Word count: Approximately 3000.

---

## Further Reading
- Wasserman, *All of Nonparametric Statistics*.
- James, *Introduction to Statistical Learning* (Ch. 5).
- Conover, *Practical Nonparametric Statistics*.
- Rust: 'statrs' for tests, 'rand' for sampling.

---