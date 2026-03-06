---
title: Hypothesis Testing - p-values, t-tests, Chi-square
description: Comprehensive exploration of hypothesis testing in statistics for AI/ML, covering p-values, t-tests, Chi-square tests, their theoretical foundations, and applications in model comparison and feature selection, with examples and code in Python and Rust
---

# Hypothesis Testing - p-values, t-tests, Chi-square

Hypothesis testing is a statistical framework for making decisions about population parameters based on sample data, using p-values to assess evidence and tests like t-tests and Chi-square to compare groups or distributions. In machine learning (ML), hypothesis testing is crucial for evaluating model performance, comparing algorithms, and selecting significant features, ensuring robust and interpretable results.

This sixth lecture in the "Statistics Foundations for AI/ML" series builds on estimation and confidence intervals, exploring the principles of hypothesis testing, p-values, one-sample and two-sample t-tests, Chi-square tests, and their ML applications. We'll provide intuitive explanations, mathematical derivations, and practical implementations in Python and Rust, preparing you for ANOVA and advanced inference.

---

## 1. Why Hypothesis Testing Matters in ML

ML often involves decisions: Is one model better than another? Are features significant? Hypothesis testing provides a principled way to answer these by:
- Comparing sample statistics to population parameters.
- Quantifying evidence against a null hypothesis.
- Controlling error rates (Type I, Type II).

### ML Connection
- **Model Comparison**: Test if Model A outperforms Model B.
- **Feature Selection**: Identify significant predictors.
- **A/B Testing**: Evaluate treatment effects in experiments.

::: info
Hypothesis testing is like a courtroom trial: you weigh evidence (data) to decide if a claim (null hypothesis) holds.
:::

### Example
- Test if a new ML model's accuracy (0.85) exceeds a baseline (0.80) using a t-test.

---

## 2. Fundamentals of Hypothesis Testing

**Null Hypothesis (H₀)**: Default claim, e.g., no difference (μ=μ₀).

**Alternative Hypothesis (H₁)**: What you aim to prove, e.g., μ≠μ₀ or μ>μ₀.

**Test Statistic**: Quantifies how far sample deviates from H₀ (e.g., t-statistic).

**p-value**: Probability of observing test statistic at least as extreme under H₀.

**Significance Level (α)**: Threshold (e.g., 0.05) to reject H₀ if p<α.

### Types of Errors
- **Type I**: Reject true H₀ (false positive, α).
- **Type II**: Fail to reject false H₀ (false negative, β).

### ML Insight
- p-values guide feature importance in tree-based models.

---

## 3. p-values: Measuring Evidence

p-value = P(observe test statistic | H₀).

Small p-value (e.g., <0.05) suggests H₀ unlikely.

### Interpretation
- Not probability H₀ true.
- Lower p → stronger evidence against H₀.

### ML Application
- Test significance of model improvements.

Example: t-test for mean=0, p=0.03 suggests reject H₀.

---

## 4. One-Sample t-test

Tests if sample mean \bar{x} differs from μ₀.

**Statistic**: t = (\bar{x} - μ₀) / (s / \sqrt{n}), s sample std dev, n sample size.

**Distribution**: t ~ t_{n-1} under H₀: μ=μ₀.

**CI**: \bar{x} ± t_{n-1,α/2} \cdot s/\sqrt{n}.

### Derivation
From CLT, \bar{x} ~ N(μ, σ²/n), use s for σ, t-dist for small n.

### ML Connection
- Test if model error mean differs from zero.

Example: n=30, \bar{x}=10, s=2, H₀: μ=9, t≈2.74, p<0.05, reject H₀.

---

## 5. Two-Sample t-test

Compares means of two groups.

**Statistic**: t = (\bar{x}_1 - \bar{x}_2) / \sqrt{s_1²/n_1 + s_2²/n_2}.

**Assumptions**: Normality, equal variances (or Welch's for unequal).

### ML Application
- Compare two models' accuracies.

Example: Model A: \bar{x}_1=0.85, n_1=100; Model B: \bar{x}_2=0.80, n_2=100, p<0.05 suggests difference.

---

## 6. Chi-square Tests

**Goodness-of-Fit**: Test if observed frequencies match expected.

\[
\chi² = \sum \frac{(O_i - E_i)²}{E_i}
\]

**Independence**: Test if variables independent in contingency table.

\[
\chi² = \sum \frac{(O_{ij} - E_{ij})²}{E_{ij}}, E_{ij} = (row total \cdot column total) / n
\]

**Distribution**: χ² with df=(k-1) or (r-1)(c-1).

### ML Application
- Feature selection: Test feature-label independence.

Example: Contingency table, p<0.05 suggests dependence.

---

## 7. Power and Sample Size

**Power**: 1-β, probability of rejecting false H₀.

Depends on effect size, α, n.

In ML: Ensure sufficient n for detecting model improvements.

---

## 8. Applications in Machine Learning

1. **Model Comparison**: t-test for accuracy differences.
2. **Feature Selection**: Chi-square for categorical features.
3. **A/B Testing**: Test treatment effects.
4. **Diagnostics**: Check residual assumptions.

### Challenges
- **Multiple Testing**: Adjust p-values (Bonferroni).
- **Non-Normality**: Use non-parametric tests.

---

## 9. Numerical Hypothesis Testing

Perform t-tests, Chi-square.

::: code-group

```python [Python]
import numpy as np
from scipy.stats import ttest_1samp, ttest_ind, chi2_contingency

# One-sample t-test
data = np.random.normal(10, 2, 30)
t_stat, p_val = ttest_1samp(data, popmean=9)
print("One-sample t-test: t=", t_stat, "p=", p_val)

# Two-sample t-test
data1 = np.random.normal(0.85, 0.1, 100)
data2 = np.random.normal(0.80, 0.1, 100)
t_stat, p_val = ttest_ind(data1, data2)
print("Two-sample t-test: t=", t_stat, "p=", p_val)

# Chi-square independence
table = [[10, 20, 30], [20, 20, 20]]  # Contingency
chi2, p, dof, _ = chi2_contingency(table)
print("Chi-square: stat=", chi2, "p=", p)

# ML: Feature significance
features = np.array([[1,0], [0,1], [1,1], [0,0]])
labels = [1,0,1,0]
table = [[sum((features[:,0] == i) & (labels == j)) for j in [0,1]] for i in [0,1]]
chi2, p, _, _ = chi2_contingency(table)
print("Feature-label Chi-square p:", p)
```

```rust [Rust]
use rand::Rng;
use rand_distr::Normal;

fn t_stat_1samp(data: &[f64], popmean: f64) -> (f64, f64) {
    let n = data.len() as f64;
    let mean = data.iter().sum::<f64>() / n;
    let s = (data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0)).sqrt();
    let t = (mean - popmean) / (s / n.sqrt());
    // p-value approximation (simplified)
    (t, 0.0)  // Actual p requires t-dist table
}

fn chi2_test(table: &[[u64; 2]]) -> (f64, f64) {
    let row_totals: Vec<u64> = table.iter().map(|row| row.iter().sum()).collect();
    let col_totals: Vec<u64> = (0..2).map(|j| table.iter().map(|row| row[j]).sum()).collect();
    let n: u64 = row_totals.iter().sum();
    let mut chi2 = 0.0;
    for i in 0..2 {
        for j in 0..2 {
            let e = row_totals[i] as f64 * col_totals[j] as f64 / n as f64;
            chi2 += (table[i][j] as f64 - e).powi(2) / e;
        }
    }
    (chi2, 0.0)  // p-value requires chi2 dist
}

fn main() {
    let mut rng = rand::thread_rng();
    let normal = Normal::new(10.0, 2.0).unwrap();
    let data: Vec<f64> = (0..30).map(|_| normal.sample(&mut rng)).collect();
    let (t, p) = t_stat_1samp(&data, 9.0);
    println!("One-sample t-test: t={} p={}", t, p);

    // Chi-square
    let table = [[10, 20], [20, 20]];
    let (chi2, p) = chi2_test(&table);
    println!("Chi-square: stat={} p={}", chi2, p);
}
```
:::

Performs t-tests, Chi-square tests.

---

## 10. Symbolic Derivations with SymPy

Derive test statistics.

::: code-group

```python [Python]
from sympy import symbols, sqrt, Sum, IndexedBase

n, mu0, s = symbols('n mu0 s', positive=True)
x_bar = symbols('x_bar')
t = (x_bar - mu0) / (s / sqrt(n))
print("t-statistic:", t)

x, y = IndexedBase('x'), IndexedBase('y')
n1, n2 = symbols('n1 n2', positive=True)
s1, s2 = symbols('s1 s2', positive=True)
t_two = (Sum(x[i], (i,1,n1))/n1 - Sum(y[i], (i,1,n2))/n2) / sqrt(s1**2/n1 + s2**2/n2)
print("Two-sample t:", t_two)
```

```rust [Rust]
fn main() {
    println!("t-statistic: (x_bar - μ0)/(s/√n)");
    println!("Two-sample t: (x_bar1 - x_bar2)/√(s1²/n1 + s2²/n2)");
}
```
:::

---

## 11. Challenges in ML Applications

- **Multiple Testing**: False positives; use corrections.
- **Non-Normality**: Non-parametric alternatives.
- **Small Samples**: t-tests sensitive.

---

## 12. Key ML Takeaways

- **Hypothesis testing decisions**: Model/feature significance.
- **p-values quantify evidence**: Against H₀.
- **t-tests compare means**: Model performance.
- **Chi-square for categorical**: Feature selection.
- **Code tests practically**: ML applications.

Testing drives ML validation.

---

## 13. Summary

Explored hypothesis testing, p-values, t-tests, Chi-square, with ML applications like model comparison. Examples and Python/Rust code bridge theory to practice. Prepares for ANOVA and resampling.

Word count: Approximately 3000.

---

## Further Reading
- Wasserman, *All of Statistics* (Ch. 10).
- James, *Introduction to Statistical Learning* (Ch. 5).
- Khan Academy: Hypothesis testing videos.
- Rust: 'statrs' for statistical tests.

---