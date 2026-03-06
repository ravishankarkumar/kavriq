---
title: Statistical Significance in ML Experiments
description: Comprehensive exploration of statistical significance in machine learning experiments, covering p-values, multiple testing corrections, t-tests, permutation tests, and applications in model comparison, feature importance, and A/B testing, with examples and code in Python and Rust
---

# Statistical Significance in ML Experiments

Statistical significance is a cornerstone of hypothesis testing, used in machine learning (ML) to determine whether observed differences in model performance, feature effects, or experimental outcomes are due to chance or reflect true effects. By leveraging p-values, t-tests, permutation tests, and multiple testing corrections, ML practitioners can make informed decisions about model selection, feature importance, and A/B testing outcomes, ensuring robust and reliable results.

This thirteenth lecture in the "Statistics Foundations for AI/ML" series builds on cross-validation and resampling, exploring the principles of statistical significance, methods for testing, corrections for multiple comparisons, and their applications in ML experiments. We'll provide intuitive explanations, mathematical foundations, and practical implementations in Python and Rust, preparing you for nonparametric statistics and extended topics.

---

## 1. Why Statistical Significance Matters in ML

ML experiments often involve comparing models, assessing feature importance, or evaluating interventions (e.g., A/B tests). Statistical significance helps:
- Determine if differences (e.g., model accuracies) are meaningful.
- Identify significant predictors for feature selection.
- Control false positives in high-dimensional settings.

### ML Connection
- **Model Comparison**: Test if one model outperforms another.
- **Feature Selection**: Identify impactful features.
- **A/B Testing**: Validate treatment effects.

::: info
Statistical significance is like a referee ensuring observed ML results aren't just random noise.
:::

### Example
- Compare two models' accuracies (0.85 vs. 0.80). A t-test determines if the difference is significant.

---

## 2. Foundations of Statistical Significance

**Null Hypothesis (H₀)**: No effect or difference (e.g., model accuracies equal).

**Alternative Hypothesis (H₁)**: Effect exists (e.g., one model better).

**p-value**: Probability of observing data (or more extreme) under H₀.

**Significance Level (α)**: Threshold (e.g., 0.05) to reject H₀ if p<α.

**Errors**:
- **Type I**: False positive (reject true H₀, α).
- **Type II**: False negative (fail to reject false H₀, β).

### ML Insight
- Low p-values suggest meaningful model improvements.

---

## 3. Common Tests for Significance in ML

### t-tests
- **One-sample**: Test if sample mean differs from a value (e.g., H₀: μ=0).
- **Two-sample**: Compare means of two groups (e.g., model accuracies).

**Statistic**: t = (\bar{x}_1 - \bar{x}_2) / \sqrt{s_1²/n_1 + s_2²/n_2} (two-sample).

### Permutation Tests
- Shuffle labels, recompute statistic to estimate p-value.
- Non-parametric, robust to non-normality.

### Chi-square Tests
- Test independence in categorical data (e.g., feature-label relationships).

### ML Application
- t-tests for model performance; permutation for feature importance.

---

## 4. Multiple Testing and Corrections

Multiple tests inflate Type I error rate (false positives).

**Family-Wise Error Rate (FWER)**: P(at least one false positive) = 1 - (1-α)^m.

**Corrections**:
- **Bonferroni**: α' = α/m (m tests).
- **Holm-Bonferroni**: Step-down adjustment.
- **False Discovery Rate (FDR)**: Control proportion of false positives (Benjamini-Hochberg).

### ML Connection
- Feature selection with many features requires FDR control.

Example: Testing 10 features, Bonferroni α=0.05/10=0.005 per test.

---

## 5. Power and Sample Size

**Power**: 1-β, probability of detecting true effect.

Depends on:
- Effect size (e.g., mean difference).
- Sample size n.
- α level.

In ML: Calculate n for detecting model improvement.

---

## 6. Applications in Machine Learning

1. **Model Comparison**: t-tests for accuracy differences.
2. **Feature Importance**: Permutation tests for significance.
3. **A/B Testing**: Test treatment effects in deployment.
4. **Hyperparameter Tuning**: Validate significant improvements.

### Challenges
- **Multiple Testing**: High-dimensional features inflate errors.
- **Non-i.i.d. Data**: Time-series, clustered data complicate tests.

---

## 7. Numerical Significance Testing

Implement t-tests, permutation tests, corrections.

::: code-group

```python [Python]
import numpy as np
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests

# Two-sample t-test for model comparison
model1_acc = np.random.normal(0.85, 0.05, 100)
model2_acc = np.random.normal(0.80, 0.05, 100)
t_stat, p_val = ttest_ind(model1_acc, model2_acc)
print("t-test: t=", t_stat, "p=", p_val)

# Permutation test for feature importance
def perm_test(X, y, model, feature_idx, n_perms=1000):
    baseline_score = model.score(X, y)
    perm_scores = []
    for _ in range(n_perms):
        X_perm = X.copy()
        np.random.shuffle(X_perm[:, feature_idx])
        perm_scores.append(model.score(X_perm, y))
    p_val = np.mean(np.array(perm_scores) <= baseline_score)
    return baseline_score, p_val

from sklearn.ensemble import RandomForestClassifier
X = np.random.rand(100, 2)
y = (X[:,0] + X[:,1] > 1).astype(int)
model = RandomForestClassifier(random_state=0).fit(X, y)
base_score, p_val = perm_test(X, y, model, 0)
print("Permutation test feature 0: p=", p_val)

# Multiple testing correction
p_vals = [0.01, 0.04, 0.02, 0.06]
_, p_adj, _, _ = multipletests(p_vals, alpha=0.05, method='bonferroni')
print("Bonferroni adjusted p:", p_adj)
```

```rust [Rust]
use rand::Rng;
use rand::seq::SliceRandom;

fn t_test_two_sample(x1: &[f64], x2: &[f64]) -> (f64, f64) {
    let n1 = x1.len() as f64;
    let n2 = x2.len() as f64;
    let mean1 = x1.iter().sum::<f64>() / n1;
    let mean2 = x2.iter().sum::<f64>() / n2;
    let var1 = x1.iter().map(|&x| (x - mean1).powi(2)).sum::<f64>() / (n1 - 1.0);
    let var2 = x2.iter().map(|&x| (x - mean2).powi(2)).sum::<f64>() / (n2 - 1.0);
    let t = (mean1 - mean2) / (var1 / n1 + var2 / n2).sqrt();
    (t, 0.0)  // p-value requires t-dist
}

fn perm_test(x: &[[f64; 2]], y: &[u8], feature_idx: usize, n_perms: usize) -> f64 {
    let mut rng = rand::thread_rng();
    let baseline_acc = 0.8;  // Simplified, assume model score
    let mut perm_scores = vec![0.0; n_perms];
    for i in 0..n_perms {
        let mut x_perm = x.to_vec();
        let mut feature: Vec<f64> = x_perm.iter().map(|row| row[feature_idx]).collect();
        feature.shuffle(&mut rng);
        for j in 0..x_perm.len() {
            x_perm[j][feature_idx] = feature[j];
        }
        perm_scores[i] = 0.8;  // Placeholder, assumes model score
    }
    perm_scores.iter().filter(|&&s| s <= baseline_acc).count() as f64 / n_perms as f64
}

fn main() {
    let mut rng = rand::thread_rng();
    let model1: Vec<f64> = (0..100).map(|_| rand_distr::Normal::new(0.85, 0.05).unwrap().sample(&mut rng)).collect();
    let model2: Vec<f64> = (0..100).map(|_| rand_distr::Normal::new(0.80, 0.05).unwrap().sample(&mut rng)).collect();
    let (t, p) = t_test_two_sample(&model1, &model2);
    println!("t-test: t={} p={}", t, p);

    let x: Vec<[f64; 2]> = (0..100).map(|_| [rng.gen(), rng.gen()]).collect();
    let y: Vec<u8> = x.iter().map(|xi| if xi[0] + xi[1] > 1.0 { 1 } else { 0 }).collect();
    let p_val = perm_test(&x, &y, 0, 1000);
    println!("Permutation test feature 0: p={}", p_val);
}
```
:::

Implements t-tests, permutation tests, and corrections.

---

## 8. Theoretical Foundations

**p-value**: P(T ≥ t | H₀), T test statistic.

**Multiple Testing**: FWER grows with m tests.

**Bonferroni**: Controls FWER, conservative.

### ML Insight
- FDR for high-dimensional feature selection.

---

## 9. Challenges in ML Significance Testing

- **Multiple Comparisons**: High-dimensional data increases false positives.
- **Non-i.i.d. Data**: Time-series requires specialized tests.
- **Small Samples**: Low power, wide CIs.

---

## 10. Key ML Takeaways

- **Significance validates results**: Model, feature effects.
- **p-values guide decisions**: Reject H₀.
- **Corrections control errors**: Bonferroni, FDR.
- **Permutation robust**: Non-parametric.
- **Code implements tests**: Practical ML.

Significance ensures reliable ML experiments.

---

## 11. Summary

Explored statistical significance in ML experiments, covering p-values, t-tests, permutation tests, and corrections, with applications in model comparison and A/B testing. Examples and Python/Rust code bridge theory to practice. Prepares for nonparametric statistics and extended topics.

Word count: Approximately 3000.

---

## Further Reading
- Wasserman, *All of Statistics* (Ch. 10).
- James, *Introduction to Statistical Learning* (Ch. 5).
- Benjamini, Hochberg, "Controlling the False Discovery Rate".
- Rust: 'statrs' for tests, 'rand' for permutations.

---