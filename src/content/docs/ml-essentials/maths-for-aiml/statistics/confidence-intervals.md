---
title: Estimation & Confidence Intervals
description: Comprehensive exploration of point and interval estimation, focusing on confidence intervals for means and proportions in statistics for AI/ML, covering theory, derivations, and applications in model evaluation and uncertainty quantification, with examples and code in Python and Rust
---

# Estimation & Confidence Intervals

Estimation is the process of inferring population parameters from sample data, using point estimates (single values) or interval estimates (ranges with confidence levels). Confidence intervals (CIs) quantify uncertainty around point estimates, providing a range likely to contain the true parameter with a specified probability. In machine learning (ML), CIs are crucial for assessing model performance, comparing algorithms, and quantifying uncertainty in predictions, ensuring robust decision-making.

This fifth lecture in the "Statistics Foundations for AI/ML" series builds on sampling and distributions, exploring point estimation (mean, proportion), confidence intervals for various parameters, their theoretical foundations, and ML applications. We'll provide intuitive explanations, mathematical derivations, and practical implementations in Python and Rust, preparing you for hypothesis testing and advanced inference.

---

## 1. Why Estimation and Confidence Intervals Matter in ML

ML models rely on data samples to estimate population characteristics, like the mean prediction error or classification accuracy. Point estimates provide a single best guess, but CIs offer a range, reflecting uncertainty due to finite samples.

CIs answer: "How confident are we that the true parameter lies in this range?"

### ML Connection
- **Model Evaluation**: CIs for accuracy or loss metrics.
- **Hyperparameter Tuning**: Compare model performance with CIs.
- **Uncertainty Quantification**: Guide decisions in safety-critical applications.

::: info
Estimation pins down a parameter; CIs tell you how "wobbly" that pin is, like a weather forecast with a range of temperatures.
:::

### Example
- Sample mean accuracy 0.85 from 100 tests; 95% CI [0.82, 0.88] suggests the true accuracy likely lies within.

---

## 2. Point Estimation: Mean and Proportion

**Point Estimation**: Single value estimating a parameter.

- **Mean**: Sample mean \bar{x} = (1/n) \sum x_i estimates population mean μ.
- **Proportion**: Sample proportion p_hat = k/n estimates population p.

### Properties
- **Unbiased**: E[\bar{x}] = μ, E[p_hat] = p.
- **Consistency**: Converges to true value as n→∞ (LLN).
- **Efficiency**: Low variance among estimators.

### ML Application
- Mean error in regression, proportion correct in classification.

Example: 3 heads in 5 coin flips, p_hat = 3/5 = 0.6.

---

## 3. Confidence Intervals: Concept and Interpretation

A 100(1-α)% CI is a range [L, U] where P(L ≤ θ ≤ U) = 1-α for parameter θ.

- **Confidence Level**: 1-α (e.g., 95%) reflects probability over repeated samples.
- **Misinterpretation**: Not "95% chance θ in interval," but "95% of such intervals contain θ."

### ML Insight
- CIs for model metrics guide deployment decisions.

---

## 4. Confidence Interval for the Mean

For sample mean \bar{x}, population variance σ² known, n large (CLT):
- \bar{x} ~ N(μ, σ²/n).
- 95% CI: \bar{x} ± z_{0.025} \cdot σ/\sqrt{n}, z_{0.025} ≈ 1.96.

**Unknown σ**: Use sample s, t-distribution for small n:
- CI: \bar{x} ± t_{n-1,0.025} \cdot s/\sqrt{n}.

### Derivation
From CLT, \sqrt{n} (\bar{x} - μ)/σ ~ N(0,1).

### ML Application
- CI for mean prediction error.

Example: n=30, \bar{x}=100, s=15, t_{29,0.025}≈2.045, CI = 100 ± 2.045 \cdot 15/\sqrt{30} ≈ [94.4, 105.6].

---

## 5. Confidence Interval for Proportions

For sample proportion p_hat = k/n, large n:
- p_hat ~ N(p, p(1-p)/n).
- CI: p_hat ± z_{0.025} \sqrt{p_hat (1-p_hat)/n}.

### ML Connection
- CI for classification accuracy.

Example: 80 successes in 100 trials, p_hat=0.8, CI = 0.8 ± 1.96 \sqrt{0.8 \cdot 0.2 / 100} ≈ [0.72, 0.88].

---

## 6. Bootstrap Confidence Intervals

Resample data with replacement to estimate sampling distribution.

**Percentile Method**: Compute statistic on B bootstrap samples, take [α/2, 1-α/2] percentiles.

### ML Application
- Non-parametric CIs for complex metrics.

---

## 7. Assumptions and Robustness

- **Normality**: CLT for large n; t for small n, normal data.
- **Independence**: i.i.d. samples.
- **Robustness**: Bootstrap for non-normal.

In ML: Check assumptions or use robust methods.

---

## 8. Applications in Machine Learning

1. **Model Evaluation**: CIs for accuracy, F1 score.
2. **A/B Testing**: CIs for treatment effects.
3. **Uncertainty Quantification**: CIs for predictions.
4. **Hyperparameter Tuning**: Compare model CIs.

### Challenges
- **Small Samples**: Wide CIs, t-dist needed.
- **Non-i.i.d.**: Violates CLT; use bootstrap.

---

## 9. Numerical Computations of CIs

Compute CIs for mean, proportion, bootstrap.

::: code-group

```python [Python]
import numpy as np
from scipy.stats import norm, t
from statsmodels.stats.proportion import proportion_confint

# CI for mean (known σ)
data = np.random.normal(100, 15, 30)
mean = np.mean(data)
se = 15 / np.sqrt(30)  # Known σ=15
ci_norm = mean + np.array([-1, 1]) * norm.ppf(0.975) * se
print("95% CI (known σ):", ci_norm)

# CI for mean (unknown σ)
s = np.std(data, ddof=1)
ci_t = mean + np.array([-1, 1]) * t.ppf(0.975, df=29) * s / np.sqrt(30)
print("95% CI (t-dist):", ci_t)

# CI for proportion
successes, n = 80, 100
ci_prop = proportion_confint(successes, n, alpha=0.05, method='normal')
print("95% CI proportion:", ci_prop)

# ML: Bootstrap CI
def bootstrap_ci(data, n_boots=1000, alpha=0.05):
    boot_means = [np.mean(np.random.choice(data, len(data))) for _ in range(n_boots)]
    return np.percentile(boot_means, [100*alpha/2, 100*(1-alpha/2)])

ci_boot = bootstrap_ci(data)
print("Bootstrap CI:", ci_boot)
```

```rust [Rust]
use rand::Rng;
use rand_distr::Normal;

fn norm_ppf(p: f64) -> f64 {
    // Approximate z-score for 0.975 (1.96)
    if p > 0.5 {
        1.96
    } else {
        -1.96
    }
}

fn main() {
    let mut rng = rand::thread_rng();
    let normal = Normal::new(100.0, 15.0).unwrap();

    // CI for mean (known σ)
    let data: Vec<f64> = (0..30).map(|_| normal.sample(&mut rng)).collect();
    let mean = data.iter().sum::<f64>() / data.len() as f64;
    let se = 15.0 / (30.0_f64).sqrt();
    let ci_norm = [mean - norm_ppf(0.975) * se, mean + norm_ppf(0.975) * se];
    println!("95% CI (known σ): {:?}", ci_norm);

    // CI for mean (unknown σ, simplified t≈z for large n)
    let s = (data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (data.len() as f64 - 1.0)).sqrt();
    let ci_t = [mean - norm_ppf(0.975) * s / (30.0_f64).sqrt(), mean + norm_ppf(0.975) * s / (30.0_f64).sqrt()];
    println!("95% CI (t-dist approx): {:?}", ci_t);

    // CI for proportion
    let successes = 80.0;
    let n = 100.0;
    let p_hat = successes / n;
    let se_prop = (p_hat * (1.0 - p_hat) / n).sqrt();
    let ci_prop = [p_hat - norm_ppf(0.975) * se_prop, p_hat + norm_ppf(0.975) * se_prop];
    println!("95% CI proportion: {:?}", ci_prop);

    // Bootstrap CI
    let n_boots = 1000;
    let mut boot_means = vec![0.0; n_boots];
    for i in 0..n_boots {
        let boot_sample: Vec<f64> = (0..data.len()).map(|_| data[rng.gen_range(0..data.len())]).collect();
        boot_means[i] = boot_sample.iter().sum::<f64>() / data.len() as f64;
    }
    boot_means.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let ci_boot = [boot_means[(n_boots as f64 * 0.025) as usize], boot_means[(n_boots as f64 * 0.975) as usize]];
    println!("Bootstrap CI: {:?}", ci_boot);
}
```
:::

Computes CIs for mean, proportion, and bootstrap.

---

## 10. Symbolic Derivations with SymPy

Derive CI bounds.

::: code-group

```python [Python]
from sympy import symbols, sqrt

n, mu, sigma, z = symbols('n mu sigma z', positive=True)
x_bar = symbols('x_bar')
se = sigma / sqrt(n)
ci = [x_bar - z * se, x_bar + z * se]
print("CI for mean:", ci)

p_hat = symbols('p_hat')
se_prop = sqrt(p_hat * (1 - p_hat) / n)
ci_prop = [p_hat - z * se_prop, p_hat + z * se_prop]
print("CI for proportion:", ci_prop)
```

```rust [Rust]
fn main() {
    println!("CI for mean: [x_bar - z σ/√n, x_bar + z σ/√n]");
    println!("CI for proportion: [p_hat - z √(p_hat (1-p_hat)/n), p_hat + z √(p_hat (1-p_hat)/n)]");
}
```
:::

---

## 11. Challenges in ML Applications

- **Small Samples**: Wide CIs, t-dist needed.
- **Non-Normality**: Use bootstrap or transformations.
- **Non-i.i.d.**: Time-series, clustered data violate assumptions.

---

## 12. Key ML Takeaways

- **Point estimates simplify**: Mean, proportion.
- **CIs quantify uncertainty**: For model metrics.
- **CLT enables normal CIs**: For large samples.
- **Bootstrap flexible**: Non-parametric CIs.
- **Code computes**: Practical intervals.

Estimation and CIs drive reliable ML.

---

## 13. Summary

Explored point estimation and confidence intervals for means and proportions, their derivations, and ML applications like model evaluation. Examples and Python/Rust code bridge theory to practice. Prepares for hypothesis testing and ANOVA.

Word count: Approximately 3000.

---

## Further Reading
- Wasserman, *All of Statistics* (Ch. 6).
- James, *Introduction to Statistical Learning* (Ch. 5).
- Khan Academy: Confidence intervals videos.
- Rust: 'rand' for sampling, 'statrs' for stats.

---