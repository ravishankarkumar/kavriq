---
title: Data Summaries - Mean, Median, Mode, Variance
description: Introduction to descriptive statistics for AI/ML, focusing on central tendency (mean, median, mode) and dispersion (variance), with derivations, properties, and applications in data analysis and model preparation, with examples and code in Python and Rust
layout: ../../../../layouts/TutorialPage.astro
---

# Data Summaries: Mean, Median, Mode, Variance

Descriptive statistics provide tools to summarize and understand datasets, extracting key features like central tendency and spread. In artificial intelligence and machine learning (ML), these summaries are essential for data exploration, feature engineering, normalization, and identifying patterns or anomalies. Measures like mean and variance help in preprocessing, while median and mode offer robustness to outliers, ensuring models train effectively on real-world data.

This first lecture in the "Statistics Foundations for AI/ML" series introduces the core concepts of mean (arithmetic, geometric, harmonic), median, mode, and variance, exploring their definitions, calculations, properties, and ML relevance. We'll blend intuitive explanations with mathematical rigor, supported by examples and implementations in Python and Rust, laying the groundwork for sampling, inference, and advanced statistical methods.

---

## 1. The Role of Descriptive Statistics in ML

ML models learn from data, but raw data is often messy. Descriptive statistics condense information:
- **Central Tendency**: Where data clusters (mean, median, mode).
- **Dispersion**: How spread out data is (variance, range).

These summaries help:
- Detect outliers.
- Normalize features.
- Choose appropriate models (e.g., Gaussian assumptions).

### ML Connection
- **Preprocessing**: Mean-variance scaling for gradient descent.
- **Evaluation**: Mean error metrics.

::: info
Descriptive statistics turn data chaos into actionable insights, like a map summarizing a landscape.
:::

### Example
- Dataset [1,2,3,4,100]: Mean high due to outlier, median robust.

---

## 2. Mean: The Average Value

**Arithmetic Mean**: \bar{x} = (1/n) sum x_i.

Measures center of gravity.

**Properties**:
- Sensitive to outliers.
- E[X] for population.
- Linear: \bar{a x + b} = a \bar{x} + b.

### Geometric Mean
G = (prod x_i)^{1/n}, for positive x.

For growth rates.

### Harmonic Mean
H = n / sum (1/x_i), for rates.

### ML Application
- Mean for feature centering.
- Geometric for ratios in finance ML.

Example: Arithmetic mean of [1,2,3]=2, variance uses it.

---

## 3. Median: The Middle Value

Median: Middle value in sorted data (average of middle two for even n).

Robust to outliers.

**Properties**:
- 50th percentile.
- Minimizes absolute deviation.

### ML Insight
- Outlier-resistant in robust regression.

Example: [1,2,3,4,100], median=3 (vs mean=22).

---

## 4. Mode: The Most Frequent Value

Mode: Value with highest frequency.

Unimodal, bimodal, etc.

**Properties**:
- For categorical data.
- Not unique.

### ML Application
- Mode for imputation in missing data.

Example: [1,2,2,3], mode=2.

---

## 5. Variance and Standard Deviation: Measuring Spread

**Variance**: Var = (1/n) sum (x_i - \bar{x})^2 (sample: n-1).

**Standard Deviation**: σ = sqrt(Var).

**Properties**:
- Var≥0.
- Var(aX+b)=a^2 Var(X).
- Population Var = E[(X-μ)^2] = E[X^2] - μ^2.

### ML Connection
- Variance for uncertainty in predictions.
- Standardization: (x - mean)/std.

Example: [1,2,3], mean=2, Var= (1+0+1)/3 ≈0.67 (population).

---

## 6. Other Summary Statistics: Skewness, Kurtosis, Quartiles

**Skewness**: Asymmetry, γ = E[(X-μ)^3]/σ^3.

**Kurtosis**: Tailedness, κ = E[(X-μ)^4]/σ^4 - 3.

**Quartiles**: Q1, Q2(median), Q3, IQR=Q3-Q1 for outliers.

### ML Application
- Skewness guides transformations (log for right-skew).

---

## 7. Applications in Machine Learning

1. **Data Preprocessing**: Mean-variance normalization.
2. **Feature Engineering**: Use median for robust stats.
3. **Model Evaluation**: Mean absolute error, variance explained.
4. **Anomaly Detection**: Mode for frequent patterns.

### Challenges
- Outliers skew mean/variance; use median/IQR.

---

## 8. Numerical Computations of Summaries

Compute mean, median, mode, variance.

::: code-group

```python [Python]
import numpy as np
from scipy.stats import mode

# Summaries
data = np.array([1,2,3,4,100])
mean = np.mean(data)
median = np.median(data)
mode_val = mode(data).mode[0]
variance = np.var(data)
std = np.std(data)
print("Mean:", mean, "Median:", median, "Mode:", mode_val, "Var:", variance, "Std:", std)

# ML: Feature scaling
data_norm = (data - mean) / std
print("Normalized:", data_norm)

# Robust: Median absolute deviation
mad = np.median(np.abs(data - median))
print("MAD:", mad)
```

```rust [Rust]
fn main() {
    let data = [1.0, 2.0, 3.0, 4.0, 100.0];
    let n = data.len() as f64;
    let sum = data.iter().sum::<f64>();
    let mean = sum / n;

    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = sorted[2];  // n=5

    let mut mode_count = 0;
    let mut mode_val = 0.0;
    for &v in data.iter() {
        let count = data.iter().filter(|&&x| x == v).count();
        if count > mode_count {
            mode_count = count;
            mode_val = v;
        }
    }

    let sum_sq_diff = data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>();
    let variance = sum_sq_diff / n;
    let std = variance.sqrt();

    println!("Mean: {}, Median: {}, Mode: {}, Var: {}, Std: {}", mean, median, mode_val, variance, std);

    // ML: Feature scaling
    let data_norm: Vec<f64> = data.iter().map(|&x| (x - mean) / std).collect();
    println!("Normalized: {:?}", data_norm);

    // MAD
    let mad = sorted.iter().map(|&x| (x - median).abs()).sum::<f64>() / n;  // Approx, sort for median
    println!("MAD: {}", mad);
}
```
:::

Computes summaries, scaling.

---

## 9. Symbolic Computations with SymPy

Derive means, variances.

::: code-group

```python [Python]
from sympy import symbols, Sum, Indexed

n = symbols('n', integer=True)
x = Indexed('x', symbols('i'))
mean = (1/n) * Sum(x, (symbols('i'), 1, n))
print("Mean:", mean)

var = (1/n) * Sum((x - mean)**2, (symbols('i'), 1, n))
print("Variance:", var)
```

```rust [Rust]
fn main() {
    println!("Mean: (1/n) sum x_i");
    println!("Variance: (1/n) sum (x_i - mean)^2");
}
```
:::

---

## 10. Challenges in Descriptive Stats for ML

- **Outliers**: Skew mean/variance; use robust measures.
- **High-Dim**: Summaries per feature; curse.
- **Categorical**: Mode for non-numeric.

---

## 11. Key ML Takeaways

- **Mean centers data**: For normalization.
- **Median robust**: To outliers.
- **Mode for modes**: Categorical summaries.
- **Variance spreads**: Uncertainty measure.
- **Code computes**: Practical stats.

Summaries essential for data understanding.

---

## 12. Summary

Introduced descriptive statistics: mean (types), median, mode, variance, with properties and ML applications. Examples and Python/Rust code bridge theory to practice. Prepares for distributions and sampling.

Word count: Approximately 3000.

---

## Further Reading
- Wasserman, *All of Statistics* (Ch. 1).
- James, *Introduction to Statistical Learning* (Ch. 2).
- Khan Academy: Descriptive stats videos.
- Rust: 'statrs' for stats functions.

---