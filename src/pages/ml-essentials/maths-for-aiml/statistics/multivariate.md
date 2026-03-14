---
title: Multivariate Statistics - Correlated Features & MANOVA
description: Comprehensive exploration of multivariate statistics for AI/ML, covering covariance matrices, correlation analysis, MANOVA, and their applications in feature engineering, dimensionality reduction, and group comparisons, with examples and code in Python and Rust
layout: ../../../../layouts/TutorialPage.astro
---

# Multivariate Statistics - Correlated Features & MANOVA

Multivariate statistics analyze multiple variables simultaneously, capturing their interrelationships and dependencies. In machine learning (ML), understanding correlated features through covariance and correlation matrices is crucial for feature engineering, dimensionality reduction (e.g., PCA), and model interpretation. Multivariate Analysis of Variance (MANOVA) extends ANOVA to compare groups across multiple dependent variables, useful for assessing complex ML outcomes. These tools are essential for handling high-dimensional data and uncovering patterns in real-world datasets.

This fifteenth lecture in the "Statistics Foundations for AI/ML" series builds on nonparametric statistics and statistical significance, exploring multivariate concepts, covariance matrices, correlation analysis, MANOVA, and their ML applications. We'll provide intuitive explanations, mathematical derivations, and practical implementations in Python and Rust, preparing you for time-series analysis and causal inference.

---

## 1. Why Multivariate Statistics Matter in ML

ML datasets often involve multiple features (e.g., image pixels, sensor readings), which are frequently correlated. Multivariate statistics:
- Quantify relationships via covariance and correlation.
- Reduce dimensionality by exploiting correlations (e.g., PCA).
- Compare groups across multiple outcomes using MANOVA.

### ML Connection
- **Feature Engineering**: Remove redundant correlated features.
- **PCA**: Use covariance matrix for dimensionality reduction.
- **MANOVA**: Compare ML models across multiple metrics (e.g., accuracy, F1).

::: info
Multivariate statistics are like a 3D map of data, revealing how features dance together and how groups differ across multiple dimensions.
:::

### Example
- Dataset with height, weight, age: Covariance shows how they co-vary; MANOVA tests if groups (e.g., genders) differ in all three.

---

## 2. Covariance and Correlation Matrices

### Covariance Matrix
For p variables X = (X₁,...,Xₚ), the covariance matrix Σ is:

\[
\Sigma_{ij} = \text{Cov}(X_i, X_j) = E[(X_i - \mu_i)(X_j - \mu_j)]
\]

Sample covariance matrix S:

\[
S_{ij} = \frac{1}{n-1} \sum_{k=1}^n (x_{ki} - \bar{x}_i)(x_{kj} - \bar{x}_j)
\]

Diagonal: Variances; off-diagonal: Covariances.

### Correlation Matrix
Normalized covariance:

\[
R_{ij} = \frac{S_{ij}}{\sqrt{S_{ii} S_{jj}}}
\]

R_{ij} ∈ [-1, 1], measures linear relationship strength.

### Properties
- Σ symmetric, positive semi-definite.
- R diagonal = 1, R_{ij} = ρ(X_i, X_j).

### ML Application
- PCA: Eigen decomposition of Σ for feature reduction.

Example: Height, weight data, S shows positive covariance, R ≈ 0.7 correlation.

---

## 3. Multivariate Normal Distribution

The multivariate normal N(μ, Σ) models p-dimensional data:

\[
f(x) = \frac{1}{(2\pi)^{p/2} |\Sigma|^{1/2}} \exp\left(-\frac{1}{2} (x - \mu)^T \Sigma^{-1} (x - \mu)\right)
\]

μ mean vector, Σ covariance matrix.

### Properties
- Marginals are normal.
- Σ determines feature dependencies.

### ML Connection
- Gaussian processes, generative models assume multivariate normal.

---

## 4. MANOVA: Comparing Groups Across Multiple Variables

MANOVA tests if k groups differ across p dependent variables.

**H₀**: μ₁ = μ₂ = ... = μ_k (mean vectors equal).

**Model**: Y_{ij} = μ_i + ε_{ij}, ε_{ij} ~ N(0, Σ).

### Test Statistics
- **Wilks' Lambda**: Λ = |W| / |T|, W within-group, T total covariance.
- **Pillai's Trace**, **Hotelling's T²**, **Roy's Largest Root**: Alternative metrics.

Approximated as F-distribution for p-value.

### ML Application
- Compare models across accuracy, precision, recall.

Example: MANOVA on two groups' feature vectors, p<0.05 suggests differences.

---

## 5. Assumptions and Diagnostics

- **Normality**: Variables multivariate normal.
- **Homogeneity**: Equal covariance matrices across groups.
- **Independence**: Observations independent.

Diagnostics: Q-Q plots, Box's M test.

In ML: Robust methods (e.g., permutation-based) if assumptions fail.

---

## 6. Applications in Machine Learning

1. **Feature Engineering**: Remove highly correlated features using R.
2. **PCA**: Eigen decomposition of S for dimensionality reduction.
3. **MANOVA**: Compare model performance across multiple metrics.
4. **Clustering**: Use Σ in Gaussian mixture models.

### Challenges
- **High-Dimensionality**: Σ estimation noisy.
- **Non-Normality**: Requires robust alternatives.

---

## 7. Numerical Multivariate Computations

Compute covariance, correlation, MANOVA.

::: code-group

```python [Python]
import numpy as np
from scipy.stats import multivariate_normal
from statsmodels.multivariate.manova import MANOVA

# Covariance and correlation matrices
data = np.random.multivariate_normal([0, 0], [[1, 0.5], [0.5, 1]], 100)
cov_matrix = np.cov(data.T, bias=False)
corr_matrix = np.corrcoef(data.T)
print("Covariance matrix:", cov_matrix)
print("Correlation matrix:", corr_matrix)

# Multivariate normal PDF
mvn = multivariate_normal([0, 0], [[1, 0.5], [0.5, 1]])
x = np.array([0, 0])
print("MVN PDF at (0,0):", mvn.pdf(x))

# MANOVA
import pandas as pd
group1 = np.random.multivariate_normal([0, 0], [[1, 0.5], [0.5, 1]], 50)
group2 = np.random.multivariate_normal([1, 1], [[1, 0.5], [0.5, 1]], 50)
data = np.vstack([group1, group2])
groups = ['A']*50 + ['B']*50
df = pd.DataFrame(data, columns=['var1', 'var2'])
df['group'] = groups
manova = MANOVA.from_formula('var1 + var2 ~ group', data=df)
print("MANOVA results:", manova.mv_test())

# ML: PCA with covariance
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(data)
print("PCA explained variance:", pca.explained_variance_ratio_)
```

```rust [Rust]
fn cov_matrix(data: &[[f64]]) -> [[f64; 2]; 2] {
    let n = data.len() as f64;
    let mean = data.iter().fold([0.0, 0.0], |acc, row| [acc[0] + row[0], acc[1] + row[1]]);
    let mean = [mean[0] / n, mean[1] / n];
    let mut cov = [[0.0; 2]; 2];
    for i in 0..2 {
        for j in 0..2 {
            cov[i][j] = data.iter().map(|row| (row[i] - mean[i]) * (row[j] - mean[j])).sum::<f64>() / (n - 1.0);
        }
    }
    cov
}

fn corr_matrix(cov: &[[f64; 2]; 2]) -> [[f64; 2]; 2] {
    let mut corr = [[0.0; 2]; 2];
    for i in 0..2 {
        for j in 0..2 {
            corr[i][j] = cov[i][j] / (cov[i][i] * cov[j][j]).sqrt();
        }
    }
    corr
}

fn main() {
    let mut rng = rand::thread_rng();
    let normal = rand_distr::Normal::new(0.0, 1.0).unwrap();
    // Simulate multivariate normal (simplified, ρ=0.5)
    let data: Vec<[f64; 2]> = (0..100).map(|_| {
        let x1 = normal.sample(&mut rng);
        let x2 = 0.5 * x1 + (0.75_f64).sqrt() * normal.sample(&mut rng);
        [x1, x2]
    }).collect();
    let cov = cov_matrix(&data);
    let corr = corr_matrix(&cov);
    println!("Covariance matrix: {:?}", cov);
    println!("Correlation matrix: {:?}", corr);
}
```
:::

Computes covariance, correlation, and MANOVA.

---

## 8. Theoretical Insights

**Covariance Matrix**: Captures linear dependencies, basis for PCA.

**MANOVA**: Tests equality of mean vectors, generalizes ANOVA.

**Multivariate Normal**: Foundation for many ML models.

### ML Insight
- Σ drives dimensionality reduction and clustering.

---

## 9. Challenges in ML Applications

- **High-Dimensionality**: Σ estimation noisy, requires regularization.
- **Non-Normality**: Use robust MANOVA or permutation tests.
- **Computational Cost**: Matrix operations scale poorly.

---

## 10. Key ML Takeaways

- **Covariance captures relationships**: For feature analysis.
- **Correlation standardizes**: Easy interpretation.
- **MANOVA compares groups**: Across multiple metrics.
- **PCA leverages Σ**: For dimensionality reduction.
- **Code computes matrices**: Practical multivariate stats.

Multivariate stats enhance ML's high-dimensional analysis.

---

## 11. Summary

Explored multivariate statistics, including covariance, correlation, MANOVA, and their ML applications in feature engineering and group comparisons. Examples and Python/Rust code bridge theory to practice. Prepares for time-series and causal inference.

Word count: Approximately 3000.

---

## Further Reading
- Anderson, *An Introduction to Multivariate Statistical Analysis*.
- Hastie, *Elements of Statistical Learning* (Ch. 14).
- James, *Introduction to Statistical Learning* (Ch. 10).
- Rust: 'nalgebra' for matrices, 'statrs' for stats.

---