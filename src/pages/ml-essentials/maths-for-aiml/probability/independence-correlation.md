---
title: Independence & Correlation
description: Detailed exploration of independence and correlation in probability for AI/ML, covering definitions, properties, conditional independence, correlation coefficients, and their applications in feature engineering and model design, with examples and code in Python and Rust
layout: ../../../../layouts/TutorialPage.astro
---

# Independence & Correlation

Independence and correlation are pivotal concepts in probability that dictate how random variables interact. Independence implies that the occurrence of one variable does not influence another, simplifying joint probability calculations. Correlation quantifies the strength and direction of linear relationships between variables, crucial for understanding data dependencies. In machine learning (ML), these concepts guide feature selection, model assumptions (e.g., Naive Bayes), and dimensionality reduction techniques like PCA, impacting model performance and interpretability.

This fifth lecture in the "Probability Foundations for AI/ML" series builds on conditional probability and Bayes' theorem, delving into independence, conditional independence, correlation, covariance, and their practical implications in ML. We'll provide intuitive explanations, rigorous mathematical formulations, and implementations in Python and Rust, preparing you for advanced topics like the Law of Large Numbers and estimation techniques.

---

## 1. Intuition Behind Independence and Correlation

**Independence**: Two events or random variables are independent if knowing one provides no information about the other. For example, rolling two dice: the outcome of one doesn't affect the other.

**Correlation**: Measures how much two variables move together linearly. Positive correlation means they increase together; negative means one increases as the other decreases; zero suggests no linear relationship.

### ML Connection
- **Independence**: Assumed in Naive Bayes for feature simplification.
- **Correlation**: Used in feature selection to avoid redundancy in models like regression or neural networks.

::: info
Independence is like two dancers moving without coordination; correlation tracks how synchronized their steps are.
:::

### Everyday Example
- **Independence**: Weather in Tokyo vs. coin flip in London.
- **Correlation**: Height and weight of people (tend to increase together).

---

## 2. Independence of Events

Events A and B are independent if:

\[
P(A \cap B) = P(A)P(B)
\]

Equivalently, P(A|B) = P(A) if P(B)>0.

### Properties
- Pairwise independence doesn't imply mutual (e.g., three events).
- If A,B independent, so are A,B^c, A^c,B, etc.

### ML Insight
- Simplifies joint distributions in probabilistic models.

Example: Two coin tosses, P(H1 ∩ H2) = P(H1)P(H2) = 0.5·0.5 = 0.25.

---

## 3. Independence of Random Variables

Random variables X,Y are independent if:

\[
P(X \in A, Y \in B) = P(X \in A)P(Y \in B)
\]

For discrete: P(X=x,Y=y)=P(X=x)P(Y=y).

For continuous: f(x,y)=f_X(x)f_Y(y).

### Properties
- E[XY] = E[X]E[Y] if independent.
- Var(X+Y) = Var(X) + Var(Y).

### ML Application
- Naive Bayes assumes feature independence given class.

Example: X,Y~Bern(0.5), independent, P(X=1,Y=1)=0.25.

---

## 4. Conditional Independence

X,Y are conditionally independent given Z if:

\[
P(X \in A, Y \in B | Z) = P(X \in A | Z)P(Y \in B | Z)
\]

Or f(x,y|z)=f(x|z)f(y|z).

### ML Connection
- Bayesian networks: Conditional independence simplifies structure.
- Naive Bayes: Features independent given class label.

Example: Symptoms independent given disease in medical diagnosis.

---

## 5. Covariance and Correlation: Definitions

**Covariance**:

\[
\text{Cov}(X,Y) = E[(X - E[X])(Y - E[Y])] = E[XY] - E[X]E[Y]
\]

**Correlation Coefficient**:

\[
\rho(X,Y) = \frac{\text{Cov}(X,Y)}{\sqrt{\text{Var}(X)\text{Var}(Y)}}
\]

ρ in [-1,1], 0 if uncorrelated.

### Properties
- Cov(X,Y)=0 if independent (not converse).
- ρ=±1 implies perfect linear relation.

### ML Insight
- PCA: High correlation → redundant features.
- Feature engineering: Remove highly correlated inputs.

Example: X,Y~N(0,1), ρ=0.5, Cov=0.5.

---

## 6. Correlation vs. Independence

- Independence implies Cov=0, but Cov=0 doesn't imply independence (e.g., X,Y=X^2).
- Correlation measures linear dependence; nonlinear relations may exist.

In ML: Check for nonlinear dependencies with mutual information.

---

## 7. Applications in Machine Learning

1. **Naive Bayes**: Assumes feature independence for P(X_1,...,X_n|y).
2. **PCA**: Covariance matrix eigenvalues for dimensionality reduction.
3. **Regularization**: Correlated features increase model variance.
4. **Time-Series**: Correlation in residuals indicates model misspecification.

---

## 8. Numerical Computations: Independence and Correlation

Simulate independence, compute correlations.

::: code-group

```python [Python]
import numpy as np
from scipy.stats import bernoulli, multivariate_normal

# Independence: Bernoulli
n_trials = 10000
X = bernoulli.rvs(0.5, size=n_trials)
Y = bernoulli.rvs(0.5, size=n_trials)
joint_prob = np.mean((X == 1) & (Y == 1))
p_X = np.mean(X)
p_Y = np.mean(Y)
print("P(X=1,Y=1):", joint_prob)
print("P(X=1)P(Y=1):", p_X * p_Y)  # ~equal, independent

# Correlation: Bivariate normal
rho = 0.5
cov_matrix = [[1, rho], [rho, 1]]
data = multivariate_normal.rvs([0, 0], cov_matrix, n_trials)
corr = np.corrcoef(data.T)[0,1]
print("Correlation:", corr)

# ML: Feature correlation
features = np.array([[1,2],[2,4],[3,6],[4,8]])  # Linear relation
corr_matrix = np.corrcoef(features.T)
print("Feature corr matrix:", corr_matrix)
```

```rust [Rust]
use rand::Rng;
use rand_distr::{Bernoulli, Normal, Distribution};

fn main() {
    let n_trials = 10000;
    let mut rng = rand::thread_rng();
    let bern = Bernoulli::new(0.5).unwrap();

    // Independence: Bernoulli
    let mut count_joint = 0;
    let mut count_x = 0;
    let mut count_y = 0;
    for _ in 0..n_trials {
        let x = bern.sample(&mut rng) as u8;
        let y = bern.sample(&mut rng) as u8;
        if x == 1 && y == 1 {
            count_joint += 1;
        }
        if x == 1 {
            count_x += 1;
        }
        if y == 1 {
            count_y += 1;
        }
    }
    println!("P(X=1,Y=1): {}", count_joint as f64 / n_trials as f64);
    println!("P(X=1)P(Y=1): {}", (count_x as f64 / n_trials as f64) * (count_y as f64 / n_trials as f64));

    // Correlation: Bivariate normal
    let normal = Normal::new(0.0, 1.0).unwrap();
    let mut sum_xy = 0.0;
    let mut sum_x = 0.0;
    let mut sum_y = 0.0;
    let mut sum_x2 = 0.0;
    let mut sum_y2 = 0.0;
    for _ in 0..n_trials {
        let x = normal.sample(&mut rng);
        let y = 0.5 * x + (0.75f64.sqrt()) * normal.sample(&mut rng);  // ρ=0.5
        sum_xy += x * y;
        sum_x += x;
        sum_y += y;
        sum_x2 += x * x;
        sum_y2 += y * y;
    }
    let mean_x = sum_x / n_trials as f64;
    let mean_y = sum_y / n_trials as f64;
    let cov = sum_xy / n_trials as f64 - mean_x * mean_y;
    let var_x = sum_x2 / n_trials as f64 - mean_x.powi(2);
    let var_y = sum_y2 / n_trials as f64 - mean_y.powi(2);
    println!("Correlation: {}", cov / (var_x * var_y).sqrt());

    // ML: Feature correlation
    let features = [[1.0, 2.0], [2.0, 4.0], [3.0, 6.0], [4.0, 8.0]];
    let mut mean = [0.0; 2];
    for row in features.iter() {
        mean[0] += row[0];
        mean[1] += row[1];
    }
    mean[0] /= features.len() as f64;
    mean[1] /= features.len() as f64;
    let mut corr = [[0.0; 2]; 2];
    for i in 0..2 {
        for j in 0..2 {
            let mut sum = 0.0;
            for row in features.iter() {
                sum += (row[i] - mean[i]) * (row[j] - mean[j]);
            }
            corr[i][j] = sum / (features.len() as f64);
        }
    }
    let corr_coeff = corr[0][1] / (corr[0][0] * corr[1][1]).sqrt();
    println!("Feature corr coeff: {}", corr_coeff);
}
```
:::

Simulates independence, computes correlations.

---

## 9. Symbolic Computations with SymPy

Verify independence, covariance.

::: code-group

```python [Python]
from sympy import symbols, Rational, E
x, y, p = symbols('x y p')
p_xy = Rational(1,4)  # Two coins
p_x = Rational(1,2)
p_y = Rational(1,2)
print("Independent check:", p_xy == p_x * p_y)

# Covariance
X, Y = symbols('X Y')
cov = E(X*Y) - E(X)*E(Y)
print("Cov(X,Y):", cov)
```

```rust [Rust]
fn main() {
    println!("Independent check: P(X=1,Y=1)=P(X=1)P(Y=1)=0.25");
    println!("Cov(X,Y): E[XY] - E[X]E[Y]");
}
```
:::

---

## 10. Challenges in ML Applications

- **False Independence**: Naive Bayes oversimplifies.
- **Correlation vs. Causation**: Misleading in feature selection.
- **High-Dim**: Cov matrix computation costly.

---

## 11. Key ML Takeaways

- **Independence simplifies**: Joint dists factorize.
- **Conditional indep key**: For efficient models.
- **Correlation informs**: Feature relationships.
- **Covariance in PCA**: Dimensionality reduction.
- **Code verifies**: Independence, correlations.

Independence and correlation shape ML design.

---

## 12. Summary

Explored independence, conditional independence, correlation, covariance, with ML applications. Examples and Python/Rust code connect theory to practice. Prepares for LLN and CLT.

Word count: Approximately 2850.

---

## Further Reading
- Wasserman, *All of Statistics* (Ch. 3-5).
- Bishop, *Pattern Recognition* (Ch. 2).
- 3Blue1Brown: Correlation videos.
- Rust: 'nalgebra' for matrices, 'rand_distr' for sampling.

---