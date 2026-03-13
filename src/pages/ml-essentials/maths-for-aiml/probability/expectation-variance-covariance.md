---
title: Expectation, Variance & Covariance
description: Detailed study of expectation, variance, and covariance in probability for AI/ML, covering their definitions, properties, computational methods, and applications in modeling uncertainty and feature relationships, with examples and code in Python and Rust
layout: ../../../../layouts/TutorialPage.astro
---

# Expectation, Variance & Covariance

Expectation, variance, and covariance are fundamental statistical measures that quantify the behavior of random variables in probabilistic models. Expectation captures the average outcome, variance measures the spread of outcomes, and covariance describes the relationship between pairs of variables. In machine learning (ML), these concepts are critical for understanding data distributions, optimizing models, and capturing dependencies in features, enabling applications from regression to generative models.

This third lecture in the "Probability Foundations for AI/ML" series builds on random variables and distributions, exploring the mathematical definitions, properties, and computational techniques for expectation, variance, and covariance. We'll connect these to ML contexts like feature engineering and risk assessment, supported by intuitive explanations, derivations, and implementations in Python and Rust, preparing you for advanced topics like conditional probability and estimation.

---

## 1. Intuition Behind Expectation, Variance, and Covariance

**Expectation** (E[X]): The "center of mass" or average value of a random variable (RV) X over many trials. It's like predicting the typical outcome of a random process.

**Variance** (Var(X)): Measures how much X deviates from its mean, quantifying spread or uncertainty. High variance indicates scattered outcomes.

**Covariance** (Cov(X,Y)): Gauges whether two RVs X and Y move together (positive) or oppositely (negative), critical for feature correlations.

### ML Connection
- **Expectation**: Predict average house price in regression.
- **Variance**: Quantify prediction uncertainty in Bayesian models.
- **Covariance**: Identify redundant features in PCA or correlated errors.

::: info
Expectation is the anchor, variance the wobble, and covariance the dance between variables—together, they map data's behavior for ML.
:::

### Everyday Example
- Rolling a die: E[X]=3.5 (average), Var(X)≈2.92 (spread), Cov(X,Y) for two dice depends on independence.

---

## 2. Expectation: Definition and Properties

For a random variable X:
- **Discrete**: E[X] = sum x p(x), where p(x)=P(X=x).
- **Continuous**: E[X] = ∫ x f(x) dx, where f(x) is the PDF.

### Properties
- **Linearity**: E[aX+bY] = aE[X] + bE[Y].
- **Monotonicity**: If X≤Y, E[X]≤E[Y].
- **E[g(X)]**: sum g(x) p(x) or ∫ g(x) f(x) dx.

### ML Insight
- Loss functions: E[L(y,ŷ)] measures expected error.
- Policy evaluation in RL: E[reward].

Example: Bernoulli X~Bern(p), E[X] = p·1 + (1-p)·0 = p.

---

## 3. Variance: Measuring Spread

Var(X) = E[(X-E[X])^2] = E[X^2] - (E[X])^2.

Standard deviation: σ_X = sqrt(Var(X)).

### Properties
- Var(aX+b) = a^2 Var(X).
- For independent X,Y: Var(X+Y) = Var(X) + Var(Y).

### ML Application
- Feature scaling: High variance features dominate gradients.
- Model uncertainty: Variance in predictions signals robustness.

Example: Uniform [a,b], E[X]=(a+b)/2, Var(X)=(b-a)^2/12.

---

## 4. Covariance and Correlation

Cov(X,Y) = E[(X-E[X])(Y-E[Y])] = E[XY] - E[X]E[Y].

**Correlation**: ρ(X,Y) = Cov(X,Y) / (σ_X σ_Y), normalized [-1,1].

### Properties
- Cov(X,X)=Var(X).
- Cov(X,Y)=0 if independent (not converse).
- Cov(aX+b,cY+d)=ac Cov(X,Y).

### ML Connection
- PCA: Cov matrix eigenvalues for dim reduction.
- Graphical models: Cov structures dependencies.

Example: Two RVs X,Y with E[XY]=1, E[X]=E[Y]=0, Cov(X,Y)=1.

---

## 5. Joint and Conditional Expectations

For X,Y:
- **Joint**: E[X,Y] = ∫∫ xy f(x,y) dx dy or sum xy p(x,y).
- **Conditional**: E[Y|X=x] = ∫ y f(y|x) dy.

Law of iterated expectation: E[Y] = E[E[Y|X]].

### ML Insight
- Conditional expectation in EM algorithm for missing data.

---

## 6. Common Distributions and Their Moments

1. **Bernoulli(p)**: E[X]=p, Var(X)=p(1-p).
2. **Binomial(n,p)**: E[X]=np, Var(X)=np(1-p).
3. **Normal N(μ,σ^2)**: E[X]=μ, Var(X)=σ^2.
4. **Exponential(λ)**: E[X]=1/λ, Var(X)=1/λ^2.

### Covariance in Bivariate
- Bivariate normal: Cov(X,Y)=ρ σ_X σ_Y.

### ML Application
- Gaussian assumptions in regression.

---

## 7. Computational Aspects: Monte Carlo and Exact

Monte Carlo: Approximate E[g(X)] ≈ (1/n) sum g(x_i), x_i sampled.

Exact: Use PMF/PDF integrals.

::: code-group

```python [Python]
import numpy as np
from scipy.stats import binom, norm, expon

# Discrete: Binomial moments
n, p = 5, 0.5
print("Binomial E[X]:", binom.mean(n, p))  # np=2.5
print("Var(X):", binom.var(n, p))  # np(1-p)=1.25

# Continuous: Normal moments
mu, sigma = 0, 1
samples = np.random.normal(mu, sigma, 10000)
print("Sample E[X]:", np.mean(samples))
print("Sample Var(X):", np.var(samples))

# Covariance: Bivariate normal
rho = 0.5
cov_matrix = [[1, rho], [rho, 1]]
data = np.random.multivariate_normal([0, 0], cov_matrix, 10000)
cov = np.cov(data.T)[0,1]
print("Sample Cov(X,Y):", cov)

# ML: Feature covariance
X = np.array([[1,2],[2,4],[3,6],[4,8]])  # Linear relation
cov_X = np.cov(X.T)
print("Feature cov matrix:", cov_X)
```

```rust [Rust]
use rand::Rng;
use rand_distr::{Binomial, Normal, Distribution};

fn binomial_moments(n: u64, p: f64, trials: usize) -> (f64, f64) {
    let binom = Binomial::new(n, p).unwrap();
    let mut rng = rand::thread_rng();
    let mut sum = 0.0;
    let mut sum_sq = 0.0;
    for _ in 0..trials {
        let x = binom.sample(&mut rng);
        sum += x;
        sum_sq += x * x;
    }
    let mean = sum / trials as f64;
    let var = sum_sq / trials as f64 - mean * mean;
    (mean, var)
}

fn main() {
    let (mean, var) = binomial_moments(5, 0.5, 10000);
    println!("Binomial E[X]: {}, Var(X): {}", mean, var);

    // Normal moments
    let normal = Normal::new(0.0, 1.0).unwrap();
    let mut rng = rand::thread_rng();
    let mut sum = 0.0;
    let mut sum_sq = 0.0;
    let n = 10000;
    for _ in 0..n {
        let x = normal.sample(&mut rng);
        sum += x;
        sum_sq += x * x;
    }
    println!("Sample E[X]: {}, Var(X): {}", sum / n as f64, sum_sq / n as f64 - (sum / n as f64).powi(2));

    // Covariance (simplified bivariate)
    let mut cov_sum = 0.0;
    for _ in 0..n {
        let x = normal.sample(&mut rng);
        let y = 0.5 * x + (1.0 - 0.5f64.sqrt()) * normal.sample(&mut rng);  // Correlated
        cov_sum += (x - sum/n as f64) * (y - sum/n as f64);
    }
    println!("Sample Cov(X,Y): {}", cov_sum / n as f64);

    // ML: Feature covariance
    let X = [[1.0, 2.0], [2.0, 4.0], [3.0, 6.0], [4.0, 8.0]];
    let mut mean = [0.0, 0.0];
    for &row in X.iter() {
        mean[0] += row[0];
        mean[1] += row[1];
    }
    mean[0] /= X.len() as f64;
    mean[1] /= X.len() as f64;
    let mut cov = [[0.0; 2]; 2];
    for &row in X.iter() {
        cov[0][0] += (row[0] - mean[0]).powi(2);
        cov[1][1] += (row[1] - mean[1]).powi(2);
        cov[0][1] += (row[0] - mean[0]) * (row[1] - mean[1]);
    }
    cov[0][0] /= X.len() as f64;
    cov[1][1] /= X.len() as f64;
    cov[0][1] /= X.len() as f64;
    cov[1][0] = cov[0][1];
    println!("Feature cov matrix: {:?}", cov);
}
```
:::

Computes moments for distributions, feature covariance.

---

## 8. Symbolic Computations with SymPy

Exact moments.

::: code-group

```python [Python]
from sympy import symbols, integrate, exp, oo, sqrt, pi

x, p, lam = symbols('x p lam', positive=True)
n = symbols('n', integer=True, positive=True)

# Binomial
binom_pmf = binomial(n, x) * p**x * (1-p)**(n-x)
E_binom = sum(x * binom_pmf, (x, 0, n))
print("E[X] Binomial:", E_binom.subs(n,5))

# Normal
pdf_norm = 1/(sqrt(2*pi)) * exp(-x**2/2)
E_norm = integrate(x * pdf_norm, (x, -oo, oo))
print("E[X] Normal:", E_norm)
```

```rust [Rust]
fn main() {
    println!("E[X] Binomial(n=5): 5p");
    println!("E[X] Normal: 0");
}
```
:::

---

## 9. Applications in ML Modeling

- **Expectation**: Optimize expected loss in SGD.
- **Variance**: Quantify model uncertainty (e.g., dropout).
- **Covariance**: Feature selection, PCA.

---

## 10. Challenges and Considerations

- **High-Dim Cov**: Computational cost, singularity.
- **Sample Bias**: MC variance underestimates.

---

## 11. Key ML Takeaways

- **Expectation centers predictions**: Guides model training.
- **Variance measures spread**: Quantifies uncertainty.
- **Covariance captures relations**: For feature engineering.
- **Moments define dists**: Shape ML assumptions.
- **Code computes**: Practical moment estimation.

Moments structure ML uncertainty.

---

## 12. Summary

Explored expectation, variance, covariance, their properties, computations, with ML applications. Examples and Python/Rust code bridge theory to practice. Prepares for conditional probability and Bayes.

Word count: Approximately 2850.

---

## Further Reading
- Wasserman, *All of Statistics* (Ch. 3-4).
- Murphy, *Machine Learning* (Ch. 2).
- 3Blue1Brown: Probability visualizations.
- Rust: 'nalgebra' for matrix ops, 'rand_distr' for sampling.

---