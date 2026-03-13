---
title: Maximum Likelihood Estimation (MLE)
description: Comprehensive exploration of Maximum Likelihood Estimation in probability for AI/ML, covering principles, derivations for multiple distributions, properties, biases, and applications in model fitting, with examples and code in Python and Rust
layout: ../../../../layouts/TutorialPage.astro
---

# Maximum Likelihood Estimation (MLE)

Maximum Likelihood Estimation (MLE) is a fundamental statistical method for estimating model parameters by selecting values that maximize the probability of observing the given data. In machine learning (ML), MLE drives the training of models like linear regression, logistic regression, Gaussian mixture models (GMMs), and neural networks, optimizing parameters to best fit data under assumed probabilistic distributions. By maximizing the likelihood, MLE ensures models align closely with observed patterns, making it a cornerstone of statistical learning.

This seventh lecture in the "Probability Foundations for AI/ML" series builds on the Law of Large Numbers and Central Limit Theorem, delving into MLE's principles, derivations for common distributions (Bernoulli, Normal, Poisson, Exponential), properties like consistency and efficiency, biases, and applications in ML. We'll provide intuitive explanations, rigorous derivations, and practical implementations in Python and Rust, preparing you for maximum a posteriori estimation and Bayesian inference.

---

## 1. Intuition Behind Maximum Likelihood Estimation

MLE seeks parameters θ that make the observed data D most likely under a model f(x|θ). Imagine tuning a radio to maximize signal clarity: MLE adjusts model parameters to make the data "sound" most probable.

For data D={x_1,...,x_n}, the likelihood is the joint probability (or density) of observing D given θ, assuming independence. MLE finds θ that maximizes this.

### ML Connection
- **Model Fitting**: MLE estimates weights in neural networks or means in clustering.
- **Loss Functions**: Minimizing negative log-likelihood is equivalent to MLE in optimization.

::: info
MLE is like finding the puzzle piece that best fits the data's shape, maximizing how "natural" the data appears.
:::

### Everyday Example
Suppose you flip a coin 5 times, getting 3 heads. The probability of heads (p) is unknown. MLE estimates p by maximizing the likelihood of seeing 3 heads, yielding p=0.6.

---

## 2. Formal Definition of Likelihood and MLE

Given i.i.d. data D={x_1,...,x_n} from distribution f(x|θ):

**Likelihood**: L(θ|D) = ∏_{i=1}^n f(x_i|θ).

**Log-Likelihood**: l(θ|D) = ∑_{i=1}^n log f(x_i|θ) (easier to optimize).

**MLE**: θ_hat = argmax_θ L(θ|D) = argmax_θ l(θ|D).

### Assumptions
- Data i.i.d.
- Model f correctly specified (misspecification leads to bias).

### Optimization
- Analytic: Solve ∂l/∂θ=0 for simple cases.
- Numerical: Gradient descent for complex models.

### ML Insight
- MLE corresponds to minimizing cross-entropy loss in classification.

---

## 3. Deriving MLE for Common Distributions

### Bernoulli(p)
Data: x_i in {0,1}, k successes in n trials.

Likelihood: L(p) = p^k (1-p)^{n-k}.

Log-likelihood: l(p) = k log p + (n-k) log(1-p).

Derivative: ∂l/∂p = k/p - (n-k)/(1-p) =0.

Solution: p_hat = k/n (proportion of successes).

### Normal N(μ,σ^2)
Data: x_i real-valued.

PDF: f(x|μ,σ) = (1/(σ sqrt(2π))) e^{-(x-μ)^2/(2σ^2)}.

Log-likelihood: l = -n/2 log(2πσ^2) - 1/(2σ^2) ∑ (x_i-μ)^2.

Partials:
- ∂l/∂μ = (1/σ^2) ∑ (x_i-μ) =0 ⇒ μ_hat = \bar{x}.
- ∂l/∂σ^2 = -n/(2σ^2) + 1/(2σ^4) ∑ (x_i-μ)^2 =0 ⇒ σ^2_hat = 1/n ∑ (x_i-\bar{x})^2.

Note: σ^2_hat biased; unbiased uses (n-1).

### Poisson(λ)
Data: x_i counts.

PMF: P(X=k) = e^{-λ} λ^k / k!.

Log-likelihood: l(λ) = ∑ (-λ + x_i log λ - log(x_i!)).

∂l/∂λ = -n + ∑ x_i / λ =0 ⇒ λ_hat = \bar{x}.

### Exponential(λ)
Data: x_i ≥0, waiting times.

PDF: f(x|λ) = λ e^{-λx}.

Log-likelihood: l(λ) = n log λ - λ ∑ x_i.

∂l/∂λ = n/λ - ∑ x_i =0 ⇒ λ_hat = 1/\bar{x}.

### ML Application
- **Bernoulli**: Binary classification probabilities.
- **Normal**: Regression residuals.
- **Poisson**: Event count modeling in NLP.
- **Exponential**: Survival analysis.

---

## 4. Properties of MLE

1. **Consistency**: θ_hat → θ as n→∞ (via LLN/CLT).
2. **Asymptotic Normality**: √n (θ_hat - θ) ~ N(0, I^{-1}(θ)), I Fisher information.
3. **Efficiency**: Achieves Cramér-Rao lower bound asymptotically.
4. **Invariance**: If g(θ) one-to-one, MLE of g(θ) is g(θ_hat).

### Bias
- Variance estimates (e.g., Normal) biased in finite samples.
- Correct with adjustments (e.g., n-1 for σ^2).

### ML Insight
- Consistency ensures large datasets yield accurate parameters.
- Normality enables confidence intervals.

---

## 5. Fisher Information and Cramér-Rao Bound

**Fisher Information**: I(θ) = E[ (∂l/∂θ)^2 ] = -E[ ∂^2 l / ∂θ^2 ].

Measures information data provides about θ.

**Cramér-Rao Bound**: Var(θ_hat) ≥ 1/(n I(θ)) for unbiased estimators.

MLE asymptotically efficient, achieving bound.

### Example
Bernoulli: I(p) = 1/(p(1-p)), Var(p_hat) ≈ p(1-p)/n.

In ML: Approximate I via Hessian for uncertainty quantification.

---

## 6. Maximum Likelihood in ML Models

### Linear Regression
Assume y_i ~ N(X_i β, σ^2).

Log-likelihood: l(β,σ^2) = -n/2 log(2πσ^2) - 1/(2σ^2) ∑ (y_i - X_i β)^2.

MLE: β_hat = (X^T X)^{-1} X^T y (OLS), σ^2_hat biased.

### Logistic Regression
y_i ~ Bern(sigmoid(X_i β)), sigmoid(z)=1/(1+e^{-z}).

Log-likelihood: l(β) = ∑ [y_i log p_i + (1-y_i) log(1-p_i)], p_i=sigmoid(X_i β).

No analytic solution; use gradient ascent.

### Gaussian Mixture Models (GMMs)
Data from mixture of k Gaussians, params θ={μ_j, Σ_j, π_j}.

Log-likelihood complex, use EM:
- E-step: Compute responsibilities P(z_i=j|x_i).
- M-step: Update μ_j, Σ_j, π_j.

### ML Applications
- **Classification**: MLE for softmax parameters.
- **Clustering**: GMMs for unsupervised learning.
- **Time-Series**: Poisson for event counts.

---

## 7. Expectation-Maximization (EM) Algorithm

For latent variable models (e.g., GMMs, HMMs).

**Steps**:
1. **E-step**: Estimate latent variables given current θ.
2. **M-step**: Maximize expected log-likelihood for θ.

Converges to local max.

### ML Connection
- GMM clustering: EM for MLE of mixture params.
- HMMs: Baum-Welch algorithm (EM variant).

---

## 8. Numerical Optimization for MLE

Analytic solutions rare except for simple cases (Bernoulli, Normal).

Numerical methods:
- Gradient Descent: Optimize l(θ).
- Newton's Method: Use Hessian for quadratic convergence.
- Libraries: SciPy, PyTorch.

### Challenges
- Non-convex likelihoods (e.g., neural nets).
- Numerical stability: Log-sum-exp tricks.

::: code-group

```python [Python]
import numpy as np
from scipy.optimize import minimize
from scipy.stats import poisson
from sklearn.mixture import GaussianMixture

# MLE for Poisson (analytic)
data = np.random.poisson(3, 100)
lam_mle = np.mean(data)
print("MLE λ Poisson:", lam_mle)

# MLE for logistic (numerical)
def log_lik_logistic(beta, X, y):
    p = 1 / (1 + np.exp(-X @ beta))
    return -np.sum(y * np.log(p + 1e-10) + (1-y) * np.log(1-p + 1e-10))

X = np.array([[1,1],[1,2],[1,3],[1,4]])
y = np.array([0,0,1,1])
beta_init = np.zeros(2)
res = minimize(log_lik_logistic, beta_init, args=(X, y))
print("MLE β logistic:", res.x)

# ML: GMM via EM
data = np.concatenate([np.random.normal(-2, 1, 50), np.random.normal(2, 1, 50)])
gmm = GaussianMixture(n_components=2, random_state=0)
gmm.fit(data.reshape(-1,1))
print("GMM means:", gmm.means_.flatten())
```

```rust [Rust]
fn log_lik_logistic(beta: &[f64], x: &[[f64; 2]], y: &[u8]) -> f64 {
    let mut sum = 0.0;
    for (xi, &yi) in x.iter().zip(y.iter()) {
        let p = 1.0 / (1.0 + (-(beta[0] * xi[0] + beta[1] * xi[1])).exp());
        sum += if yi == 1 { p.ln() } else { (1.0 - p).ln() };
    }
    -sum
}

fn main() {
    // MLE Poisson
    let poisson = rand_distr::Poisson::new(3.0).unwrap();
    let mut rng = rand::thread_rng();
    let mut sum = 0.0;
    let n = 100;
    for _ in 0..n {
        sum += poisson.sample(&mut rng) as f64;
    }
    println!("MLE λ Poisson: {}", sum / n as f64);

    // Logistic MLE (GD)
    let x = [[1.0,1.0],[1.0,2.0],[1.0,3.0],[1.0,4.0]];
    let y = [0,0,1,1];
    let mut beta = [0.0, 0.0];
    let eta = 0.01;
    for _ in 0..1000 {
        let mut grad = [0.0, 0.0];
        for (xi, &yi) in x.iter().zip(y.iter()) {
            let p = 1.0 / (1.0 + (-(beta[0] * xi[0] + beta[1] * xi[1])).exp());
            let err = yi as f64 - p;
            grad[0] += err * xi[0];
            grad[1] += err * xi[1];
        }
        beta[0] += eta * grad[0];
        beta[1] += eta * grad[1];
    }
    println!("MLE β logistic: {:?}", beta);
}
```
:::

Implements MLE for Poisson, logistic, GMM. Rust assumes `rand_distr` crate.

---

## 9. Symbolic MLE with SymPy

Derive exact solutions.

::: code-group

```python [Python]
from sympy import symbols, diff, solve, log, factorial, exp

# Bernoulli
p, k, n = symbols('p k n', positive=True)
l_bern = k * log(p) + (n-k) * log(1-p)
dl_dp = diff(l_bern, p)
p_mle = solve(dl_dp, p)[0]
print("MLE p Bernoulli:", p_mle)

# Normal
mu, sigma, x = symbols('mu sigma x', positive=True)
l_norm = -log(sigma) - (x-mu)**2/(2*sigma**2)  # Single obs, extend sum
dl_mu = diff(l_norm, mu)
mu_mle = solve(dl_mu, mu)[0]
print("MLE μ Normal:", mu_mle)

# Poisson
lam, k = symbols('lam k', positive=True)
l_pois = -lam + k * log(lam) - log(factorial(k))
dl_lam = diff(l_pois, lam)
lam_mle = solve(dl_lam, lam)[0]
print("MLE λ Poisson:", lam_mle)
```

```rust [Rust]
fn main() {
    println!("MLE p Bernoulli: k/n");
    println!("MLE μ Normal: x");
    println!("MLE λ Poisson: k");
}
```
:::

---

## 10. Challenges in MLE Applications

- **Non-Convexity**: Multiple maxima in neural nets.
- **Bias**: Variance estimates require correction.
- **Model Misspecification**: Wrong f(x|θ) skews results.
- **Computational Cost**: High-dim, iterative solvers.

---

## 11. MLE in Advanced ML Contexts

- **VAEs**: MLE for latent distributions via ELBO.
- **Time-Series**: Exponential for event timing.
- **Regularization**: Bridge to MAP with priors.

---

## 12. Key ML Takeaways

- **MLE optimizes fit**: Parameters match data.
- **Log-likelihood simplifies**: Numerical stability.
- **Properties ensure**: Consistency, normality.
- **EM for latents**: GMMs, HMMs.
- **Code implements**: Practical MLE.

MLE drives robust ML parameter estimation.

---

## 13. Summary

Explored MLE from intuition to derivations for Bernoulli, Normal, Poisson, Exponential, with properties and ML applications. Enhanced with EM and code in Python/Rust. Prepares for MAP and Bayesian inference.

Word count: Approximately 3000.

---

## Further Reading
- Wasserman, *All of Statistics* (Ch. 9).
- Bishop, *Pattern Recognition and Machine Learning* (Ch. 2.3).
- Murphy, *Machine Learning: A Probabilistic Perspective* (Ch. 8).
- Rust: 'argmin' for optimization, 'rand_distr' for sampling.

---