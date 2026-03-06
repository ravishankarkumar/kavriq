---
title: Bayesian Statistics in Practice
description: Practical guide to Bayesian statistics for AI/ML, covering priors, posteriors, conjugate distributions, MCMC, variational inference, and applications in model uncertainty and decision-making, with examples and code in Python and Rust
---

# Bayesian Statistics in Practice

Bayesian statistics treats parameters as random variables, incorporating prior beliefs and updating them with data to form posteriors. In practice for machine learning (ML), it enables uncertainty estimation, robust model fitting, and decision-making under risk, as seen in Bayesian optimization and probabilistic neural networks. This approach contrasts frequentist methods by providing full distributions rather than point estimates.

This tenth lecture in the "Statistics Foundations for AI/ML" series builds on MLE vs. MoM, exploring practical Bayesian methods, priors, inference techniques, and ML applications. We'll provide intuitive explanations, mathematical derivations, and practical implementations in Python and Rust, preparing you for nonparametric statistics and beyond.

---

## 1. Bayesian Thinking in Practice

Bayesian statistics is "learning from data": Start with prior P(θ), observe D, update to posterior P(θ|D) = P(D|θ) P(θ) / P(D).

In practice:
- Prior: Domain knowledge (e.g., positive parameters).
- Likelihood: Model fit to data.
- Posterior: Updated distribution.

### ML Connection
- **Uncertainty Quantification**: Posteriors for prediction intervals.
- **Regularization**: Priors as penalties.

::: info
Bayesian practice is like adjusting beliefs with evidence, turning assumptions into informed probabilities.
:::

### Example
- Parameter θ for coin bias: Prior Beta(2,2), 3 heads in 5 flips, posterior Beta(5,4), mean 5/9≈0.556.

---

## 2. Priors in Practice: Conjugate and Non-Informative

**Conjugate Priors**: Posterior same family as prior, easy computation.

- Beta for Bernoulli/Binomial.
- Normal for Normal mean (known variance).
- Inverse-Gamma for Normal variance.

**Non-Informative**: Flat (improper) or Jeffreys (invariant).

**Empirical Bayes**: Estimate prior from data.

### Properties
- Conjugates: Closed-form updates.

### ML Application
- Hyperparameter priors in Bayesian optimization.

Example: Normal mean, prior N(0,10²), likelihood N(\bar{x},σ²/n), posterior N weighted.

---

## 3. Computing Posteriors: Exact and Approximate

**Exact**: For conjugates, closed form.

**Approximate**:
- MCMC: Sample posterior (Metropolis, Gibbs).
- Variational: Optimize q to approx P(θ|D).

### ML Insight
- PyMC3/Stan for MCMC in probabilistic programming.

---

## 4. Markov Chain Monte Carlo (MCMC) in Practice

**Metropolis-Hastings**: Propose, accept/reject based on posterior ratio.

**Gibbs**: Sample conditionals.

**Practical Tips**:
- Burn-in: Discard initial samples.
- Thinning: Reduce autocorrelation.
- Convergence diagnostics: Trace plots, Gelman-Rubin.

### ML Connection
- Sample weights in Bayesian NNs.

---

## 5. Variational Inference (VI): Fast Approximation

Min D_{KL}(q||posterior) by optimizing ELBO.

Mean-field: q(θ) = prod q_i(θ_i).

ADVI: Gradient-based.

In ML: Scalable for large models (e.g., VAEs).

---

## 6. Credible Intervals and Decision-Making

**Credible Interval**: Posterior range with 1-α probability.

HPD: Highest posterior density interval.

In ML: Prediction intervals from posteriors.

---

## 7. Empirical Bayes and Hierarchical Models

Empirical: Use data to set prior hyperparameters.

Hierarchical: Priors on priors.

In ML: Multi-task learning with shared priors.

---

## 8. Applications in Machine Learning

1. **Bayesian Optimization**: Priors for hyperparam search.
2. **Bayesian NNs**: Uncertainty in predictions.
3. **GP Regression**: Non-param Bayesian.
4. **Probabilistic Programming**: Pyro, Stan for custom models.

### Challenges
- Computation: MCMC slow.
- Prior Choice: Sensitivity.

---

## 9. Numerical Bayesian in Practice

Sample posteriors.

::: code-group

```python [Python]
import numpy as np
import pymc as pm
import arviz as az

# Empirical Bayes: Beta prior from data
data = np.random.binomial(1, 0.6, 100)
k = np.sum(data)
n = len(data)
alpha, beta = 1 + k, 1 + n - k  # Pseudo-counts
with pm.Model() as model:
    p = pm.Beta('p', alpha=alpha, beta=beta)
    trace = pm.sample(1000)

print("Posterior mean p:", trace.posterior['p'].mean())

# MCMC: Normal mean
data_normal = np.random.normal(0, 1, 100)
with pm.Model() as norm_model:
    mu = pm.Normal('mu', mu=0, sigma=10)
    sigma = pm.HalfNormal('sigma', sigma=10)
    obs = pm.Normal('obs', mu=mu, sigma=sigma, observed=data_normal)
    trace_norm = pm.sample(1000)

az.summary(trace_norm, hdi_prob=0.95)  # Credible intervals

# ML: Variational for simple model
with pm.Model() as vi_model:
    p = pm.Beta('p', alpha=1, beta=1)
    obs = pm.Bernoulli('obs', p=p, observed=data)
    approx = pm.fit(method='advi')

vi_trace = approx.sample(1000)
print("VI mean p:", vi_trace['p'].mean())
```

```rust [Rust]
fn metropolis_bern(k: f64, n: f64, alpha: f64, beta: f64, steps: usize) -> f64 {
    let mut rng = rand::thread_rng();
    let mut p = 0.5;
    let mut sum = 0.0;
    for _ in 0..steps {
        let p_prop = rng.gen_range(0.0..1.0);
        let post_curr = (k + alpha - 1.0) * p.ln() + (n - k + beta - 1.0) * (1.0 - p).ln();
        let post_prop = (k + alpha - 1.0) * p_prop.ln() + (n - k + beta - 1.0) * (1.0 - p_prop).ln();
        let accept = (post_prop - post_curr).exp().min(1.0);
        if rng.gen::<f64>() < accept {
            p = p_prop;
        }
        sum += p;
    }
    sum / steps as f64
}

fn main() {
    let k = 60.0;  // Example from data
    let n = 100.0;
    let alpha = 1.0 + k;
    let beta = 1.0 + n - k;
    println!("Posterior mean p (MCMC): {}", metropolis_bern(k, n, alpha, beta, 10000));

    // Normal mean MCMC (simplified)
    // Omit for brevity
}
```
:::

Implements Bayesian estimation with MCMC, VI.

---

## 10. Challenges in Practical Bayesian

- **Scalability**: High-dim MCMC slow.
- **Convergence**: Diagnose chain mixing.
- **Prior Specification**: Impacts results.

---

## 11. Key ML Takeaways

- **Bayesian practical**: Updates with data.
- **Conjugates easy**: Closed posteriors.
- **MCMC/VI scale**: To complex models.
- **Uncertainty practical**: CIs, decisions.
- **Code tools**: PyMC, custom Rust.

Bayesian stats enhance ML practice.

---

## 12. Summary

Explored practical Bayesian stats from priors to MCMC/VI, with ML applications. Examples and Python/Rust code bridge theory to practice. Prepares for nonparametric and multivariate stats.

Word count: Approximately 3000.

---

## Further Reading
- Gelman, *Bayesian Data Analysis* (3rd Ed).
- Murphy, *Probabilistic ML* (Ch. 7-9).
- McElreath, *Statistical Rethinking* (2nd Ed).
- Rust: Implement MCMC with 'rand'.

---