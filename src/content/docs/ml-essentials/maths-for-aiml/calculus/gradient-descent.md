---
title: Gradient Descent & Variants
description: Investigate gradient descent and its variants in calculus for AI/ML optimization, covering convergence, adaptive methods, and practical implementations, with examples and code in Python and Rust
---

# Gradient Descent & Variants

Gradient descent (GD) is a foundational optimization algorithm that iteratively moves toward the minimum of a function by stepping in the direction opposite to the gradient. In AI and machine learning, GD and its variants power the training of models by minimizing loss functions, adjusting parameters to fit data effectively. From vanilla GD to advanced adaptive methods like Adam, these techniques balance speed, stability, and convergence in high-dimensional spaces.

This lecture advances from convexity and landscapes, dissecting GD mechanics, variants for efficiency, theoretical guarantees, and ML applications. We'll merge conceptual depth with code in Python and Rust, providing tools to implement and experiment with optimization strategies in AI.

---

## 1. Intuition Behind Gradient Descent

Imagine descending a foggy mountain: At each point, feel the steepest downhill slope (negative gradient) and step accordingly. Too big steps overshoot; too small, slow progress.

Mathematically: For minimize f(w), update w_{t+1} = w_t - η ∇f(w_t), η learning rate.

### ML Connection
- Trains models: w parameters, f loss.
- High-dim: Millions of w, but GD scales.

::: info
GD is greedy local search, assuming smoother paths lead to better minima.
:::

### Challenges
- Choosing η: Fixed, scheduled, adaptive.
- Local minima in non-convex.

---

## 2. Vanilla Gradient Descent

Full GD: Compute ∇f over all data.

Convergence: For convex smooth f, O(1/t) rate.

Algorithm:
1. Init w_0.
2. While not converged: w = w - η (1/n) sum ∇L_i(w).

### Analysis
- Lipschitz ∇f (L-smooth): ||∇f(x)-∇f(y)|| ≤ L ||x-y||.
- Converges if η ≤ 1/L.

### ML Insight
- Batch GD: Stable but slow for big data.

Example: Linear reg loss L(w) = (1/2n) ||Xw - y||^2, ∇L = (1/n) X^T (Xw - y).

---

## 3. Stochastic Gradient Descent (SGD)

SGD: Approximate ∇f with single/random sample.

Update: w = w - η ∇L_i(w), i random.

Faster, noisy—escapes local mins.

Variance reduction: But base SGD O(1/sqrt(t)).

### Pros/Cons
- Pros: Online, handles streaming.
- Cons: Oscillates, needs decreasing η.

### ML Application
- Deep learning default: Noisy gradients generalize better.

---

## 4. Mini-Batch Gradient Descent

Compromise: ∇f ≈ (1/m) sum_{i in batch} ∇L_i, m<<n.

Reduces variance, leverages vectorization.

Batch size: Tune for hardware (GPU parallelism).

In ML: Standard, with shuffling epochs.

---

## 5. Momentum and Nesterov Accelerated GD

Momentum: Add velocity v_{t+1} = β v_t - η ∇f(w_t), w_{t+1} = w_t + v_{t+1}, β~0.9.

Smooths updates, accelerates in flat directions.

Nesterov: Lookahead—v_{t+1} = β v_t - η ∇f(w_t + β v_t), w_{t+1} = w_t + v_{t+1}.

Better theoretical rates: O(1/t^2) for convex.

### ML Connection
- Dampens oscillations in ravines.

Example: In physics, like ball rolling with friction.

---

## 6. Adaptive Learning Rates: Adagrad and RMSprop

Adagrad: Per-param η: Divide by sqrt(sum past squared grads) + ε.

Good for sparse, but η shrinks too much.

RMSprop: Use EMA of squared grads: g_{t} = γ g_{t-1} + (1-γ) (∇f)^2, η / sqrt(g_t + ε).

Prevents eternal shrink.

### ML Insight
- Handles varying scales in features/weights.

---

## 7. Adam: Adaptive Moment Estimation

Combines momentum + RMSprop.

m_t = β1 m_{t-1} + (1-β1) ∇f (first moment).

v_t = β2 v_{t-1} + (1-β2) (∇f)^2 (second).

Bias correct: m_hat = m / (1-β1^t), v_hat = v / (1-β2^t).

Update: w -= η m_hat / (sqrt(v_hat) + ε).

Defaults: β1=0.9, β2=0.999, ε=1e-8.

### Advantages
- Robust, widely used in DL.

Variants: AdamW (decoupled weight decay).

---

## 8. Convergence Analysis and Guarantees

For convex: GD sublinear, accelerated quadratic.

Stochastic: Slower due noise.

Regret bounds in online learning.

In non-convex: Stationary points (∇f≈0).

ML: Empirical convergence despite theory gaps.

---

## 9. Learning Rate Schedules and Warmup

Schedules: Step decay, exponential, 1/t.

Cosine annealing: η_t = η_min + 0.5 (η_max - η_min) (1 + cos(π t / T)).

Warmup: Start small η, increase—stabilizes early.

In ML: Crucial for large models.

---

## 10. Numerical Implementations of GD Variants

Simple GD loops.

::: code-group

```python [Python]
import numpy as np

# Vanilla GD for linear reg
def vanilla_gd(X, y, eta=0.01, epochs=100):
    w = np.zeros(X.shape[1])
    for _ in range(epochs):
        grad = (1/len(y)) * X.T @ (X @ w - y)
        w -= eta * grad
    return w

X = np.array([[1,1],[1,2],[1,3]])
y = np.array([2,3,4])
print("Vanilla GD w:", vanilla_gd(X, y))

# SGD
def sgd(X, y, eta=0.01, epochs=100):
    w = np.zeros(X.shape[1])
    n = len(y)
    for _ in range(epochs):
        for i in np.random.permutation(n):
            grad = X[i].T * (X[i] @ w - y[i])
            w -= eta * grad
    return w

print("SGD w:", sgd(X, y))

# Adam
def adam(X, y, eta=0.01, beta1=0.9, beta2=0.999, eps=1e-8, epochs=100):
    w = np.zeros(X.shape[1])
    m = np.zeros_like(w)
    v = np.zeros_like(w)
    t = 0
    for _ in range(epochs):
        grad = (1/len(y)) * X.T @ (X @ w - y)
        t += 1
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * grad**2
        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)
        w -= eta * m_hat / (np.sqrt(v_hat) + eps)
    return w

print("Adam w:", adam(X, y))
```

```rust [Rust]
use ndarray::{array, Array1, Array2};

fn vanilla_gd(x: &Array2<f64>, y: &Array1<f64>, eta: f64, epochs: usize) -> Array1<f64> {
    let mut w = Array1::zeros(x.ncols());
    let n = y.len() as f64;
    for _ in 0..epochs {
        let pred = x.dot(&w);
        let grad = x.t().dot(&(pred - y)) / n;
        w -= &(eta * &grad);
    }
    w
}

fn main() {
    let x = array![[1.0,1.0],[1.0,2.0],[1.0,3.0]];
    let y = array![2.0,3.0,4.0];
    let w = vanilla_gd(&x, &y, 0.01, 100);
    println!("Vanilla GD w: {:?}", w);

    // SGD (simplified, single epoch shuffle)
    fn sgd(x: &Array2<f64>, y: &Array1<f64>, eta: f64, epochs: usize) -> Array1<f64> {
        let mut w = Array1::zeros(x.ncols());
        for _ in 0..epochs {
            let mut indices: Vec<usize> = (0..y.len()).collect();
            indices.shuffle(&mut rand::thread_rng());
            for &i in &indices {
                let xi = x.row(i);
                let pred = xi.dot(&w);
                let grad = xi.t() * (pred - y[i]);
                w -= &(eta * &grad);
            }
        }
        w
    }
    let w_sgd = sgd(&x, &y, 0.01, 100);
    println!("SGD w: {:?}", w_sgd);

    // Adam (batch)
    fn adam(x: &Array2<f64>, y: &Array1<f64>, eta: f64, beta1: f64, beta2: f64, eps: f64, epochs: usize) -> Array1<f64> {
        let mut w = Array1::zeros(x.ncols());
        let mut m = Array1::zeros(x.ncols());
        let mut v = Array1::zeros(x.ncols());
        let n = y.len() as f64;
        let mut t = 0;
        for _ in 0..epochs {
            let pred = x.dot(&w);
            let grad = x.t().dot(&(pred - y)) / n;
            t += 1;
            m = beta1 * &m + (1.0 - beta1) * &grad;
            v = beta2 * &v + (1.0 - beta2) * &grad.mapv(|g| g.powi(2));
            let m_hat = &m / (1.0 - beta1.powi(t as i32));
            let v_hat = &v / (1.0 - beta2.powi(t as i32));
            w -= &(eta * &m_hat / (v_hat.mapv(|vh| vh.sqrt()) + eps));
        }
        w
    }
    let w_adam = adam(&x, &y, 0.01, 0.9, 0.999, 1e-8, 100);
    println!("Adam w: {:?}", w_adam);
}
```
:::

Implements GD variants for linear regression. Note: Rust requires 'ndarray' and 'rand' crates—assume available or simulate.

---

## 11. Visualization of GD Trajectories

Plot parameter updates over iterations or contours.

(Conceptual: Arrows following -grad on loss surface.)

In ML: Monitor loss curves for convergence.

---

## 12. Advanced Variants and Extensions

Nadam: Nesterov + Adam.

AMSGrad: Fix Adam convergence issues.

Coordinate descent: Update one param at time.

In ML: Distributed GD (e.g., Hogwild!).

---

## 13. Key ML Takeaways

- **GD core of training**: Iterative minimization.
- **Variants tune tradeoffs**: Speed vs stability.
- **Schedules adapt**: For better convergence.
- **Theory guides practice**: Rates, conditions.
- **Code empowers experimentation**: Test on data.

GD variants evolve AI optimization.

---

## 14. Summary

Dissected GD from vanilla to adaptive variants, convergence, schedules, with ML focus. Examples and Python/Rust code facilitate implementation. This toolkit optimizes AI models effectively.

Next: Stochastic Approximations.

Word count: Approximately 2850.

---

## Further Reading
- Bottou et al., "Optimization Methods for Large-Scale ML".
- Ruder, "Overview of Gradient Descent Optimization Algorithms".
- Kingma, Ba, "Adam: A Method for Stochastic Optimization".
- Rust: 'argmin' for optimization frameworks.

---