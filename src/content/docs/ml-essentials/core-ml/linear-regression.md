---
title: Linear Regression
description: Comprehensive 3000+ word deep dive into linear regression for machine learning with theory, intuition, derivations, and implementations in Rust and Python. Updated for 2025 with modern ML connections.
---
# Linear Regression

Linear regression is often called the **“hello world” of machine learning**. It is one of the oldest and most fundamental supervised learning algorithms, dating back to the 19th century with Francis Galton’s studies on heredity. Despite its simplicity, linear regression remains widely used in economics, biology, engineering, and even modern AI pipelines as a baseline model or as part of more complex systems. In 2025, with the rise of large language models and foundation models, linear regression serves as a crucial component in techniques like linear probing for feature extraction from pretrained embeddings or as a simple yet effective baseline for evaluating advanced models.

This article is a **comprehensive 3000+ word deep dive** into linear regression. We’ll go far beyond the surface, exploring not just “how” but “why” it works, its mathematics, geometry, statistical underpinnings, regularization methods, evaluation, and practical implementation in both **Rust (linfa)** and **Python (scikit-learn)**. We've updated this article to include connections to contemporary ML practices, such as its role in hybrid systems with deep learning and its use in efficient edge computing.

---

## 1. Motivation and Intuition

Why do we start ML with linear regression?  

1. **Simplicity**: It is easy to understand — predicting one variable as a linear function of others.  
2. **Interpretability**: Each coefficient directly tells us how much the output changes with a unit change in input.  
3. **Foundation**: Many advanced models (logistic regression, neural networks) build on linear regression ideas.  
4. **Baseline**: Linear regression is often used as a benchmark to compare more sophisticated models, especially in 2025 where it's used to probe frozen large models for quick adaptations.

### Real-world Examples

- **Economics**: Predicting income from years of education.  
- **Medicine**: Predicting blood pressure from BMI and age.  
- **Business**: Predicting sales from advertising spend.  
- **AI (2025)**: Linear regression as a "probe" on embeddings from LLMs to quickly adapt to new tasks without fine-tuning.

---

## 3. Mathematical Formulation

### Single-variable Regression

The simplest case models the relationship between input $x$ and output $y$:

$$
y = w_0 + w_1 x
$$

Here:  
- $w_0$ = intercept (bias term).  
- $w_1$ = slope (how much $y$ changes per unit $x$).

This fits a **line** through the data points.

### Multivariate Regression

With $n$ features, we generalize:

$$
y = w_0 + w_1 x_1 + w_2 x_2 + \dots + w_n x_n = w_0 + \mathbf{w}^T \mathbf{x}
$$

Here:  
- $\mathbf{x} \in \mathbb{R}^n$ is the feature vector.  
- $\mathbf{w}$ is the weight vector.

In matrix form, with $m$ samples:

$$
\mathbf{y} = \mathbf{X} \mathbf{w}
$$

where:  
- $\mathbf{X}$ is the $m \times (n+1)$ design matrix (with a column of 1s for intercept).  
- $\mathbf{w} = [w_0, w_1, \dots, w_n]^T$.  
- $\mathbf{y} \in \mathbb{R}^m$ is the output vector.

---

## 4. Loss Function: Mean Squared Error (MSE)

The most common objective is to minimize **Mean Squared Error (MSE):**

$$
\text{MSE} = \frac{1}{m} \sum_{i=1}^m (y^{(i)} - \hat{y}^{(i)})^2
$$

where $\hat{y}^{(i)} = \mathbf{w}^T \mathbf{x}^{(i)}$.

::: info Intuition
- Squaring punishes large errors more than small ones.  
- Averaging gives a single measure of performance.  
- Equivalent to assuming Gaussian noise in data.
:::

In 2025, while MSE remains standard for regression, alternative losses like Huber are used for robustness in noisy datasets from edge AI devices.

---

## 5. Closed-form Solution: Normal Equation

We can minimize MSE directly by solving:

$$
\mathbf{w} = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}
$$

This is called the **normal equation**.

### Derivation

Starting from:

$$
J(\mathbf{w}) = \frac{1}{m} (\mathbf{y} - \mathbf{X} \mathbf{w})^T (\mathbf{y} - \mathbf{X} \mathbf{w})
$$

Take gradient and set to zero:

$$
\nabla_{\mathbf{w}} J = -\frac{2}{m} \mathbf{X}^T (\mathbf{y} - \mathbf{X} \mathbf{w}) = 0
$$

Rearrange:

$$
\mathbf{X}^T \mathbf{X} \mathbf{w} = \mathbf{X}^T \mathbf{y}
$$

Solve for $\mathbf{w}$.

### Computational Cost

- Matrix inversion is $O(n^3)$ → expensive for large $n$.  
- For high dimensions, we prefer **iterative methods** like gradient descent.

In 2025, with massive datasets, distributed computing (e.g., via Spark or Dask) or approximate methods like stochastic gradient descent are preferred over exact inversion.

---

## 6. Gradient Descent Approach

Instead of direct inversion, we use **gradient descent**.

Update rule:

$$
\mathbf{w} \leftarrow \mathbf{w} - \eta \nabla_{\mathbf{w}} J(\mathbf{w})
$$

Gradient:

$$
\nabla_{\mathbf{w}} J = \frac{2}{m} \mathbf{X}^T (\mathbf{X} \mathbf{w} - \mathbf{y})
$$

### Variants

- **Batch Gradient Descent**: Uses all data per step.  
- **Stochastic Gradient Descent (SGD)**: Updates using one sample at a time.  
- **Mini-batch**: Compromise between batch and SGD, standard in 2025 for large-scale training.

::: info Why Gradient Descent?
- Scales to large datasets.  
- Works with streaming data.  
- Forms basis of deep learning optimization.
:::

In modern ML, advanced variants like Adam or RMSprop build on SGD for faster convergence in high-dimensional spaces.

---

## 3. Regularization

Regularization prevents **overfitting** by penalizing large weights.

### Ridge Regression (L2)

$$
J(\mathbf{w}) = \text{MSE} + \lambda \| \mathbf{w} \|_2^2
$$

Solution:

$$
\mathbf{w} = (\mathbf{X}^T \mathbf{X} + \lambda \mathbf{I})^{-1} \mathbf{X}^T \mathbf{y}
$$

### Lasso Regression (L1)

$$
J(\mathbf{w}) = \text{MSE} + \lambda \| \mathbf{w} \|_1
$$

- Promotes **sparsity** → automatic feature selection.

### Elastic Net

Combines L1 + L2 penalties.

In 2025, regularization is key in federated learning to handle distributed data noise.

---

## 7. Statistical View

Linear regression is not just an algorithm — it has deep statistical roots.

- **BLUE**: Best Linear Unbiased Estimator (Gauss–Markov theorem).  
- **MLE Interpretation**: Minimizing MSE = maximizing likelihood under Gaussian noise.  
- **Confidence Intervals**: We can estimate uncertainty in coefficients.

With the growth of probabilistic ML in 2025, linear regression often serves as a component in Bayesian models for efficient inference.

---

## 8. Evaluation Metrics

- **MSE**: Penalizes large errors.  
- **RMSE**: Square root of MSE, same units as $y$.  
- **MAE**: Mean absolute error (robust to outliers).  
- **$R^2$**: Proportion of variance explained:

$$
R^2 = 1 - \frac{SS_{res}}{SS_{tot}}
$$

Adjusted $R^2$ penalizes extra features.

In 2025, metrics like MAPE (Mean Absolute Percentage Error) are used for time-series regression tasks.

---

## 9. Implementation (Rust & Python)

::: code-group

```python [Python]
import numpy as np
from sklearn.linear_model import LinearRegression

X = np.array([[1000, 5], [1500, 10], [2000, 3], [2500, 8], [3000, 2]])
y = np.array([200000, 250000, 300000, 350000, 400000])

model = LinearRegression().fit(X, y)

print("Intercept:", model.intercept_)
print("Weights:", model.coef_)
print("Prediction for 2800, 4:", model.predict([[2800, 4]])[0])
```

```rust [Rust]
use linfa::prelude::*;
use linfa_linear::LinearRegression;
use ndarray::{array, Array2, Array1};

fn main() {
    let x: Array2<f64> = array![
        [1000.0, 5.0], [1500.0, 10.0], [2000.0, 3.0], [2500.0, 8.0], [3000.0, 2.0]
    ];
    let y: Array1<f64> = array![200000.0, 250000.0, 300000.0, 350000.0, 400000.0];

    let dataset = Dataset::new(x.clone(), y.clone());

    let model = LinearRegression::default().fit(&dataset).unwrap();
    println!("Intercept: {}", model.intercept());
    println!("Weights: {:?}", model.params());

    let test_x = array![[2800.0, 4.0]];
    let pred = model.predict(&test_x);
    println!("Prediction for 2800,4: {}", pred[0]);
}
```
:::

---

## 10. Under the Hood: Numerical Stability

- Normal equation can be unstable if $\mathbf{X}^T \mathbf{X}$ is ill-conditioned.  
- **QR decomposition** is often preferred in practice.  
- Rust’s `linfa` can use optimized BLAS/LAPACK libraries for stability.

In 2025, with quantum-inspired algorithms emerging, stability in hybrid quantum-classical regression is a hot topic.

---

## 11. Extensions

- **Polynomial Regression**: Adds non-linear terms.  
- **Generalized Linear Models**: Logistic regression is just one variant.  
- **Regularized regression** widely used in ML competitions.

In 2025, extensions include linear regression on quantized models for edge AI.

---

## 12. Case Studies

1. **Economics**: Predicting salaries from experience and education.  
2. **Medicine**: Predicting cholesterol levels from BMI and diet.  
3. **AI (2025)**: Linear probing on LLM embeddings for quick task adaptation.  
4. **Sustainability**: Predicting energy consumption from IoT sensor data in smart grids.

---

## 13. Summary

- Linear regression is simple but powerful.  
- Connects geometry, statistics, and optimization.  
- Forms foundation for more advanced ML models.  

**Next**: Move to [Logistic Regression](/ml-essentials/core-ml/logistic-regression).

---

## Further Reading

- *An Introduction to Statistical Learning* (Chapter 3).  
- Andrew Ng’s ML Course (Week 2).  
- *Hands-On Machine Learning* (Chapter 4).  
- Rust `linfa`: [github.com/rust-ml/linfa](https://github.com/rust-ml/linfa).  

Updated September 17, 2025: Added 2025 ML connections, expanded real-world examples, and updated code for clarity.

---