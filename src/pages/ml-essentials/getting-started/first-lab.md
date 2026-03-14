---
title: First ML Lab
description: First machine learning lab with Rust and Python
layout: ../../../layouts/TutorialPage.astro
---

# First ML Lab

This section introduces your first machine learning (ML) task: **linear regression** using either the `linfa` library in Rust or `scikit-learn` in Python.  
You'll train a model to predict a continuous output, learning the basics of supervised learning. No prior ML experience is required.

## What is Linear Regression? (Detailed)

Linear regression models the relationship between one or more input features and a continuous target variable by fitting a linear function. It is one of the simplest and most interpretable supervised learning algorithms — a great first lab for understanding ML end-to-end.

### Model and Prediction
With $n$ features, the linear model predicts:

$$
\hat{y} = w_0 + w_1 x_1 + w_2 x_2 + \dots + w_n x_n
$$

where $w_0$ is the intercept and $w_1, \dots, w_n$ are weights (parameters). For a single feature $x$, this reduces to a line $\hat{y} = w_0 + w_1 x$.

### Loss Function (Mean Squared Error)
The most common objective is to minimize the **Mean Squared Error (MSE)** over the training set of size $m$:

$$
\text{MSE} = \frac{1}{m} \sum_{i=1}^m (y^{(i)} - \hat{y}^{(i)})^2
$$

Minimizing MSE gives the ordinary least squares solution.

**Quick intuition:** The MSE penalizes large errors more than small ones (because of the square), and averaging over $m$ gives a single performance number.

::: info Explanation of MSE
This formula computes the **average squared difference** between the actual value $y^{(i)}$ and the predicted value $\hat{y}^{(i)}$.

- Squaring ensures errors don't cancel out and emphasizes larger errors.  
- Averaging over $m$ samples gives a single-number summary of performance.  
- Smaller MSE indicates better fit.

**Mini numerical example**: suppose $y = [2, 4]$ and $\hat{y} = [2.5, 3.5]$.  
Compute squared errors: $(2 - 2.5)^2 = 0.25$, $(4 - 3.5)^2 = 0.25$.  
MSE = $(0.25 + 0.25) / 2 = 0.25$.
:::

### Closed-form Solution (Normal Equation)
For linear regression without regularization, the weights can be computed in closed form using the normal equation. If $X$ is the design matrix (with a leading column of ones for the intercept) and $y$ is the target vector:

$$
w = (X^T X)^{-1} X^T y
$$

This is efficient for small-to-medium problems but can be numerically unstable or expensive when $X^T X$ is ill-conditioned or when there are many features.

**Quick intuition:** The normal equation solves the linear system $X w = y$ in a least-squares sense by projecting onto a solvable space.

::: info Explanation of Normal Equation
Start with the system of linear equations:
$$
X w = y
$$

Pre-multiply both sides by $X^T$:
$$
X^T X w = X^T y
$$

Assuming $X^T X$ is invertible, multiply both sides by $(X^T X)^{-1}$ to isolate $w$:
$$
w = (X^T X)^{-1} X^T y
$$

Why this helps: $(X^T X)$ is square (even if $X$ is not), so we can take its inverse (when it exists) and solve for $w$ explicitly.
:::

### Optimization (Gradient Descent)
When data is large or when using regularization, iterative optimization like **gradient descent** is used. For MSE, the gradient with respect to the weights is:

$$
\nabla_w = -\frac{2}{m} X^T (y - X w)
$$

and weights are updated by stepping against the gradient (e.g., $w := w - \eta \nabla_w$ where $\eta$ is the learning rate).

**Quick intuition:** The gradient points to the direction of greatest increase in loss; moving opposite the gradient reduces loss.

::: info Explanation of Gradient Descent
The gradient $\nabla_w$ tells us the **direction of steepest increase** of the loss. To minimize the loss we take steps in the **opposite direction** (negative gradient).

- The factor $\frac{2}{m}$ scales the gradient according to dataset size.  
- In practice we use a learning rate $\eta$ so updates look like: $w \leftarrow w - \eta \nabla_w$.  
- For large datasets, variants like SGD (stochastic gradient descent) or mini-batch SGD are used.
:::

### Regularization
To avoid overfitting, regularization penalizes large weights. Two common variants:

- **Ridge (L2)**: adds $\lambda \|w\|_2^2$ to the loss. Closed-form:  
  $$
  w = (X^T X + \lambda I)^{-1} X^T y
  $$  
- **Lasso (L1)**: adds $\lambda \|w\|_1$; encourages sparsity but requires iterative solvers.

::: info Explanation of Regularization
Adding a penalty term with coefficient $\lambda$ discourages large weights, which often reduces overfitting:

- **Ridge (L2)** shrinks weights smoothly and is effective when many small contributions exist.  
- **Lasso (L1)** can set some weights exactly to zero, performing feature selection automatically.  
- The hyperparameter $\lambda$ controls the strength of the penalty; larger $\lambda$ ⇒ stronger shrinkage.
:::

### Assumptions and Limitations
Linear regression assumes:

- Linearity between features and target (or transformed features).  
- Errors are independent and identically distributed with zero mean.  
- No (strong) multicollinearity among features.  

Violations lead to biased or high-variance estimates. We'll discuss diagnostics (residual plots, multicollinearity) in later modules.

---

## Practical Considerations (Before Training)

- **Train/Test Split**: Always evaluate on unseen data. Common splits: 70/30 or 80/20.  
- **Feature Scaling**: Standardize features when using gradient-based solvers or regularization.  
- **Multicollinearity**: Highly correlated features inflate variance — consider PCA or feature selection.  
- **Outliers**: Can strongly affect least-squares solutions; consider robust methods if needed.  
- **Evaluation Metrics**: MSE, RMSE (sqrt of MSE), MAE (mean absolute error), and $R^2$ (coefficient of determination).

---

## Lab: Linear Regression (Code + Explanation)

You'll train a linear regression model on a small synthetic dataset and inspect the learned parameters and predictions. Both Rust and Python examples are provided — use the tab UI to switch between them in the docs.

::: code-group

```python [Python]
# first_lab.py - scikit-learn example
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Synthetic dataset: feature (x) and target (y)
X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
y = np.array([2.1, 4.2, 6.1, 8.3, 10.0])

# Train/test split (tiny dataset — in practice use larger data)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# Train linear regression model
model = LinearRegression().fit(X_train, y_train)

# Predict on test set and new data
y_pred = model.predict(X_test)
prediction_for_6 = model.predict(np.array([[6.0]]))[0]

# Metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Model intercept:", model.intercept_)
print("Model weights:", model.coef_)
print("Test MSE:", mse)
print("Test R2:", r2)
print("Prediction for x=6:", prediction_for_6)
```

```rust [Rust]
// main.rs - linfa example
use linfa::prelude::*;
use linfa_linear::LinearRegression;
use ndarray::{array, Axis};
use linfa::dataset::Dataset;
use ndarray::Array2;

fn main() {
    // Synthetic dataset
    let x = array![[1.0], [2.0], [3.0], [4.0], [5.0]];
    let y = array![2.1, 4.2, 6.1, 8.3, 10.0];

    // Create dataset (linfa expects an Array2 for features)
    let x2: Array2<f64> = x.clone();
    let dataset = Dataset::new(x2, y);

    // Split dataset manually (simple example)
    let (train, test) = dataset.split_with_ratio(0.6);

    // Train model
    let model = LinearRegression::default().fit(&train).unwrap();

    // Predict
    let y_pred = model.predict(&test.records);
    let prediction_for_6 = model.predict(&array![[6.0]]);

    // Compute simple MSE manually
    let mse: f64 = (&y_pred - &test.targets).mapv(|v| v.powi(2)).mean().unwrap();

    println!("Intercept: {}", model.intercept());
    println!("Weights: {:?}", model.params());
    println!("Test MSE: {}", mse);
    println!("Prediction for x=6: {}", prediction_for_6[0]);
}
```

:::

### Dependencies

::: code-group

```python (pip)
pip install numpy scikit-learn
```

```rust (toml)
[dependencies]
linfa = "0.7.1"
linfa-linear = "0.7.0"
ndarray = "0.15.0"
linfa-datasets = "0.7.0"
```

:::

### Run the Program

::: code-group

```python
python first_lab.py
```

```rust
cargo run
```

:::

## Interpreting Results (Deeper)

- **Intercept and Weights**: The intercept is the model's baseline; weights show how much the target changes per unit change in each feature.  
- **Goodness of Fit**: Use $R^2$ to measure the fraction of variance explained by the model:  

$$
R^2 = 1 - \frac{SS_{res}}{SS_{tot}}
$$

- **Bias–Variance Tradeoff**: Simple models (high bias) underfit; very flexible models (high variance) overfit. Linear regression is low-variance if features are few and samples many.  
- **Residual Analysis**: Plot residuals to check homoscedasticity (constant variance) and patterns indicating model misspecification.

::: info Explanation of R^2
$R^2$ compares the model against a naive baseline (mean of $y$):

- $SS_{res}$ = sum of squared residuals (errors) from your model.  
- $SS_{tot}$ = total sum of squared deviations from the mean of $y$.  

**Mini numerical example**: for $y = [2, 4]$ and $\hat{y} = [2.5, 3.5]$:  
- $SS_{res} = (2 - 2.5)^2 + (4 - 3.5)^2 = 0.5$  
- $SS_{tot} = (2 - 3)^2 + (4 - 3)^2 = 2$  
So $R^2 = 1 - 0.5/2 = 0.75$ (the model explains 75% of variance).
:::

## Practical Extensions

- **Polynomial Features**: Model non-linear relationships by adding polynomial terms (e.g., $x^2$, $x^3$).  
- **Regularized Regression**: Use Ridge or Lasso to penalize large weights.  
- **Cross-validation**: Use K-fold CV to better estimate performance on small datasets.  
- **Feature Engineering**: Create meaningful features, handle categorical variables (one-hot encoding), and impute missing values.

## Next Steps

- Proceed to [Mathematical Foundations](/ml-essentials/maths-for-aiml/linear-algebra/scalars-vectors-matrices) for ML's mathematical basis.  
- Or revisit [Rust Basics](/ml-essentials/getting-started/rust-basics) / [Python Basics](/ml-essentials/getting-started/python-basics).

## Further Reading

- *An Introduction to Statistical Learning* by James et al. (Chapter 3)  
- Andrew Ng's Machine Learning Specialization (Course 1, Week 1)  
- `linfa` Documentation: https://github.com/rust-ml/linfa  
- `scikit-learn` Documentation: https://scikit-learn.org
