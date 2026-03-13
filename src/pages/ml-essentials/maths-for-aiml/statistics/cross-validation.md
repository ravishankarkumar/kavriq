---
title: Cross-Validation & Resampling for ML
description: Comprehensive exploration of cross-validation and resampling techniques in statistics for AI/ML, covering k-fold, leave-one-out, bootstrap, and their applications in model evaluation and hyperparameter tuning, with examples and code in Python and Rust
layout: ../../../../layouts/TutorialPage.astro
---

# Cross-Validation & Resampling for ML

Cross-validation and resampling are essential techniques in machine learning (ML) for assessing model performance, estimating generalization error, and tuning hyperparameters. Cross-validation splits data into subsets to simulate out-of-sample testing, while resampling methods like bootstrap provide robust estimates of variability. These approaches ensure models generalize well to unseen data, avoiding overfitting and underfitting.

This twelfth lecture in the "Statistics Foundations for AI/ML" series builds on bias-variance and resampling, exploring k-fold cross-validation, leave-one-out, stratified variants, bootstrap methods, and their ML applications. We'll provide intuitive explanations, theoretical foundations, and practical implementations in Python and Rust, preparing you for statistical significance and advanced topics.

---

## 1. Why Cross-Validation and Resampling Matter in ML

ML models must generalize to new data, but training on all available data risks overfitting, and a single train-test split may be unreliable. Cross-validation and resampling:
- Estimate test error without sacrificing training data.
- Assess model stability and variability.
- Guide hyperparameter tuning for optimal performance.

### ML Connection
- **Model Evaluation**: Estimate accuracy or loss on unseen data.
- **Hyperparameter Tuning**: Select best model configurations.
- **Uncertainty Quantification**: Bootstrap for confidence intervals.

::: info
Cross-validation is like tasting multiple spoonfuls from a pot of soup to ensure it's consistently good; resampling adds confidence to the flavor profile.
:::

### Example
- K-fold CV: Split data into 5 parts, train on 4, test on 1, repeat to estimate model accuracy.

---

## 2. Cross-Validation: Principles and Methods

**Cross-Validation (CV)**: Split data into training and validation sets multiple times to estimate performance.

### K-Fold Cross-Validation
- Divide data into k folds (e.g., k=5 or 10).
- Train on k-1 folds, test on 1, repeat k times.
- Average performance across folds.

**Error Estimate**:

\[
\text{CV Error} = \frac{1}{k} \sum_{i=1}^k \text{Error}_i
\]

### Leave-One-Out CV (LOOCV)
- k=n (n samples), each sample is test set once.
- Unbiased but computationally expensive.

### Stratified K-Fold
- Ensure class proportions preserved in folds (for classification).

### Properties
- **Bias**: LOOCV low bias, high variance; k-fold balances.
- **Variance**: Smaller k, lower variance, higher bias.

### ML Application
- K-fold for hyperparameter tuning in grid search.
- Stratified CV for imbalanced datasets.

Example: 5-fold CV on 100 samples, each fold tests 20 samples, average accuracy.

---

## 3. Bootstrap for Model Evaluation

**Bootstrap**: Sample with replacement to estimate statistic variability.

**Out-of-Bag (OOB) Error**:
- Samples not in bootstrap used as test set.
- OOB error approximates CV error.

**Algorithm**:
1. Draw B bootstrap samples (size n).
2. Train model on each, evaluate on OOB samples.
3. Average performance or compute CI.

### Properties
- Robust for small datasets.
- Higher variance than k-fold.

### ML Connection
- Bootstrap CI for model accuracy.

---

## 4. Theoretical Foundations

**CV Error**: Estimates expected test error E[L(\hat{f}, D_{test})].

**Bootstrap**: Approximates sampling distribution via empirical distribution.

**Bias-Variance Tradeoff**:
- K-fold: Smaller k increases bias, reduces variance.
- LOOCV: Low bias, high variance.
- Bootstrap: Balances via OOB.

### Assumptions
- i.i.d. data for unbiased estimates.
- Stationarity for time-series adjustments.

---

## 5. Applications in Machine Learning

1. **Model Evaluation**: K-fold CV for robust accuracy estimates.
2. **Hyperparameter Tuning**: Grid search with CV.
3. **Feature Selection**: Assess stability via bootstrap.
4. **Ensemble Methods**: OOB error in random forests.

### Challenges
- **Computation**: LOOCV costly for large n.
- **Non-i.i.d. Data**: Time-series, use time-series CV.
- **Imbalanced Data**: Stratified CV critical.

---

## 6. Numerical Cross-Validation and Resampling

Implement k-fold CV, bootstrap.

::: code-group

```python [Python]
import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# K-fold CV
X = np.random.rand(100, 2)
y = (X[:,0] + X[:,1] > 1).astype(int)
model = LogisticRegression()
kf = KFold(n_splits=5, shuffle=True, random_state=0)
scores = cross_val_score(model, X, y, cv=kf, scoring='accuracy')
print("5-Fold CV Accuracy:", scores.mean(), "±", scores.std())

# Stratified K-fold
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
scores_strat = cross_val_score(model, X, y, cv=skf, scoring='accuracy')
print("Stratified CV Accuracy:", scores_strat.mean(), "±", scores_strat.std())

# Bootstrap OOB
def bootstrap_oob(X, y, model, n_boots=200):
    n = len(y)
    scores = []
    for _ in range(n_boots):
        idx = np.random.choice(n, n)
        oob_idx = np.setdiff1d(np.arange(n), np.unique(idx))
        if len(oob_idx) == 0:
            continue
        X_train, y_train = X[idx], y[idx]
        X_oob, y_oob = X[oob_idx], y[oob_idx]
        model.fit(X_train, y_train)
        scores.append(accuracy_score(y_oob, model.predict(X_oob)))
    return np.mean(scores), np.std(scores)

oob_mean, oob_std = bootstrap_oob(X, y, LogisticRegression())
print("Bootstrap OOB Accuracy:", oob_mean, "±", oob_std)
```

```rust [Rust]
use rand::Rng;
use rand::seq::SliceRandom;

struct LogisticModel {
    weights: Vec<f64>,
}

impl LogisticModel {
    fn new(dim: usize) -> Self {
        LogisticModel { weights: vec![0.0; dim] }
    }
    fn fit(&mut self, x: &[[f64]], y: &[u8], max_iter: usize) {
        let eta = 0.01;
        for _ in 0..max_iter {
            let mut grad = vec![0.0; x[0].len()];
            for (xi, &yi) in x.iter().zip(y.iter()) {
                let p = 1.0 / (1.0 + (-xi.iter().zip(self.weights.iter()).map(|(&xij, &wj)| xij * wj).sum::<f64>()).exp());
                let err = yi as f64 - p;
                for j in 0..grad.len() {
                    grad[j] += err * xi[j];
                }
            }
            for j in 0..self.weights.len() {
                self.weights[j] += eta * grad[j];
            }
        }
    }
    fn predict(&self, x: &[[f64]]) -> Vec<u8> {
        x.iter().map(|xi| {
            let p = 1.0 / (1.0 + (-xi.iter().zip(self.weights.iter()).map(|(&xij, &wj)| xij * wj).sum::<f64>()).exp());
            if p > 0.5 { 1 } else { 0 }
        }).collect()
    }
}

fn k_fold_cv(x: &[[f64]], y: &[u8], k: usize) -> (f64, f64) {
    let mut rng = rand::thread_rng();
    let n = x.len();
    let fold_size = n / k;
    let mut indices: Vec<usize> = (0..n).collect();
    indices.shuffle(&mut rng);
    let mut scores = vec![];
    for i in 0..k {
        let test_start = i * fold_size;
        let test_end = if i == k-1 { n } else { (i+1) * fold_size };
        let mut x_train = vec![];
        let mut y_train = vec![];
        let mut x_test = vec![];
        let mut y_test = vec![];
        for j in 0..n {
            if j >= test_start && j < test_end {
                x_test.push(x[indices[j]].to_vec());
                y_test.push(y[indices[j]]);
            } else {
                x_train.push(x[indices[j]].to_vec());
                y_train.push(y[indices[j]]);
            }
        }
        let mut model = LogisticModel::new(x[0].len());
        model.fit(&x_train, &y_train, 100);
        let y_pred = model.predict(&x_test);
        let acc = y_test.iter().zip(y_pred.iter()).filter(|(&yt, &yp)| yt == yp).count() as f64 / y_test.len() as f64;
        scores.push(acc);
    }
    let mean = scores.iter().sum::<f64>() / scores.len() as f64;
    let var = scores.iter().map(|&s| (s - mean).powi(2)).sum::<f64>() / scores.len() as f64;
    (mean, var.sqrt())
}

fn main() {
    let mut rng = rand::thread_rng();
    let x: Vec<Vec<f64>> = (0..100).map(|_| vec![rng.gen(), rng.gen()]).collect();
    let y: Vec<u8> = x.iter().map(|xi| if xi[0] + xi[1] > 1.0 { 1 } else { 0 }).collect();
    let (mean, std) = k_fold_cv(&x, &y, 5);
    println!("5-Fold CV Accuracy: {} ± {}", mean, std);
}
```
:::

Implements k-fold CV and bootstrap OOB.

---

## 7. Theoretical Insights

**CV Error**: Approximates E[L(\hat{f}, D_{test})], low bias for LOOCV, balanced for k-fold.

**Bootstrap OOB**: Similar to CV, slightly biased but robust.

### ML Insight
- K-fold CV standard for robust evaluation.

---

## 8. Challenges in ML Applications

- **Computation**: LOOCV, high k costly.
- **Imbalanced Data**: Stratified CV needed.
- **Non-i.i.d.**: Time-series CV for temporal data.

---

## 9. Key ML Takeaways

- **CV estimates generalization**: Robust performance.
- **K-fold balances bias-variance**: Common choice.
- **Stratified for imbalanced**: Class proportion.
- **Bootstrap for uncertainty**: CIs, OOB.
- **Code implements**: Practical CV.

CV and resampling ensure reliable ML.

---

## 10. Summary

Explored cross-validation (k-fold, LOOCV, stratified) and bootstrap, with ML applications in evaluation and tuning. Examples and Python/Rust code bridge theory to practice. Prepares for significance testing and nonparametric stats.

Word count: Approximately 3000.

---

## Further Reading
- Hastie, *Elements of Statistical Learning* (Ch. 7).
- James, *Introduction to Statistical Learning* (Ch. 5).
- Kohavi, "A Study of Cross-Validation and Bootstrap".
- Rust: 'rand' for resampling, 'nalgebra' for data.

---