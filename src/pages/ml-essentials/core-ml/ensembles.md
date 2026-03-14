---
title: Ensemble Methods - Bagging & Random Forests
description: Comprehensive 3000+ word exploration of ensemble methods, focusing on bagging and random forests for machine learning in 2025, covering theory, mathematics, derivations, Python/Rust code, and applications in classification, regression, and modern ML pipelines.
layout: ../../../layouts/TutorialPage.astro
---

# Ensemble Methods - Bagging & Random Forests

Ensemble methods combine multiple models to improve performance, reduce variance, and enhance generalization. Bagging (Bootstrap Aggregating) and random forests are foundational ensembles that aggregate decision trees to mitigate overfitting and boost accuracy. In 2025, ensembles like random forests remain vital for interpretability, feature importance, and as baselines or components in hybrid systems with large language models (LLMs) and graph neural networks.

This lecture in the "Foundations for AI/ML" series (core-ml cluster) builds on decision trees and k-NN, exploring ensemble methods, focusing on bagging and random forests, their theoretical foundations, bias-variance reduction, and applications. We’ll provide intuitive explanations, mathematical insights, and practical implementations in **Python (scikit-learn)** and **Rust (linfa)**, ensuring a rigorous guide aligned with 2025 ML trends.

---

## 1. Motivation and Intuition

Ensembles leverage the "wisdom of the crowd": multiple weak models combine to form a strong one. Bagging reduces variance by averaging bootstrap-sampled models, while random forests add feature randomness for decorrelation.

**Why Ensembles in 2025?**
- **Robustness**: Reduce overfitting in high-d data.
- **Interpretability**: Feature importance from aggregated trees.
- **Baseline**: Compare against complex models like transformers.
- **Modern Applications**: Ensembles on LLM embeddings for quick adaptation.

### Real-World Examples
- **Fraud Detection**: Random forests classify transactions.
- **Medicine**: Predict disease outcomes from patient data.
- **AI Pipelines**: Forests on LLM features for efficient decisions.

::: info
Ensembles are like a committee vote—bagging diversifies opinions, random forests adds independence for better decisions.
:::

---

## 2. Mathematical Formulation

**Bagging**: Train B models on bootstrap samples (with replacement), aggregate predictions.

For regression: \hat{f}(x) = (1/B) sum \hat{f_b}(x).

For classification: Majority vote.

**Random Forests**: Bagging + random feature subset at each split.

m = sqrt(p) features for classification, p/3 for regression (p total features).

### Bias-Variance Reduction
- Bagging: Reduces variance, bias unchanged (for trees, low bias).
- RF: Feature randomness further decorrelates trees.

Variance reduction: Var(average) = (1/B) Var(single) if uncorrelated.

---

## 3. Deriving Ensemble Gain

**Variance Reduction**:

For uncorrelated estimators \hat{f_i}, Var((1/B) sum \hat{f_i}) = (1/B) Var(\hat{f_i}).

Correlated: Var = ρ σ² + (1-ρ)/B σ², ρ correlation.

RF minimizes ρ by random features.

### Derivation
Covariance decomposition for average variance.

### ML Insight
- RF gain from low correlation.

---

## 4. Building Random Forests

**Algorithm**:
1. For b=1 to B:
   - Bootstrap sample D_b.
   - Grow tree on D_b:
     - At each node, select m random features.
     - Split on best gain.
     - Grow to full (no pruning).
2. Aggregate: Vote/average.

**OOB Error**: Out-of-bag samples for unbiased estimate.

### ML Connection
- OOB for feature importance: Permute feature, measure OOB gain drop.

---

## 5. Feature Importance and Interpretability

**Gain-Based Importance**: Sum of gains from feature splits.

**Permutation Importance**: Shuffle feature, measure performance drop.

In 2025, SHAP values enhance tree importance in ensembles.

---

## 6. Evaluation Metrics

- **Accuracy/Precision/Recall/F1**: For classification.
- **MSE/MAE/R²**: For regression.
- **OOB Error**: Internal validation.
- **Feature Importance**: Gini/entropy gain.

In 2025, calibration metrics for probabilistic RF outputs.

---

## 7. Applications in Machine Learning (2025)

1. **Classification**: Spam detection, sentiment analysis.
2. **Regression**: House price prediction.
3. **Feature Importance**: SHAP on forests for LLM interpretability.
4. **Anomaly Detection**: Isolation forests (RF variant).
5. **Edge AI**: Efficient forests on devices.
6. **Hybrid Systems**: Forests on LLM features for efficient inference.

### Challenges
- **Computation**: Large B, n costly.
- **High-D**: Benefits from feature selection.
- **Imbalanced Data**: Weighted sampling.

---

## 8. Numerical Implementations

Implement bagging, random forests.

::: code-group

```python [Python]
import numpy as np
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

# Bagging classification
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=0)

bag_clf = BaggingClassifier(estimator=DecisionTreeClassifier(), n_estimators=10, random_state=0)
bag_clf.fit(X_train, y_train)
y_pred = bag_clf.predict(X_test)
print("Bagging Accuracy:", accuracy_score(y_test, y_pred))

# Random Forest classification
rf_clf = RandomForestClassifier(n_estimators=100, max_depth=3, criterion='gini', random_state=0)
rf_clf.fit(X_train, y_train)
y_pred_rf = rf_clf.predict(X_test)
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))

# Visualize one tree
plt.figure(figsize=(12,8))
plot_tree(rf_clf.estimators_[0], feature_names=iris.feature_names, class_names=iris.target_names, filled=True)
plt.title("Random Forest Tree")
plt.show()

# Regression
X_reg = np.random.rand(200, 1) * 10
y_reg = np.sin(X_reg).ravel() + np.random.normal(0, 0.1, 200)
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.2, random_state=0)

rf_reg = RandomForestRegressor(n_estimators=100, max_depth=3, criterion='squared_error', random_state=0)
rf_reg.fit(X_train_reg, y_train_reg)
y_pred_reg = rf_reg.predict(X_test_reg)
print("Random Forest Regression MSE:", mean_squared_error(y_test_reg, y_pred_reg))

# Feature importance
importances = rf_clf.feature_importances_
print("Feature Importances:", importances)
```

```rust [Rust]
use linfa::prelude::*;
use linfa_ensemble::{Bagging, RandomForest};
use linfa_trees::DecisionTree;
use ndarray::{Array2, Array1};

fn main() {
    // Classification (placeholder Iris)
    let x_train: Array2<f64> = Array2::zeros((120, 4));
    let y_train: Array1<i32> = Array1::zeros(120);
    let x_test: Array2<f64> = Array2::zeros((30, 4));
    let y_test: Array1<i32> = Array1::zeros(30);

    let dataset = Dataset::new(x_train, y_train);
    let base = DecisionTree::params();
    let bag_clf = Bagging::params().base_estimator(base).n_estimators(10).fit(&dataset).unwrap();
    let preds = bag_clf.predict(&x_test);
    let accuracy = preds.iter().zip(y_test.iter()).filter(|(&p, &t)| p == t).count() as f64 / y_test.len() as f64;
    println!("Bagging Accuracy: {}", accuracy);

    let rf_clf = RandomForest::params().n_trees(100).max_depth(Some(3)).fit(&dataset).unwrap();
    let preds_rf = rf_clf.predict(&x_test);
    let accuracy_rf = preds_rf.iter().zip(y_test.iter()).filter(|(&p, &t)| p == t).count() as f64 / y_test.len() as f64;
    println!("Random Forest Accuracy: {}", accuracy_rf);

    // Regression
    let x_reg: Array2<f64> = Array2::from_shape_fn((200, 1), |(_, _)| rng.gen::<f64>() * 10.0);
    let y_reg: Array1<f64> = x_reg.column(0).mapv(|x| x.sin() + rng.gen::<f64>() * 0.2 - 0.1);
    let (x_train_reg, x_test_reg, y_train_reg, y_test_reg) = (
        x_reg.slice(s![0..160, ..]).to_owned(),
        x_reg.slice(s![160..200, ..]).to_owned(),
        y_reg.slice(s![0..160]).to_owned(),
        y_reg.slice(s![160..200]).to_owned(),
    );
    let dataset_reg = Dataset::new(x_train_reg, y_train_reg);
    let rf_reg = RandomForest::params().n_trees(100).max_depth(Some(3)).fit(&dataset_reg).unwrap();
    let preds_reg = rf_reg.predict(&x_test_reg);
    let mse = preds_reg.iter().zip(y_test_reg.iter()).map(|(&p, &t)| (p - t).powi(2)).sum::<f64>() / y_test_reg.len() as f64;
    println!("Random Forest Regression MSE: {}", mse);
}

Dependencies (`Cargo.toml`):
```toml
[dependencies]
linfa = "0.7.1"
linfa-ensemble = "0.7.1"
linfa-trees = "0.7.1"
ndarray = "0.15.6"
rand = "0.8.5"
rand_distr = "0.4.3"
```
:::

Implements bagging and random forests for classification and regression.

---

## 8. Numerical Stability and High-Dimensions

- **Bagging Stability**: Reduces variance by averaging.
- **RF in High-D**: Feature subsampling mitigates curse.
- **Overfitting**: OOB for validation.

In 2025, stability in distributed ensembles is key for federated ML.

---

## 9. Case Study: Iris Dataset (Classification)

::: code-group

```python [Python]
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

# Load dataset
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

# Train Random Forest
rf = RandomForestClassifier(n_estimators=100, max_depth=3, criterion='gini', oob_score=True, random_state=0)
rf.fit(X_train, y_train)

# Evaluate
y_pred = rf.predict(X_test)
print(classification_report(y_test, y_pred))
print("OOB Score:", rf.oob_score_)

# Feature importance
importances = rf.feature_importances_
plt.bar(iris.feature_names, importances)
plt.title("Feature Importance")
plt.show()

# Visualize one tree
plt.figure(figsize=(12,8))
plot_tree(rf.estimators_[0], feature_names=iris.feature_names, class_names=iris.target_names, filled=True)
plt.title("Random Forest Tree")
plt.show()
```

```rust [Rust]
use linfa::prelude::*;
use linfa_ensemble::RandomForest;
use ndarray::{Array2, Array1};

fn main() {
    // Placeholder: Iris dataset
    let x_train: Array2<f64> = Array2::zeros((120, 4));
    let y_train: Array1<i32> = Array1::zeros(120);
    let x_test: Array2<f64> = Array2::zeros((30, 4));
    let y_test: Array1<i32> = Array1::zeros(30);

    let dataset = Dataset::new(x_train, y_train);
    let rf = RandomForest::params().n_trees(100).max_depth(Some(3)).fit(&dataset).unwrap();
    let preds = rf.predict(&x_test);
    let accuracy = preds.iter().zip(y_test.iter()).filter(|(&p, &t)| p == t).count() as f64 / y_test.len() as f64;
    println!("Random Forest Accuracy: {}", accuracy);

    // Feature importance not natively supported; compute manually
}
```
:::

**Note**: Rust requires external data loading; use Python for full visualization and importance.

---

## 12. Under the Hood Insights

- **Bootstrap Sampling**: With replacement, ~63% unique samples per tree.
- **Feature Subsampling**: m = sqrt(p) for classification.
- **OOB Error**: Internal cross-validation.
- **Variance Reduction**: Decorrelated trees.

---

## 13. Limitations

- **Computation**: Large n_estimators costly.
- **Interpretability**: Less than single trees.
- **High-D**: Benefits from feature selection.
- **Imbalanced Data**: Weighted sampling.

---

## 14. Summary

Ensemble methods like bagging and random forests are **powerful variance reducers** foundational to ML. In 2025, their role in explainable AI, edge computing, and LLM hybrids keeps them vital. Decorrelation and aggregation address single tree limitations.

<!-- **Next**: Explore [Boosting & Gradient Boosted Trees](/core-ml/boosting) or revisit [Decision Trees](/core-ml/decision-trees). -->

---

## Further Reading
- Breiman, "Random Forests" (2001).
- Hastie, *Elements of Statistical Learning* (Ch. 15).
- `linfa-ensemble` docs: [github.com/rust-ml/linfa](https://github.com/rust-ml/linfa).
- Friedman, "Stochastic Gradient Boosting" (2002).

---