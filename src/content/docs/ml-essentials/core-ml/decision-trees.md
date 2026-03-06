---
title: Decision Trees
description: Comprehensive 3000+ word exploration of decision trees for machine learning in 2025, covering theory, mathematics, splitting criteria, Python/Rust code, and applications in classification, regression, and ensemble methods.
---

# Decision Trees

Decision trees are versatile supervised learning algorithms that model decisions through a tree-like structure of if-then rules, used for both classification and regression. Known for interpretability and handling nonlinear relationships, decision trees form the basis of ensemble methods like random forests. In 2025, they remain essential in ML for feature importance, explainable AI, and as building blocks in hybrid systems with large language models.

This lecture in the "Foundations for AI/ML" series (core-ml cluster) builds on k-NN and logistic regression, exploring decision trees, their theoretical foundations, splitting criteria, pruning, and applications. We’ll provide intuitive explanations, mathematical insights, and practical implementations in **Python (scikit-learn)** and **Rust (linfa)**, ensuring a rigorous yet practical guide aligned with 2025 ML trends.

---

## 1. Motivation and Intuition

Decision trees mimic human decision-making: start with a question, branch based on the answer, repeat until a conclusion. For ML, they split data based on features to minimize impurity, creating interpretable rules.

**Why Decision Trees in 2025?**
- **Interpretability**: Explain predictions with rules.
- **Nonlinearity**: Handle complex relationships without transformations.
- **Baseline**: For ensemble methods like XGBoost.
- **Modern Applications**: Feature importance in LLMs, edge AI.

### Real-World Examples
- **Medical Diagnosis**: Tree based on symptoms for disease prediction.
- **Finance**: Credit approval based on income, age.
- **AI Pipelines**: Trees on LLM embeddings for quick decisions.

::: info
Decision trees are like a flowchart for ML—simple questions lead to powerful predictions, with branches capturing data complexity.
:::

---

## 2. Mathematical Formulation

A decision tree partitions feature space into regions R_m, assigning prediction c_m to each.

For regression: c_m = mean y in R_m.

For classification: c_m = majority class in R_m.

**Tree Structure**:
- Nodes: Splitting features/thresholds.
- Leaves: Predictions.

### Splitting Criteria
- **Regression**: Minimize MSE: sum (y_i - c_m)^2.
- **Classification**:
  - **Gini Impurity**: G = sum p_k (1 - p_k).
  - **Entropy**: H = - sum p_k log p_k.
- Gain = parent impurity - weighted child impurities.

### ML Connection
- Gini/entropy measure node purity.

---

## 3. Building the Tree: Recursive Splitting

**Algorithm**:
1. Start with all data at root.
2. For each feature/threshold, compute gain.
3. Split on best gain.
4. Recurse on child nodes until stopping criteria (depth, samples, gain).

**Stopping**: Max depth, min samples split/leaf, min gain.

### Derivation
Maximize gain to reduce impurity, approximating MLE for partitions.

---

## 4. Pruning: Preventing Overfitting

Grow full tree, prune branches with low gain.

**Cost-Complexity Pruning**: Minimize error + α branches.

In ML: Prune for generalization.

---

## 5. Decision Trees for Regression

**CART Regression**: Split to minimize MSE.

Prediction: Mean of leaf.

In 2025, regression trees in ensembles for time-series.

---

## 6. Evaluation Metrics

**Classification**:
- Accuracy, Precision/Recall/F1, ROC-AUC.

**Regression**:
- MSE, MAE, R².

**Feature Importance**: Gain or split count.

In 2025, SHAP for tree explainability.

---

## 7. Applications in Machine Learning (2025)

1. **Classification**: Fraud detection, sentiment analysis.
2. **Regression**: House price prediction.
3. **Feature Importance**: SHAP on trees for LLM interpretability.
4. **Ensemble Building**: Base for RF, XGBoost.
5. **Edge AI**: Lightweight trees on devices.
6. **Hybrid Systems**: Trees on LLM features for efficient inference.

### Challenges
- **Overfitting**: Requires pruning/ensembles.
- **Instability**: Small data changes alter tree.
- **Bias**: Axis-aligned splits miss oblique patterns.

---

## 8. Numerical Implementations

Implement decision trees for classification/regression.

::: code-group

```python [Python]
import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

# Classification
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=0)

dt_clf = DecisionTreeClassifier(max_depth=3, criterion='gini', random_state=0)
dt_clf.fit(X_train, y_train)
y_pred = dt_clf.predict(X_test)
print("Decision Tree Classification Accuracy:", accuracy_score(y_test, y_pred))

# Visualize tree
plt.figure(figsize=(12,8))
plot_tree(dt_clf, feature_names=iris.feature_names, class_names=iris.target_names, filled=True)
plt.title("Decision Tree Classification")
plt.show()

# Regression
X_reg = np.random.rand(200, 1) * 10
y_reg = np.sin(X_reg).ravel() + np.random.normal(0, 0.1, 200)
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.2, random_state=0)

dt_reg = DecisionTreeRegressor(max_depth=3, criterion='squared_error', random_state=0)
dt_reg.fit(X_train_reg, y_train_reg)
y_pred_reg = dt_reg.predict(X_test_reg)
print("Decision Tree Regression MSE:", mean_squared_error(y_test_reg, y_pred_reg))

# Visualize
plt.scatter(X_test_reg, y_test_reg, label='True')
plt.scatter(X_test_reg, y_pred_reg, label='Predicted')
plt.title("Decision Tree Regression")
plt.legend()
plt.show()
```

```rust [Rust]
use linfa::prelude::*;
use linfa_trees::DecisionTree;
use ndarray::{array, Array2, Array1};

fn main() {
    // Classification (placeholder Iris)
    let x_train: Array2<f64> = Array2::zeros((120, 4));
    let y_train: Array1<i32> = Array1::zeros(120);
    let x_test: Array2<f64> = Array2::zeros((30, 4));
    let y_test: Array1<i32> = Array1::zeros(30);

    let dataset = Dataset::new(x_train, y_train);
    let model = DecisionTree::params().fit(&dataset).unwrap();
    let preds = model.predict(&x_test);
    let accuracy = preds.iter().zip(y_test.iter()).filter(|(&p, &t)| p == t).count() as f64 / y_test.len() as f64;
    println!("Decision Tree Classification Accuracy: {}", accuracy);

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
    let model_reg = DecisionTree::params().fit(&dataset_reg).unwrap();
    let preds_reg = model_reg.predict(&x_test_reg);
    let mse = preds_reg.iter().zip(y_test_reg.iter()).map(|(&p, &t)| (p - t).powi(2)).sum::<f64>() / y_test_reg.len() as f64;
    println!("Decision Tree Regression MSE: {}", mse);
}
```

Dependencies (`Cargo.toml`):
```toml
[dependencies]
linfa = "0.7.1"
linfa-trees = "0.7.0"
ndarray = "0.15.6"
rand = "0.8.5"
rand_distr = "0.4.3"
```
:::

Implements decision trees for classification and regression.

---

## 8. Numerical Stability and High-Dimensions

- **Instability**: Trees sensitive to data changes; ensembles stabilize.
- **High-D**: Curse of dimensionality; use feature selection.
- **Overfitting**: Pruning essential.

In 2025, stability in federated trees is key for distributed ML.

---

## 9. Case Study: Iris Dataset (Classification)

::: code-group

```python [Python]
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, plot_tree
import matplotlib.pyplot as plt

# Load dataset
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

# Train with pruning
dt = DecisionTreeClassifier(max_depth=3, min_samples_leaf=5, criterion='entropy', random_state=0)
dt.fit(X_train, y_train)

# Evaluate
y_pred = dt.predict(X_test)
print(classification_report(y_test, y_pred))

# Visualize
plt.figure(figsize=(12,8))
plot_tree(dt, feature_names=iris.feature_names, class_names=iris.target_names, filled=True)
plt.title("Decision Tree on Iris")
plt.show()
```

```rust [Rust]
use linfa::prelude::*;
use linfa_trees::DecisionTree;
use ndarray::{Array2, Array1};

fn main() {
    // Placeholder: Iris dataset
    let x_train: Array2<f64> = Array2::zeros((120, 4));
    let y_train: Array1<i32> = Array1::zeros(120);
    let x_test: Array2<f64> = Array2::zeros((30, 4));
    let y_test: Array1<i32> = Array1::zeros(30);

    let dataset = Dataset::new(x_train, y_train);
    let model = DecisionTree::params().fit(&dataset).unwrap();
    let preds = model.predict(&x_test);
    let accuracy = preds.iter().zip(y_test.iter()).filter(|(&p, &t)| p == t).count() as f64 / y_test.len() as f64;
    println!("Decision Tree Accuracy: {}", accuracy);
}
```
:::

**Note**: Rust requires external data loading; use Python for full visualization.

---

## 10. Under the Hood Insights

- **Recursive Partitioning**: Greedy but effective.
- **Impurity Measures**: Gini vs. Entropy (Gini faster).
- **Pruning**: Cost-complexity balances accuracy and complexity.
- **Interpretability**: Rules extractable for explainable AI.

---

## 11. Limitations

- **Overfitting**: Full trees capture noise; pruning/ensembles needed.
- **Instability**: Small changes alter structure.
- **Axis-Aligned**: Misses oblique boundaries.
- **Imbalanced Data**: Biased toward majority class.

---

## 12. Summary

Decision trees are **interpretable, nonlinear classifiers/regressors** foundational to ML. In 2025, their role in ensembles, explainable AI, and edge computing keeps them vital. Pruning and regularization address limitations.

<!-- **Next**: Explore [Random Forests](/core-ml/random-forests) or revisit [Curse of Dimensionality](/ml-essentials/maths-for-aiml/misc-math/curse-dimensionality). -->

---

## Further Reading
- Breiman, "Statistical Modeling: The Two Cultures".
- Hastie, *Elements of Statistical Learning* (Ch. 9).
- `linfa-trees` docs: [github.com/rust-ml/linfa](https://github.com/rust-ml/linfa).
- Loh, "Fifty Years of Classification and Regression Trees" (2014).

---