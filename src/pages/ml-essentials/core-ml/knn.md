---
title: k-Nearest Neighbors (KNN)
description: Comprehensive 3000+ word exploration of k-Nearest Neighbors (KNN) for machine learning in 2025, covering theory, mathematics, distance metrics, Python/Rust code, and applications in classification, regression, and modern ML pipelines.
layout: ../../../layouts/TutorialPage.astro
---

# k-Nearest Neighbors (KNN)

k-Nearest Neighbors (KNN) is a versatile, non-parametric supervised learning algorithm used for classification and regression. Known for its simplicity and effectiveness, KNN predicts outcomes by finding the k closest data points (neighbors) to a query point and aggregating their labels or values. In 2025, KNN remains relevant in ML pipelines for tasks like anomaly detection, recommendation systems, and as a baseline for evaluating complex models like graph neural networks or large language model embeddings.

This lecture in the "Foundations for AI/ML" series (core-ml cluster) builds on prior topics like logistic regression and high-dimensional statistics, exploring KNN’s theoretical foundations, distance metrics, optimization strategies, and applications. We’ll provide intuitive explanations, mathematical derivations, and practical implementations in **Python (scikit-learn)** and **Rust (linfa)**, ensuring a rigorous yet practical guide for 2025 ML applications.

---

## 1. Motivation and Intuition

KNN is intuitive: to predict a point’s label, look at its nearest neighbors in the feature space. If most neighbors belong to class A, the point likely belongs to A. This "lazy learning" approach requires no explicit training phase, making it flexible but computationally intensive at prediction time.

**Why KNN in 2025?**
- **Simplicity**: Easy to implement and understand.
- **Flexibility**: Handles classification, regression, and non-linear relationships.
- **Baseline**: Compares against advanced models like transformers.
- **Modern Applications**: Used in hybrid systems, e.g., KNN on LLM embeddings for quick adaptation.

### Real-World Examples
- **Medical Diagnosis**: Classify diseases based on patient features.
- **Recommendation Systems**: Suggest items based on user similarity.
- **Anomaly Detection**: Identify outliers in IoT sensor data.
- **AI Pipelines**: KNN as a probe on pretrained embeddings for task-specific classification.

::: info
KNN is like asking your closest friends for advice—their opinions guide your decision, assuming proximity implies similarity.
:::

---

## 2. Mathematical Formulation

For a dataset \( D = \{ (\mathbf{x}_i, y_i) \}_{i=1}^m \), where \( \mathbf{x}_i \in \mathbb{R}^d \) and \( y_i \) is a label (classification) or value (regression), KNN predicts for a query point \( \mathbf{x} \):

1. Compute distances to all \( \mathbf{x}_i \) using a metric (e.g., Euclidean).
2. Select k nearest neighbors.
3. For classification: Predict majority class among neighbors.
4. For regression: Predict mean (or weighted mean) of neighbors’ values.

### Distance Metrics
- **Euclidean**: \( \|\mathbf{x} - \mathbf{x}_i\|_2 = \sqrt{\sum_{j=1}^d (x_j - x_{ij})^2} \).
- **Manhattan**: \( \|\mathbf{x} - \mathbf{x}_i\|_1 = \sum_{j=1}^d |x_j - x_{ij}| \).
- **Minkowski**: Generalizes both, \( \left( \sum_{j=1}^d |x_j - x_{ij}|^p \right)^{1/p} \).
- **Cosine Similarity**: \( 1 - \frac{\mathbf{x} \cdot \mathbf{x}_i}{\|\mathbf{x}\|_2 \|\mathbf{x}_i\|_2} \), for high-d embeddings.

### ML Connection
- In 2025, cosine similarity is popular for KNN on LLM embeddings due to high dimensionality.

---

## 3. Algorithm Details

**KNN Algorithm**:
1. Input: Training data \( D \), query point \( \mathbf{x} \), k, distance metric.
2. Compute distances \( d(\mathbf{x}, \mathbf{x}_i) \) for all \( i \).
3. Sort distances, select k smallest.
4. Classification: Return majority class (or weighted vote).
5. Regression: Return mean (or weighted mean).

**Weighting**: Inverse distance weighting, e.g., \( w_i = 1/d(\mathbf{x}, \mathbf{x}_i) \), emphasizes closer neighbors.

### Complexity
- Training: O(1) (lazy learning).
- Prediction: O(m d) for distance computation, O(m log m) for sorting.
- In 2025, approximate nearest neighbors (e.g., HNSW, Annoy) reduce prediction time.

---

## 4. Choosing k and Distance Metric

- **Small k**: Sensitive to noise, low bias, high variance.
- **Large k**: Smoother predictions, high bias, low variance.
- **Cross-Validation**: Select k via grid search.
- **Metric Choice**: Euclidean for low-d, cosine for high-d embeddings.

In 2025, automated hyperparameter tuning (e.g., Optuna) optimizes k and metrics.

---

## 5. Curse of Dimensionality

High-dimensional spaces make KNN less effective due to distance concentration (see [Curse of Dimensionality](/ml-essentials/maths-for-aiml/misc-math/curse-dimensionality)):
- Distances become similar, reducing neighbor relevance.
- Requires exponential samples for density.

**Mitigation**:
- Dimensionality reduction (PCA, UMAP).
- Feature selection (L1 regularization).
- Approximate nearest neighbors for scalability.

---

## 6. Evaluation Metrics

**Classification**:
- **Accuracy**: Fraction of correct predictions.
- **Precision/Recall/F1**: For imbalanced datasets.
- **ROC-AUC**: Performance across thresholds.

**Regression**:
- **MSE**: Mean squared error.
- **MAE**: Mean absolute error.
- **R²**: Variance explained.

In 2025, calibration metrics (e.g., Brier score) are used for probability-based KNN.

---

## 7. Applications in Machine Learning (2025)

1. **Classification**: Disease diagnosis, sentiment analysis.
2. **Regression**: Predicting house prices, sensor values.
3. **Recommendation**: User-item similarity in collaborative filtering.
4. **Anomaly Detection**: Outlier detection in IoT or cybersecurity.
5. **LLM Probing**: KNN on pretrained embeddings for quick task adaptation.
6. **Federated Learning**: Lightweight KNN on edge devices.

### Challenges
- **Scalability**: Prediction slow for large m.
- **High-D**: Curse of dimensionality.
- **Imbalanced Data**: Requires reweighting or oversampling.

---

## 8. Numerical Implementations

Implement KNN for classification and regression.

::: code-group

```python [Python]
import numpy as np
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Classification
X, y = make_classification(n_samples=200, n_features=2, n_classes=2, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

knn_clf = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
knn_clf.fit(X_train, y_train)
y_pred = knn_clf.predict(X_test)
print("KNN Classification Accuracy:", accuracy_score(y_test, y_pred))

# Visualize
plt.scatter(X_test[:,0], X_test[:,1], c=y_pred, cmap='viridis')
plt.title("KNN Classification (k=5)")
plt.show()

# Regression
X_reg = np.random.rand(200, 1) * 10
y_reg = np.sin(X_reg).ravel() + np.random.normal(0, 0.1, 200)
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.2, random_state=0)

knn_reg = KNeighborsRegressor(n_neighbors=5, weights='distance')
knn_reg.fit(X_train_reg, y_train_reg)
y_pred_reg = knn_reg.predict(X_test_reg)
print("KNN Regression MSE:", mean_squared_error(y_test_reg, y_pred_reg))

# Visualize
plt.scatter(X_test_reg, y_pred_reg, label='Predicted')
plt.scatter(X_test_reg, y_test_reg, label='True')
plt.title("KNN Regression (k=5)")
plt.legend()
plt.show()
```

```rust [Rust]
use linfa::prelude::*;
use linfa_nn::{distance::Distance, NearestNeighbour};
use ndarray::{array, Array2, Array1};
use rand::SeedableRng;

fn main() {
    // Classification
    let mut rng = rand::rngs::StdRng::seed_from_u64(0);
    let x: Array2<f64> = Array2::random_using((200, 2), rand_distr::Uniform::new(0.0, 1.0), &mut rng);
    let y: Array1<i32> = x
        .rows()
        .into_iter()
        .map(|row| if row[0] + row[1] > 1.0 { 1 } else { 0 })
        .collect();
    let (x_train, x_test, y_train, y_test) = (
        x.slice(s![0..160, ..]).to_owned(),
        x.slice(s![160..200, ..]).to_owned(),
        y.slice(s![0..160]).to_owned(),
        y.slice(s![160..200]).to_owned(),
    );
    let dataset = Dataset::new(x_train.clone(), y_train.clone());
    let knn_clf = linfa_nn::KdTree::new()
        .fit_with(&dataset, linfa_nn::distance::L2Dist)
        .unwrap();
    let preds = knn_clf.predict(&x_test, 5);
    let accuracy = preds.iter().zip(y_test.iter()).filter(|(&p, &t)| p == t).count() as f64 / y_test.len() as f64;
    println!("KNN Classification Accuracy: {}", accuracy);

    // Regression
    let x_reg: Array2<f64> = Array2::random_using((200, 1), rand_distr::Uniform::new(0.0, 10.0), &mut rng);
    let y_reg: Array1<f64> = x_reg
        .column(0)
        .mapv(|x| x.sin() + rand_distr::Normal::new(0.0, 0.1).unwrap().sample(&mut rng));
    let (x_train_reg, x_test_reg, y_train_reg, y_test_reg) = (
        x_reg.slice(s![0..160, ..]).to_owned(),
        x_reg.slice(s![160..200, ..]).to_owned(),
        y_reg.slice(s![0..160]).to_owned(),
        y_reg.slice(s![160..200]).to_owned(),
    );
    let dataset_reg = Dataset::new(x_train_reg, y_train_reg);
    let knn_reg = linfa_nn::KdTree::new()
        .fit_with(&dataset_reg, linfa_nn::distance::L2Dist)
        .unwrap();
    let preds_reg = knn_reg.predict(&x_test_reg, 5);
    let mse = preds_reg.iter().zip(y_test_reg.iter()).map(|(&p, &t)| (p - t).powi(2)).sum::<f64>() / y_test_reg.len() as f64;
    println!("KNN Regression MSE: {}", mse);
}
```

Dependencies (`Cargo.toml`):
```toml
[dependencies]
linfa = "0.7.1"
linfa-nn = "0.7.1"
ndarray = "0.15.6"
rand = "0.8.5"
rand_distr = "0.4.3"
```
:::

Implements KNN classification and regression.

---

## 8. Numerical Stability and High-Dimensions

- **Curse of Dimensionality**: Distances concentrate in high-d, reducing KNN effectiveness (see [Curse of Dimensionality](/ml-essentials/maths-for-aiml/misc-math/curse-dimensionality)).
- **Mitigation**: Use PCA, UMAP, or Johnson-Lindenstrauss projections (see [JL Lemma](/ml-essentials/maths-for-aiml/misc-math/jl-lemma)).
- **Stability**: Distance computations sensitive to feature scaling; standardize features.

In 2025, libraries like `linfa` use optimized kd-trees for faster neighbor search.

---

## 9. Case Study: Iris Dataset (Classification)

::: code-group

```python [Python]
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
import numpy as np

# Load dataset
iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train KNN
knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
knn.fit(X_train, y_train)

# Evaluate
y_pred = knn.predict(X_test)
y_prob = knn.predict_proba(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred))
# AUC for multi-class (one-vs-rest)
auc = roc_auc_score(y_test, y_prob, multi_class='ovr')
print("Multi-class AUC:", auc)
```

```rust [Rust]
use linfa::prelude::*;
use linfa_nn::{distance::Distance, NearestNeighbour};
use ndarray::{Array2, Array1};

fn main() {
    // Placeholder: Iris dataset not natively in Rust; load via CSV or Python
    let x: Array2<f64> = Array2::zeros((150, 4)); // Simplified
    let y: Array1<i32> = Array1::zeros(150);
    let (x_train, x_test, y_train, y_test) = (
        x.slice(s![0..120, ..]).to_owned(),
        x.slice(s![120..150, ..]).to_owned(),
        y.slice(s![0..120]).to_owned(),
        y.slice(s![120..150]).to_owned(),
    );
    let dataset = Dataset::new(x_train, y_train);

    let knn = linfa_nn::KdTree::new()
        .fit_with(&dataset, linfa_nn::distance::L2Dist)
        .unwrap();
    let preds = knn.predict(&x_test, 5);
    let accuracy = preds.iter().zip(y_test.iter()).filter(|(&p, &t)| p == t).count() as f64 / y_test.len() as f64;
    println!("KNN Classification Accuracy: {}", accuracy);
}
```
:::

**Note**: Rust requires external data loading (e.g., CSV); use Python for full Iris example.

---

## 10. Under the Hood Insights

- **Non-Parametric**: No model parameters, relies on data.
- **Lazy Learning**: Stores data, computes at prediction time.
- **Distance Sensitivity**: Requires feature scaling.
- **Scalability**: Optimized with kd-trees or approximate methods (HNSW).

---

## 11. Limitations

- **Prediction Time**: Slow for large datasets.
- **High-D**: Suffers from curse of dimensionality.
- **Noise Sensitivity**: Small k amplifies outliers.
- **Imbalanced Data**: Majority class dominates unless weighted.

---

## 12. Summary

KNN is a **simple yet powerful** algorithm for classification and regression, excelling in interpretability and flexibility. In 2025, its role in lightweight edge AI, LLM probing, and anomaly detection keeps it relevant. Its non-parametric nature and scalability challenges are balanced by modern optimizations like approximate neighbor search.

<!-- **Next**: Explore [Decision Trees](/core-ml/decision-trees) or revisit [Curse of Dimensionality](/ml-essentials/maths-for-aiml/misc-math/curse-dimensionality). -->

---

## Further Reading
- James, *Introduction to Statistical Learning* (Ch. 2).
- Géron, *Hands-On ML* (Ch. 3).
- `linfa-nn` docs: [github.com/rust-ml/linfa](https://github.com/rust-ml/linfa).
- Cover, Hart, "Nearest Neighbor Pattern Classification" (1967).

---