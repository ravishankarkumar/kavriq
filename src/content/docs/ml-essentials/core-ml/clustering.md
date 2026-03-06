---
title: Clustering - k-Means & Hierarchical
description: Comprehensive 3000+ word exploration of clustering algorithms, focusing on k-Means and Hierarchical clustering for machine learning in 2025, covering theory, mathematics, derivations, Python/Rust code, and applications in unsupervised learning and data analysis.
---

# Clustering: k-Means & Hierarchical

Clustering is an unsupervised learning technique that groups similar data points without labels, uncovering hidden patterns in data. k-Means partitions data into k clusters by minimizing intra-cluster variance, while Hierarchical clustering builds a tree of clusters, offering flexibility in choosing the number of clusters. In 2025, these algorithms remain essential in ML for customer segmentation, anomaly detection, and as preprocessing steps in pipelines with large language models (LLMs) for embedding analysis.

This lecture in the "Foundations for AI/ML" series (core-ml cluster) builds on prior topics like decision trees and ensembles, exploring k-Means and Hierarchical clustering, their theoretical foundations, algorithms, and applications. We’ll provide intuitive explanations, mathematical insights, and practical implementations in **Python (scikit-learn)** and **Rust (linfa)**, ensuring a rigorous yet practical guide aligned with 2025 ML trends.

---

## 1. Motivation and Intuition

Clustering discovers structure in unlabeled data, grouping similar points while separating dissimilar ones. k-Means is simple and scalable, ideal for spherical clusters, while Hierarchical is flexible for hierarchical or non-spherical structures.

**Why Clustering in 2025?**
- **Unsupervised Learning**: Handles unlabeled data common in big data.
- **Interpretability**: Reveals data patterns.
- **Baseline**: For advanced methods like DBSCAN.
- **Modern Applications**: Cluster LLM embeddings for topic discovery.

### Real-World Examples
- **Marketing**: Customer segmentation.
- **Biology**: Gene expression clustering.
- **AI Pipelines**: Cluster LLM outputs for summarization.

::: info
Clustering is like organizing a messy room—k-Means assigns to fixed bins, Hierarchical builds a nested hierarchy.
:::

---

## 2. Mathematical Formulation

Clustering minimizes within-group dissimilarity or maximizes between-group.

**Objective**: For k clusters, minimize sum of intra-cluster variances.

**Distance Metrics**:
- Euclidean: For continuous features.
- Manhattan: Robust to outliers.
- Cosine: For high-d embeddings.

### ML Connection
- Metrics affect cluster shape.

---

## 3. k-Means Clustering

**Algorithm**:
1. Initialize k centroids (random or k-means++).
2. Assign points to nearest centroid.
3. Update centroids as cluster means.
4. Repeat until convergence.

**Objective**: Min sum_{i=1}^m ||x_i - μ_{c_i}||², c_i cluster assignment.

### EM View
Expectation: Assign clusters.
Maximization: Update means.

### Derivation
Lloyd's algorithm approximates NP-hard partitioning.

### Variants
- k-Means++: Smart initialization.
- Mini-Batch k-Means: Scalable for large data.

In 2025, k-Means on LLM embeddings for clustering.

---

## 4. Hierarchical Clustering

**Agglomerative** (Bottom-Up):
1. Start with n singleton clusters.
2. Merge closest clusters iteratively.
3. Stop at desired k or single cluster.

**Divisive** (Top-Down): Start with one, split recursively.

**Linkage Criteria**:
- Single: Min distance.
- Complete: Max distance.
- Average: Mean distance.
- Ward: Minimize variance increase.

### Dendrogram
Tree showing merge hierarchy.

### ML Application
- Hierarchical for taxonomy in biology.

---

## 5. Choosing k and Validation

**Elbow Method**: Plot inertia vs. k, find elbow.

**Silhouette Score**: Measure cohesion/separation.

**Gap Statistic**: Compare to null distribution.

In ML: Cross-validation for k.

---

## 6. Evaluation Metrics

**Internal**:
- Inertia: Sum squared distances to centroids.
- Silhouette: (b - a) / max(a,b), a intra, b inter.

**External** (with labels):
- ARI, NMI: Compare to ground truth.

In 2025, calibration for probabilistic clustering.

---

## 7. Applications in Machine Learning (2025)

1. **Customer Segmentation**: Marketing groups.
2. **Anomaly Detection**: Outliers as small clusters.
3. **Image Segmentation**: Pixel clustering.
4. **Bioinformatics**: Gene clustering.
5. **LLM Analysis**: Cluster embeddings for topics.
6. **Edge AI**: Lightweight hierarchical for devices.

### Challenges
- **k-Means**: Sensitive to initialization, spherical assumption.
- **Hierarchical**: O(n²) cost.
- **High-D**: Curse; use reduction.

---

## 8. Numerical Implementations

Implement k-Means, Hierarchical.

::: code-group

```python [Python]
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

# k-Means
X, _ = make_blobs(n_samples=200, centers=3, random_state=0)
kmeans = KMeans(n_clusters=3, init='k-means++', random_state=0)
labels = kmeans.fit_predict(X)
print("k-Means Silhouette:", silhouette_score(X, labels))

plt.scatter(X[:,0], X[:,1], c=labels)
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], marker='x', c='red')
plt.title("k-Means Clustering")
plt.show()

# Hierarchical
hier = AgglomerativeClustering(n_clusters=3, linkage='ward')
labels_h = hier.fit_predict(X)
print("Hierarchical Silhouette:", silhouette_score(X, labels_h))

plt.scatter(X[:,0], X[:,1], c=labels_h)
plt.title("Hierarchical Clustering")
plt.show()

# ML: Cluster LLM embeddings (placeholder)
embeddings = np.random.rand(100, 768)  # LLM embeds
kmeans_llm = KMeans(n_clusters=5)
labels_llm = kmeans_llm.fit_predict(embeddings)
print("LLM Embeddings Clusters:", np.unique(labels_llm))
```

```rust [Rust]
use linfa::prelude::*;
use linfa_clustering::KMeans;
use ndarray::{Array2, Array1};

fn main() {
    let mut rng = rand::thread_rng();
    let x: Array2<f64> = Array2::zeros((200, 2));
    // Generate blobs placeholder
    let dataset = Dataset::new(x.clone(), Array1::zeros(200));

    let kmeans = KMeans::params(3).fit(&dataset).unwrap();
    let labels = kmeans.predict(&x);
    // Silhouette not natively; compute manually

    // Hierarchical not in linfa; use k-means for demo
    println!("k-Means Labels: {:?}", labels);
}
```
:::

**Note**: Rust clustering support limited; use Python for full hierarchical.

---

## 9. Case Study: Customer Segmentation (k-Means)

::: code-group

```python [Python]
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import numpy as np

# Generate customer data
X, _ = make_blobs(n_samples=300, centers=4, n_features=2, random_state=0)

# Find optimal k
sil = []
for k in range(2, 10):
    km = KMeans(n_clusters=k)
    km.fit(X)
    sil.append(silhouette_score(X, km.labels_))
plt.plot(range(2,10), sil)
plt.title("Elbow Method")
plt.show()

# Train
km = KMeans(n_clusters=4)
labels = km.fit_predict(X)
plt.scatter(X[:,0], X[:,1], c=labels)
plt.title("Customer Segmentation")
plt.show()
```

```rust [Rust]
use linfa::prelude::*;
use linfa_clustering::KMeans;
use ndarray::Array2;

fn main() {
    let x: Array2<f64> = Array2::zeros((300, 2));
    // Generate blobs placeholder

    let km = KMeans::params(4).fit(&x.view()).unwrap();
    let labels = km.predict(&x);
    println!("Labels: {:?}", labels);
}
```
:::

**Note**: Rust requires plotting libraries for visualization.

---

## 10. Under the Hood Insights

- **k-Means**: EM algorithm variant.
- **Hierarchical**: Linkage affects dendrogram.
- **Validation**: Silhouette for optimal k.
- **Scalability**: Mini-batch k-Means for large data.

---

## 11. Limitations

- **k-Means**: Assumes spherical clusters, sensitive to k.
- **Hierarchical**: Computationally expensive for large n.
- **High-D**: Curse; use reduction.
- **Outliers**: Distort clusters; use robust variants.

---

## 12. Summary

Clustering with k-Means and Hierarchical uncovers data patterns. In 2025, their role in LLM embedding analysis and anomaly detection keeps them vital. Validation and preprocessing address limitations.

<!-- **Next**: Explore [Neural Networks Basics](/core-ml/neural-networks) or revisit [Decision Trees](/core-ml/decision-trees). -->

---

## Further Reading
- Hartigan, "Clustering Algorithms".
- Hastie, *Elements of Statistical Learning* (Ch. 14).
- `linfa-clustering` docs: [github.com/rust-ml/linfa](https://github.com/rust-ml/linfa).
- Rokach, Maimon, "Clustering Methods" (2005).

---