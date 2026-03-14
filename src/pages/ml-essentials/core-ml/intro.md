---
title: Introduction to Machine Learning
description: A detailed introduction to machine learning concepts, types, workflows, and real-world applications.
layout: ../../../layouts/TutorialPage.astro
---

# Introduction to Machine Learning

Machine Learning (ML) is one of the most important fields in modern computer science and artificial intelligence. It enables systems to **learn from data** instead of being explicitly programmed with rules. This first lecture in the *Core Machine Learning* series sets the stage for deeper exploration of algorithms, math foundations, and applications.

---

## 1. What is Machine Learning?

Arthur Samuel defined ML in 1959 as:  
> *“The field of study that gives computers the ability to learn without being explicitly programmed.”*

Tom Mitchell gave a more formal definition in 1997:  
> *“A computer program is said to learn from experience E with respect to some task T and some performance measure P, if its performance on T, as measured by P, improves with experience E.”*

**Key components of ML definition:**
- **Task (T):** What problem is the algorithm solving? (e.g., classify emails as spam or not).
- **Experience (E):** The data it learns from (e.g., labeled emails).
- **Performance Measure (P):** How success is measured (e.g., classification accuracy).

### What ML is NOT
- It is **not magic** – it requires quality data and good features.  
- It is **not rule-based programming** – no fixed if-else rules.  
- It is **not human-level intelligence** – though it powers parts of AI.

---

## 2. Why is Machine Learning Important?

ML underpins nearly every modern AI system:
- **Recommendation systems** (YouTube, Netflix, Amazon).  
- **Search engines** (Google uses ML ranking).  
- **Speech recognition & NLP** (Siri, Alexa, GPT).  
- **Computer vision** (self-driving cars, medical imaging).  
- **Fraud detection** (banking, cybersecurity).  

### The Explosion of ML
Three factors caused ML’s rapid growth:
1. **Data availability** (Big Data, IoT).  
2. **Computational power** (GPUs, TPUs, cloud).  
3. **Algorithmic advances** (deep learning, transformers).  

---

## 3. Types of Machine Learning

### 3.1 Supervised Learning
- Learns from **labeled data** (inputs + correct outputs).  
- Goal: Predict output for new unseen inputs.  
- Examples: Linear regression, logistic regression, decision trees, SVMs, neural networks.  

**Formula (Regression):**
$$
y = f(x) + \epsilon
$$
where $f(x)$ is the learned function, $y$ is the output, and $\epsilon$ is noise.

::: info Explanation
Supervised learning assumes a mapping from features $x$ to labels $y$. The model generalizes this mapping from known examples.
:::

### 3.2 Unsupervised Learning
- Learns from **unlabeled data** (only inputs, no outputs).  
- Goal: Find hidden structure in data.  
- Examples: Clustering (k-means), dimensionality reduction (PCA).  

### 3.3 Reinforcement Learning (RL)
- Learns by interacting with an environment.  
- Receives **rewards** or **penalties** as feedback.  
- Examples: AlphaGo, robotics, recommendation personalization.  

### 3.4 Semi-Supervised & Self-Supervised Learning
- **Semi-supervised:** Mix of labeled and unlabeled data.  
- **Self-supervised:** Learns signals from data itself (e.g., predicting missing words → GPT training).  

---

## 4. Typical ML Workflow

1. **Problem Definition** – What is being predicted?  
2. **Data Collection** – Gather data (structured/unstructured).  
3. **Data Preprocessing** – Cleaning, handling missing values, normalization.  
4. **Feature Engineering** – Selecting or creating informative variables.  
5. **Model Selection** – Choosing appropriate algorithms.  
6. **Training** – Optimizing model parameters on training data.  
7. **Evaluation** – Testing on unseen data, using metrics.  
8. **Deployment** – Serving models in production.  
9. **Monitoring & Maintenance** – Detecting drift, updating models.

---

## 5. The Role of Mathematics in ML

ML is powered by math, especially:
- **Linear Algebra** → Representing data (vectors, matrices).  
- **Calculus** → Optimization, gradients, backpropagation.  
- **Probability & Statistics** → Modeling uncertainty.  
- **Information Theory** → Loss functions, entropy, KL divergence.  

::: info Why math matters
Without math, ML becomes a black box. Understanding math ensures **interpretability**, **debugging ability**, and **better research contributions**.
:::

---

## 6. Minimal Code Example (Python & Rust)

::: code-group

```python [Python]
from sklearn.linear_model import LinearRegression
import numpy as np

# dataset
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2.2, 4.1, 6.0, 8.1, 9.9])

# train model
model = LinearRegression().fit(X, y)

# prediction
print("Prediction for 6:", model.predict(np.array([[6]]))[0])
```

```rust [Rust]
use linfa::prelude::*;
use linfa_linear::LinearRegression;
use ndarray::array;

fn main() {
    // dataset
    let x = array![[1.0], [2.0], [3.0], [4.0], [5.0]];
    let y = array![2.2, 4.1, 6.0, 8.1, 9.9];

    let dataset = Dataset::new(x, y);
    let model = LinearRegression::default().fit(&dataset).unwrap();

    let new_x = array![[6.0]];
    let prediction = model.predict(&new_x);
    println!("Prediction for 6: {}", prediction[0]);
}
```
:::

---

## 7. Real-World Case Studies

- **Finance:** Credit scoring with logistic regression.  
- **Healthcare:** Cancer detection using CNNs.  
- **Transportation:** Self-driving with deep RL.  
- **E-commerce:** Personalized recommendations with collaborative filtering.  
- **NLP:** GPT models for chat, summarization, translation.  

---

## 8. Connections to Next Lessons

This introduction lays the foundation. Next topics:  
- **Linear Regression** → First algorithm in detail.  
- **Logistic Regression** → Classification problems.  
- **Decision Trees** → Rule-based predictive modeling.  
- **Model Evaluation** → Metrics to judge performance.

---

## 9. Further Reading

- Tom Mitchell, *Machine Learning*.  
- Bishop, *Pattern Recognition and Machine Learning*.  
- Goodfellow et al., *Deep Learning*.  
- scikit-learn documentation.  
- Rust linfa documentation.

---

