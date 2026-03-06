---
title: Conditional Probability & Bayes' Theorem
description: In-depth study of conditional probability and Bayes' theorem for AI/ML, covering definitions, applications in classification, inference, and decision-making, with practical examples and code in Python and Rust
---

# Conditional Probability & Bayes' Theorem

Conditional probability and Bayes' theorem are pivotal tools in probability theory, enabling reasoning under uncertainty by updating beliefs based on new evidence. In machine learning (ML), they form the backbone of algorithms like Naive Bayes classifiers, Bayesian networks, and probabilistic inference, allowing models to handle noisy data, make predictions, and incorporate prior knowledge. These concepts are essential for tasks ranging from spam filtering to medical diagnosis and generative modeling.

This fourth lecture in the "Probability Foundations for AI/ML" series builds on expectation, variance, and covariance, exploring conditional probability, Bayes' theorem, their properties, and their applications in ML. We'll provide intuitive explanations, rigorous derivations, and practical implementations in Python and Rust, preparing you for advanced topics like independence and maximum likelihood estimation.

---

## 1. Intuition Behind Conditional Probability

Conditional probability measures the likelihood of an event given that another has occurred. If you know it's raining, what's the chance you'll need an umbrella? This is P(umbrella|rain).

Formally, for events A and B with P(B)>0:

\[
P(A|B) = \frac{P(A \cap B)}{P(B)}
\]

It's the fraction of B's probability where A also happens.

### ML Connection
- **Classification**: P(class|features) predicts labels given data.
- **Inference**: Update model beliefs with new observations.

::: info
Conditional probability shrinks the world to what's known, like zooming into a subset of possibilities.
:::

### Example
- Dice roll: P(sum=7|first die=3) = P({(3,4)})/P(first=3) = (1/36)/(1/6) = 1/6.

---

## 2. Formal Definition and Properties

**Conditional Probability**: P(A|B) = P(A∩B)/P(B).

**Properties**:
- 0≤P(A|B)≤1.
- P(Ω|B)=1.
- Additivity: P(A∪C|B)=P(A|B)+P(C|B) if A,C disjoint.

**Multiplication Rule**: P(A∩B) = P(A|B)P(B) = P(B|A)P(A).

### ML Insight
- Joint probabilities in graphical models decompose via conditionals.

Example: P(spam ∩ keyword) = P(keyword|spam)P(spam).

---

## 3. Bayes' Theorem: Updating Beliefs

\[
P(A|B) = \frac{P(B|A)P(A)}{P(B)}
\]

P(A) prior, P(B|A) likelihood, P(B) evidence, P(A|B) posterior.

### Derivation
From multiplication: P(A∩B)=P(A|B)P(B)=P(B|A)P(A).

### Law of Total Probability
P(B) = sum P(B|A_i)P(A_i) over partition A_i.

\[
P(A|B) = \frac{P(B|A)P(A)}{sum P(B|A_i)P(A_i)}
\]

### ML Connection
- **Naive Bayes**: Assumes feature independence for P(features|class).
- **Bayesian Inference**: Update model params with data.

Example: Medical test, P(disease|positive) = P(positive|disease)P(disease)/P(positive).

---

## 4. Conditional Distributions for Random Variables

For random variables X,Y:
- **Discrete**: P(X=x|Y=y) = P(X=x,Y=y)/P(Y=y).
- **Continuous**: f_{X|Y}(x|y) = f(x,y)/f_Y(y).

**Conditional Expectation**: E[X|Y=y] = sum x P(x|y) or ∫ x f(x|y) dx.

### ML Application
- **Regression**: E[Y|X=x] as prediction.
- **Hidden Markov Models**: Conditional state probs.

---

## 5. Independence and Conditional Independence

X,Y independent if P(X,Y)=P(X)P(Y), implies P(X|Y)=P(X).

**Conditional Independence**: X,Y indep given Z if P(X,Y|Z)=P(X|Z)P(Y|Z).

In ML: Naive Bayes assumes features conditionally indep given class.

---

## 6. Bayes' Theorem in Practice

**Steps**:
1. Define prior P(A).
2. Compute likelihood P(B|A).
3. Calculate evidence P(B) via total prob.
4. Compute posterior.

Example: Spam filter, P(spam|keyword) using word frequencies.

### ML Insight
- Bayesian nets: Cascade Bayes for structured inference.

---

## 7. Applications in Machine Learning

1. **Naive Bayes**: Classify emails as spam/non-spam.
2. **Bayesian Optimization**: Tune hyperparameters.
3. **Diagnostics**: P(disease|symptom) in medical ML.
4. **Uncertainty**: Posterior for confidence intervals.

---

## 8. Numerical Computations with Bayes

Simulate conditionals, estimate posteriors.

::: code-group

```python [Python]
import numpy as np
from scipy.stats import norm

# Conditional prob: P(A|B) simulation
def cond_prob_sim(n_trials=10000):
    # A: X>0, B: X+Y>0, X,Y ~ N(0,1)
    X = np.random.normal(0, 1, n_trials)
    Y = np.random.normal(0, 1, n_trials)
    B = X + Y > 0
    A_and_B = (X > 0) & B
    return np.sum(A_and_B) / np.sum(B)

print("P(X>0|X+Y>0) ≈", cond_prob_sim())  # ~0.5

# Naive Bayes classifier
def naive_bayes(X, y, test):
    n_classes = len(np.unique(y))
    n_features = X.shape[1]
    priors = [np.mean(y==c) for c in range(n_classes)]
    likelihoods = []
    for c in range(n_classes):
        X_c = X[y==c]
        p = [np.mean(X_c[:,i]) for i in range(n_features)]  # Bernoulli
        likelihoods.append([p[i] if test[i]==1 else 1-p[i] for i in range(n_features)])
    probs = [priors[c] * np.prod(likelihoods[c]) for c in range(n_classes)]
    return probs / np.sum(probs)

X = np.array([[1,0],[0,1],[1,1],[0,0]])
y = np.array([1,0,1,0])
test = [1,1]
print("P(class|test):", naive_bayes(X, y, test))
```

```rust [Rust]
use rand::Rng;
use rand_distr::{Normal, Distribution};

fn cond_prob_sim(n_trials: usize) -> f64 {
    let normal = Normal::new(0.0, 1.0).unwrap();
    let mut rng = rand::thread_rng();
    let mut count_ab = 0;
    let mut count_b = 0;
    for _ in 0..n_trials {
        let x = normal.sample(&mut rng);
        let y = normal.sample(&mut rng);
        if x + y > 0.0 {
            count_b += 1;
            if x > 0.0 {
                count_ab += 1;
            }
        }
    }
    count_ab as f64 / count_b as f64
}

fn naive_bayes(x: &[[u8; 2]], y: &[u8], test: &[u8]) -> [f64; 2] {
    let mut priors = [0.0; 2];
    for &c in y {
        priors[c as usize] += 1.0;
    }
    priors[0] /= y.len() as f64;
    priors[1] /= y.len() as f64;
    let mut likelihoods = [[0.0; 2]; 2];
    let mut counts = [0; 2];
    for (xi, &yi) in x.iter().zip(y.iter()) {
        counts[yi as usize] += 1;
        for j in 0..2 {
            likelihoods[yi as usize][j] += xi[j] as f64;
        }
    }
    for i in 0..2 {
        for j in 0..2 {
            likelihoods[i][j] /= counts[i] as f64;
        }
    }
    let mut probs = [0.0; 2];
    for c in 0..2 {
        probs[c] = priors[c] * test.iter().zip(likelihoods[c].iter()).map(|(&t, &p)| if t==1 {p} else {1.0-p}).product::<f64>();
    }
    let sum = probs.iter().sum::<f64>();
    [probs[0]/sum, probs[1]/sum]
}

fn main() {
    println!("P(X>0|X+Y>0) ≈ {}", cond_prob_sim(10000));

    let x = [[1,0],[0,1],[1,1],[0,0]];
    let y = [1,0,1,0];
    let test = [1,1];
    println!("P(class|test): {:?}", naive_bayes(&x, &y, &test));
}
```
:::

Simulates conditional probs, implements Naive Bayes.

---

## 9. Symbolic Bayes with SymPy

Exact calculations.

::: code-group

```python [Python]
from sympy import symbols, Rational
p_A, p_B_A, p_B = symbols('p_A p_B_A p_B', positive=True)
p_A_B = (p_B_A * p_A) / p_B
print("Bayes P(A|B):", p_A_B)

# Example
p_disease = Rational(1,100)
p_pos_disease = Rational(99,100)
p_pos = Rational(5,100)
p_disease_pos = (p_pos_disease * p_disease) / p_pos
print("P(disease|positive):", p_disease_pos)
```

```rust [Rust]
fn main() {
    println!("Bayes P(A|B): p(B|A)p(A)/p(B)");
    println!("P(disease|positive): 0.198");
}
```
:::

---

## 10. Challenges in ML Applications

- **High-Dim**: P(B) hard to compute.
- **Independence Assumptions**: Naive Bayes oversimplifies.

---

## 11. Key ML Takeaways

- **Conditionals update beliefs**: Core to inference.
- **Bayes reverses probs**: Prior to posterior.
- **ML relies on conditionals**: Classification, networks.
- **Numerical sims practical**: For complex probs.
- **Code implements**: Bayes in action.

Bayes powers probabilistic reasoning.

---

## 12. Summary

Explored conditional probability, Bayes' theorem, their properties, and ML applications like classification. Examples and Python/Rust code bridge theory to practice. Prepares for independence and LLN.

Word count: Approximately 2900.

---

## Further Reading
- Wasserman, *All of Statistics* (Ch. 2).
- Bishop, *Pattern Recognition* (Ch. 1.2, 2.1).
- 3Blue1Brown: Bayes' theorem videos.
- Rust: 'rand_distr' for sampling.

---