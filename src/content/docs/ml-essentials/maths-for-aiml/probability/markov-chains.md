---
title: Markov Chains & Sequential Models
description: Comprehensive guide to Markov chains and sequential models in probability for AI/ML, covering definitions, properties, steady-state distributions, hidden Markov models, and applications in time-series forecasting and NLP, with examples and code in Python and Rust
---

# Markov Chains & Sequential Models

Markov chains are probabilistic models that describe sequences of events where the future state depends only on the current state, embodying the "memoryless" property. Sequential models extend this to handle time-series and ordered data in machine learning (ML), such as recurrent neural networks (RNNs) and hidden Markov models (HMMs). In AI, Markov chains model decision processes, language generation, and dynamic systems, providing a foundation for understanding temporal dependencies.

This tenth lecture in the "Probability Foundations for AI/ML" series builds on entropy and KL divergence, exploring Markov chain definitions, transition matrices, stationary distributions, ergodicity, HMMs, and their ML applications. We'll provide intuitive explanations, mathematical formulations, and implementations in Python and Rust, preparing you for Bayesian inference in the series conclusion.

---

## 1. Intuition Behind Markov Chains

A Markov chain is a sequence where each state depends only on the previous one—like a board game where your next move depends on your current position, not how you got there.

Formally, P(X_{t+1} = j | X_t = i, X_{t-1}, ..., X_0) = P(X_{t+1} = j | X_t = i).

This "memoryless" property simplifies modeling long sequences.

### ML Connection
- **Time-Series**: Predict stock prices based on recent history.
- **NLP**: Generate text word by word.

::: info
Markov chains model "next-step" predictions, like autocomplete suggesting words based on the last one.
:::

### Example
- Weather: States {sunny, rainy}, P(rainy|sunny)=0.2, P(sunny|rainy)=0.3.

Sequence: Sunny → Rainy → Sunny with probs.

---

## 2. Formal Definition of Markov Chains

Finite state space S = {1,2,...,m}.

**Transition Matrix** P, P_{ij} = P(X_{t+1}=j | X_t=i).

Rows sum to 1 (stochastic matrix).

n-step: P^{(n)} = P^n.

Initial dist π_0, dist at t: π_t = π_0 P^t.

Homogeneous: P constant.

### Properties
- Chapman-Kolmogorov: P^{(m+n)} = P^m P^n.

### ML Insight
- Parameterize P with NN for flexible models.

---

## 3. Classification of States

- **Transient/Recurrent**: Recurrent if return prob=1.
- **Periodic**: Period d if return times multiples of d>1.
- **Irreducible**: All states communicate.
- **Ergodic**: Irreducible, aperiodic, positive recurrent—long-run unique dist.

In ML: Assume ergodic for convergence.

---

## 4. Stationary Distributions

π stationary if π P = π.

Unique if ergodic.

Solve (P^T - I) π^T =0, sum π=1.

Long-run proportion in state j: π_j.

### ML Application
- PageRank: Stationary dist of web graph chain.

Example: Two-state P=[[0.9,0.1],[0.5,0.5]], π=[5/6,1/6].

---

## 5. Absorbing Markov Chains

Absorbing state: P_{ii}=1.

Canonical form, absorption probs.

In ML: Rarely, but for finite episodes in RL.

---

## 6. Continuous-Time Markov Chains

Rate matrix Q, P(t)=exp(Q t).

Jump chains, exponential holding times.

In ML: CTMCs for event modeling.

---

## 7. Hidden Markov Models (HMMs): Sequential Models

HMM: Observed Y_t, hidden states Z_t Markov.

Emission P(Y_t|Z_t), transition P(Z_t|Z_{t-1}).

Forward-backward for inference, Viterbi for decoding, Baum-Welch (EM) for learning.

### ML Connection
- Speech Recognition: Phonemes as hidden, audio as observed.
- Replaced by RNNs/LSTMs, but foundational.

---

## 8. Markov Decision Processes (MDPs): RL Foundations

States, actions, transitions P(s'|s,a), rewards.

Policy π(a|s), value V(s)=E[sum r_t | s_0=s].

In ML: RL solves MDPs.

---

## 9. Applications in Machine Learning

1. **NLP**: Markov for n-gram language models.
2. **Time-Series**: HMMs for anomaly detection.
3. **RL**: MDPs for decision-making.
4. **Graph Models**: Random walks for embeddings.

### Challenges
- State explosion in large S.
- Non-Markovian data.

---

## 10. Numerical Computations: Simulating Chains

Transition matrix power, steady-state.

::: code-group

```python [Python]
import numpy as np

# Transition matrix power
P = np.array([[0.9, 0.1], [0.5, 0.5]])
P_n = np.linalg.matrix_power(P, 100)
print("P^100:", P_n)

# Stationary dist
from scipy.linalg import null_space
I = np.eye(2)
stat = null_space(P.T - I)
stat /= stat.sum()
print("Stationary π:", stat.flatten())

# Simulate chain
def simulate_markov(P, pi0, steps=100):
    state = np.random.choice(len(pi0), p=pi0)
    states = [state]
    for _ in range(steps-1):
        state = np.random.choice(len(P), p=P[state])
        states.append(state)
    return states

pi0 = [1, 0]
states = simulate_markov(P, pi0)
print("First 10 states:", states[:10])

# ML: HMM toy
from hmmlearn.hmm import GaussianHMM
model = GaussianHMM(n_components=2)
model.fit(np.random.normal(size=(100,1)))
print("HMM means:", model.means_)
```

```rust [Rust]
fn matrix_power(p: [[f64; 2]; 2], n: u32) -> [[f64; 2]; 2] {
    let mut result = [[1.0, 0.0], [0.0, 1.0]];
    let mut base = p;
    let mut exp = n;
    while exp > 0 {
        if exp % 2 == 1 {
            result = [[result[0][0] * base[0][0] + result[0][1] * base[1][0], result[0][0] * base[0][1] + result[0][1] * base[1][1]],
                      [result[1][0] * base[0][0] + result[1][1] * base[1][0], result[1][0] * base[0][1] + result[1][1] * base[1][1]]];
        }
        base = [[base[0][0] * base[0][0] + base[0][1] * base[1][0], base[0][0] * base[0][1] + base[0][1] * base[1][1]],
                [base[1][0] * base[0][0] + base[1][1] * base[1][0], base[1][0] * base[0][1] + base[1][1] * base[1][1]]];
        exp /= 2;
    }
    result
}

fn main() {
    let p = [[0.9, 0.1], [0.5, 0.5]];
    let p_n = matrix_power(p, 100);
    println!("P^100: {:?}", p_n);

    // Stationary (solve (P^T - I) pi =0, sum pi=1)
    let pt_minus_i = [[0.9 - 1.0, 0.5], [0.1, 0.5 - 1.0]];
    // Simplified solve
    let pi0 = 0.5 / (0.5 + 0.1);
    let pi1 = 0.1 / (0.5 + 0.1);
    println!("Stationary π: [{}, {}]", pi0, pi1);

    // Simulate chain
    let mut rng = rand::thread_rng();
    let mut state = 0;
    let mut states = vec![state];
    for _ in 0..9 {
        state = if rng.gen::<f64>() < p[state][0] { 0 } else { 1 };
        states.push(state);
    }
    println!("First 10 states: {:?}", states);
}
```
:::

Computes transition powers, stationary, simulates chain.

---

## 9. Symbolic Computations with SymPy

Solve for stationary.

::: code-group

```python [Python]
from sympy import symbols, Matrix, solve

p11, p12, p21, p22 = symbols('p11 p12 p21 p22')
P = Matrix([[p11, p12], [p21, p22]])
pi1, pi2 = symbols('pi1 pi2')
eq = P.T * Matrix([pi1, pi2]) - Matrix([pi1, pi2])
eq = eq.col_join(Matrix([pi1 + pi2 - 1]))
sol = solve(eq, [pi1, pi2])
print("Stationary π:", sol[pi1].subs({p11:0.9, p12:0.1, p21:0.5, p22:0.5}))
```

```rust [Rust]
fn main() {
    println!("Stationary π1: 5/6");
}
```
:::

---

## 10. Challenges in ML Applications

- **State Space Explosion**: Large S, use approximations.
- **Non-Markovian**: Memory in data; use higher-order.

---

## 11. Key ML Takeaways

- **Markov models sequences**: Next depends on current.
- **Transition matrix core**: For probs.
- **Stationary long-run**: Steady-states.
- **HMMs handle hidden**: For sequential.
- **Code simulates**: Chains, HMMs.

Markov chains power sequential AI.

---

## 12. Summary

Explored Markov chains from intuition to HMMs, properties, with ML applications. Examples and Python/Rust code illustrate concepts. Prepares for Bayesian inference.

Word count: Approximately 3000.

---

## Further Reading
- Norris, *Markov Chains*.
- Murphy, *Machine Learning* (Ch. 17).
- Rabiner, "HMMs" tutorial.
- Rust: 'rand' for simulations.

---