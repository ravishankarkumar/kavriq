---
title: Recurrent Neural Networks
description: In-depth exploration of RNNs for sequence modeling
layout: ../../../layouts/TutorialPage.astro
---
# Recurrent Neural Networks

Recurrent Neural Networks (RNNs) are specialized neural networks for processing sequential data, such as time series or text, by maintaining a hidden state that captures temporal dependencies. This section provides a comprehensive exploration of RNN architecture, backpropagation through time (BPTT), and variants like Long Short-Term Memory (LSTM) units, with a Rust lab using `tch-rs`. We'll dive into sequence processing mechanics, gradient computation challenges, and Rust's performance advantages, building on convolutional neural networks.

## Theory

RNNs process a sequence $\mathbf{x} = [\mathbf{x}_1, \mathbf{x}_2, \dots, \mathbf{x}_T]$, where $\mathbf{x}_t \in \mathbb{R}^n$ is the input at time $t$, producing outputs $\mathbf{y} = [\mathbf{y}_1, \mathbf{y}_2, \dots, \mathbf{y}_T]$. The hidden state $\mathbf{h}_t \in \mathbb{R}^h$ is updated recursively:
$$
\mathbf{h}_t = g(\mathbf{W}_h \mathbf{h}_{t-1} + \mathbf{W}_x \mathbf{x}_t + \mathbf{b}_h)
$$
where $\mathbf{W}_h \in \mathbb{R}^{h \times h}$, $\mathbf{W}_x \in \mathbb{R}^{h \times n}$, $\mathbf{b}_h \in \mathbb{R}^h$ are parameters, and $g$ is a non-linear activation (e.g., tanh, $g(z) = \tanh(z)$). The output is:
$$
\mathbf{y}_t = \mathbf{W}_y \mathbf{h}_t + \mathbf{b}_y
$$
For classification, $\mathbf{y}_t$ is passed through a softmax to predict probabilities.

### Derivation: Backpropagation Through Time

RNNs are trained to minimize a loss, such as cross-entropy for sequence classification:
$$
J(\boldsymbol{\theta}) = -\frac{1}{m} \sum_{i=1}^m \sum_{t=1}^T \sum_{k=1}^K y_{itk} \log \hat{y}_{itk}
$$
where $\boldsymbol{\theta}$ includes $\mathbf{W}_h$, $\mathbf{W}_x$, $\mathbf{W}_y$, $\mathbf{b}_h$, $\mathbf{b}_y$, $y_{itk}$ is 1 if sample $i$ at time $t$ is class $k$, and $m$ is the batch size. **Backpropagation Through Time (BPTT)** computes gradients by unrolling the RNN over $T$ time steps, treating it as a deep feedforward network with shared weights.

For a single sample, the loss at time $t$ is $J_t$. The gradient for $\mathbf{W}_h$ is:
$$
\frac{\partial J}{\partial \mathbf{W}_h} = \sum_{t=1}^T \frac{\partial J_t}{\partial \mathbf{h}_t} \cdot \frac{\partial \mathbf{h}_t}{\partial \mathbf{W}_h}
$$
The error term is:
$$
\boldsymbol{\delta}_t = \frac{\partial J_t}{\partial \mathbf{h}_t} = \left( \mathbf{W}_y^T \frac{\partial J_t}{\partial \mathbf{y}_t} + \mathbf{W}_h^T \boldsymbol{\delta}_{t+1} \right) \cdot g'(\mathbf{z}_t)
$$
where $\mathbf{z}_t = \mathbf{W}_h \mathbf{h}_{t-1} + \mathbf{W}_x \mathbf{x}_t + \mathbf{b}_h$, and $g'$ is the activation derivative (e.g., for tanh, $g'(z) = 1 - \tanh^2(z)$). The weight gradient is:
$$
\frac{\partial J_t}{\partial \mathbf{W}_h} = \boldsymbol{\delta}_t \cdot \mathbf{h}_{t-1}^T
$$
Gradients are summed over time steps and averaged over the batch.

**Under the Hood**: BPTT unrolls the RNN, creating a deep computational graph, costing $O(T h^2)$ per sample for $h$ hidden units. Long sequences cause **vanishing gradients** (gradients shrink exponentially) or **exploding gradients** (grow uncontrollably). `tch-rs` mitigates this with gradient clipping, leveraging Rust's memory safety to prevent tensor corruption during unrolling, unlike C++ where pointer errors risk crashes. Rust's compiled performance outpaces Python's `pytorch` for CPU-bound BPTT, reducing latency.

## LSTM: Addressing Gradient Issues

**Long Short-Term Memory (LSTM)** units address vanishing gradients by introducing gates to control information flow:
- **Forget Gate**: $\mathbf{f}_t = \sigma(\mathbf{W}_f \mathbf{h}_{t-1} + \mathbf{U}_f \mathbf{x}_t + \mathbf{b}_f)$
- **Input Gate**: $\mathbf{i}_t = \sigma(\mathbf{W}_i \mathbf{h}_{t-1} + \mathbf{U}_i \mathbf{x}_t + \mathbf{b}_i)$
- **Cell Update**: $\tilde{\mathbf{c}}_t = \tanh(\mathbf{W}_c \mathbf{h}_{t-1} + \mathbf{U}_c \mathbf{x}_t + \mathbf{b}_c)$
- **Cell State**: $\mathbf{c}_t = \mathbf{f}_t \cdot \mathbf{c}_{t-1} + \mathbf{i}_t \cdot \tilde{\mathbf{c}}_t$
- **Output Gate**: $\mathbf{o}_t = \sigma(\mathbf{W}_o \mathbf{h}_{t-1} + \mathbf{U}_o \mathbf{x}_t + \mathbf{b}_o)$
- **Hidden State**: $\mathbf{h}_t = \mathbf{o}_t \cdot \tanh(\mathbf{c}_t)$

The cell state $\mathbf{c}_t$ retains long-term dependencies, with gates modulating updates. Gradients flow through additive updates, avoiding vanishing issues.

**Under the Hood**: LSTMs increase computational cost ($O(T h^2)$ per gate) but improve training stability. `tch-rs` optimizes gate computations with vectorized operations, ensuring memory efficiency via Rust's ownership model, unlike Python's dynamic allocation, which can fragment memory for long sequences.

## Optimization

RNNs are trained with BPTT and optimizers like **Adam**, minimizing the loss. Regularization (e.g., dropout, $L_2$ penalty) prevents overfitting. Truncated BPTT limits unrolling to $T' < T$ steps, balancing accuracy and computation.

**Under the Hood**: Truncated BPTT reduces memory usage but may miss long-term dependencies. `tch-rs` implements efficient truncation, leveraging Rust's zero-cost abstractions for performance, outpacing Python's `pytorch` for CPU tasks. Rust's type safety ensures correct tensor shapes, preventing runtime errors common in C++ during sequence unrolling.

## Evaluation

Performance is evaluated with:

- **Classification**: Accuracy, Precision, Recall, F1-Score, ROC-AUC.
- **Regression**: MSE, RMSE, MAE.
- **Perplexity** (for language models): $\exp\left(-\frac{1}{T} \sum_{t=1}^T \log P(y_t | \mathbf{x}_t)\right)$.

**Under the Hood**: Perplexity measures sequence prediction quality, with lower values indicating better models. `tch-rs` computes metrics efficiently, using GPU acceleration when available, with Rust's compiled performance reducing overhead compared to Python's interpreter.

## Lab: LSTM with `tch-rs`

You'll train an LSTM on a synthetic sequence dataset for binary classification, evaluating accuracy and loss.

1. **Edit** `src/main.rs` in your `rust_ml_tutorial` project:
    ```rust
    use tch::{nn, nn::Module, nn::OptimizerConfig, Device, Tensor};
    use ndarray::{array, Array3, Array2};

    fn main() -> Result<(), tch::TchError> {
        // Synthetic dataset: 10 sequences, 5 time steps, 2 features
        let x: Array3<f64> = array![
            // Class 0: low values
            [[0.1, 0.2], [0.2, 0.3], [0.1, 0.2], [0.3, 0.4], [0.2, 0.3]],
            [[0.2, 0.1], [0.3, 0.2], [0.2, 0.1], [0.4, 0.3], [0.3, 0.2]],
            [[0.1, 0.3], [0.2, 0.4], [0.1, 0.3], [0.3, 0.2], [0.2, 0.1]],
            [[0.3, 0.2], [0.4, 0.3], [0.3, 0.2], [0.2, 0.1], [0.1, 0.2]],
            [[0.2, 0.3], [0.1, 0.2], [0.2, 0.3], [0.1, 0.4], [0.3, 0.2]],
            // Class 1: high values
            [[0.9, 0.8], [0.8, 0.9], [0.9, 0.8], [0.7, 0.9], [0.8, 0.7]],
            [[0.8, 0.9], [0.9, 0.8], [0.8, 0.7], [0.9, 0.8], [0.7, 0.9]],
            [[0.7, 0.8], [0.8, 0.9], [0.9, 0.7], [0.8, 0.9], [0.9, 0.8]],
            [[0.9, 0.7], [0.8, 0.8], [0.7, 0.9], [0.9, 0.8], [0.8, 0.7]],
            [[0.8, 0.9], [0.9, 0.8], [0.8, 0.9], [0.7, 0.8], [0.9, 0.7]],
        ];
        let y: Array2<f64> = array![[0.0], [0.0], [0.0], [0.0], [0.0], [1.0], [1.0], [1.0], [1.0], [1.0]];

        // Convert to tensors
        let device = Device::Cpu;
        let xs = Tensor::from_slice(x.as_slice().unwrap()).to_device(device).reshape(&[10, 5, 2]);
        let ys = Tensor::from_slice(y.as_slice().unwrap()).to_device(device).reshape(&[10, 1]);

        // Define LSTM
        let vs = nn::VarStore::new(device);
        let lstm_config = nn::LSTMConfig { hidden_size: 10, num_layers: 1, ..Default::default() };
        let net = nn::seq()
            .add(nn::lstm(&vs.root() / "lstm", 2, 10, lstm_config))
            .add_fn(|xs| xs.slice(1, 4, 5, 1)) // Take last time step
            .add(nn::linear(&vs.root() / "fc", 10, 1, Default::default()))
            .add_fn(|xs| xs.sigmoid());

        // Optimizer (Adam)
        let mut opt = nn::Adam::default().build(&vs, 0.01)?;

        // Training loop
        for epoch in 1..=100 {
            let logits = net.forward(&xs);
            let loss = logits.binary_cross_entropy_with_logits::<Tensor>(
                &ys, None, None, tch::Reduction::Mean);
            opt.zero_grad();
            loss.backward();
            opt.step();
            if epoch % 20 == 0 {
                println!("Epoch: {}, Loss: {}", epoch, f64::from(loss));
            }
        }

        // Evaluate accuracy
        let preds = net.forward(&xs).ge(0.5).to_kind(tch::Kind::Float);
        let correct = preds.eq_tensor(&ys).sum(tch::Kind::Int64);
        let accuracy = f64::from(&correct) / y.len() as f64;
        println!("Accuracy: {}", accuracy);

        Ok(())
    }
    ```

2. **Ensure Dependencies**:
   - Verify `Cargo.toml` includes:
     ```toml
     [dependencies]
     tch = "0.17.0"
     ndarray = "0.15.0"
     ```
   - Run `cargo build`.

3. **Run the Program**:
    ```bash
    cargo run
    ```
    **Expected Output** (approximate):
    ```
    Epoch: 20, Loss: 0.50
    Epoch: 40, Loss: 0.35
    Epoch: 60, Loss: 0.25
    Epoch: 80, Loss: 0.18
    Epoch: 100, Loss: 0.12
    Accuracy: 0.90
    ```

## Understanding the Results

- **Dataset**: Synthetic sequences (10 samples, 5 time steps, 2 features) represent two classes (low vs. high values), mimicking time-series data.
- **Model**: An LSTM with 10 hidden units processes sequences, outputting a class prediction at the last time step, achieving ~90% accuracy.
- **Loss**: The cross-entropy loss decreases (~0.12), indicating convergence.
- **Under the Hood**: `tch-rs` leverages PyTorch's optimized LSTM routines, with Rust's memory safety preventing tensor mismanagement during sequence unrolling, a risk in C++ BPTT. The LSTM's gates mitigate vanishing gradients, enabling longer sequence modeling than vanilla RNNs. Rust's compiled performance reduces training time compared to Python's `pytorch`, especially for CPU-bound tasks with many time steps.
- **Evaluation**: High accuracy confirms effective sequence learning, though validation data would detect overfitting in practice.

This lab introduces sequence modeling, preparing for optimization techniques.

## Next Steps

<!-- Continue to [Optimization](/deep-learning/optimization) for advanced training methods, or revisit [Convolutional Neural Networks](/deep-learning/cnns). -->

## Further Reading

- *Deep Learning* by Goodfellow et al. (Chapter 10)
- *Hands-On Machine Learning* by Géron (Chapter 16)
- `tch-rs` Documentation: [github.com/LaurentMazare/tch-rs](https://github.com/LaurentMazare/tch-rs)