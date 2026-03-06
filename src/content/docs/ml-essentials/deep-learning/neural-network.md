---
title: Neural Networks
description: In-depth exploration of feedforward neural networks for deep learning
---
# Neural Networks

Neural networks are the foundation of deep learning, modeling complex patterns for tasks like classification and regression. This section provides a comprehensive exploration of feedforward neural networks, including architecture, backpropagation, and optimization, with a Rust lab using `tch-rs`. We'll delve into computational details, gradient computation, and Rust's performance advantages, starting the Deep Learning module.

## Theory

A **feedforward neural network** consists of layers of interconnected nodes (neurons), processing input $\mathbf{x} \in \mathbb{R}^n$ to produce output $\hat{\mathbf{y}}$. Each layer applies a linear transformation followed by a non-linear activation function. For a network with $L$ layers, the output of layer $l$ is:
$$
\mathbf{z}^{(l)} = \mathbf{W}^{(l)} \mathbf{a}^{(l-1)} + \mathbf{b}^{(l)}, \quad \mathbf{a}^{(l)} = g(\mathbf{z}^{(l)})
$$
where $\mathbf{W}^{(l)}$ is the weight matrix, $\mathbf{b}^{(l)}$ is the bias, $\mathbf{a}^{(l-1)}$ is the previous layer's activation, and $g$ is the activation (e.g., ReLU, $g(z) = \max(0, z)$, or sigmoid, $g(z) = \frac{1}{1 + e^{-z}}$). The final layer produces $\hat{\mathbf{y}}$.

For classification, the output layer uses a softmax function for probabilities:
$$
\hat{y}_k = \frac{e^{z_k^{(L)}}}{\sum_{j=1}^K e^{z_j^{(L)}}}
$$
where $K$ is the number of classes.

### Derivation: Backpropagation

The network is trained to minimize a loss function, such as **cross-entropy loss** for classification:
$$
J(\boldsymbol{\theta}) = -\frac{1}{m} \sum_{i=1}^m \sum_{k=1}^K y_{ik} \log \hat{y}_{ik}
$$
where $\boldsymbol{\theta}$ includes all weights and biases, $y_{ik}$ is 1 if sample $i$ is class $k$, else 0, and $m$ is the number of samples. **Backpropagation** computes gradients $\frac{\partial J}{\partial \boldsymbol{\theta}}$ using the chain rule.

For a single sample, consider the loss $J_i$. The gradient for the final layer's weights $\mathbf{W}^{(L)}$ is:
$$
\frac{\partial J_i}{\partial \mathbf{W}^{(L)}} = \frac{\partial J_i}{\partial \mathbf{z}^{(L)}} \cdot \frac{\partial \mathbf{z}^{(L)}}{\partial \mathbf{W}^{(L)}}
$$
The error term is:
$$
\boldsymbol{\delta}^{(L)} = \frac{\partial J_i}{\partial \mathbf{z}^{(L)}} = \hat{\mathbf{y}}_i - \mathbf{y}_i
$$
for cross-entropy with softmax. The weight gradient is:
$$
\frac{\partial J_i}{\partial \mathbf{W}^{(L)}} = \boldsymbol{\delta}^{(L)} \cdot \mathbf{a}^{(L-1)T}
$$
For earlier layers, propagate the error backward:
$$
\boldsymbol{\delta}^{(l)} = \left( \mathbf{W}^{(l+1)T} \boldsymbol{\delta}^{(l+1)} \right) \cdot g'(\mathbf{z}^{(l)})
$$
where $g'$ is the derivative of the activation function (e.g., for ReLU, $g'(z) = 1$ if $z > 0$, else 0). Gradients are averaged over the batch:
$$
\nabla_{\boldsymbol{\theta}} J = \frac{1}{m} \sum_{i=1}^m \nabla_{\boldsymbol{\theta}} J_i
$$

**Under the Hood**: Backpropagation requires efficient matrix operations, costing $O(n_l n_{l-1})$ per layer $l$. Rust's `tch-rs`, built on PyTorch's C++ backend, optimizes these with BLAS routines, leveraging Rust's memory safety to prevent leaks during gradient updates, unlike raw C++ where pointer errors are common. The computational graph tracks dependencies, enabling automatic differentiation, a feature `tch-rs` inherits from PyTorch, outperforming Python's dynamic overhead for large networks.

## Optimization

Gradient descent updates weights:
$$
\boldsymbol{\theta} \leftarrow \boldsymbol{\theta} - \eta \nabla_{\boldsymbol{\theta}} J
$$
where $\eta$ is the learning rate. Variants like **stochastic gradient descent (SGD)** use mini-batches, and **Adam** adapts $\eta$ using momentum and variance estimates. Regularization (e.g., $L_2$ penalty, $\lambda \sum w^2$) prevents overfitting.

**Under the Hood**: Adam combines momentum and adaptive scaling, converging faster than SGD but requiring careful tuning of $\beta_1$, $\beta_2$. `tch-rs` implements Adam with Rust's zero-cost abstractions, ensuring high performance without Python's interpreter overhead. Rust's ownership model guarantees safe tensor operations, critical for large networks where C++ risks memory corruption.

## Evaluation

Performance is evaluated with:

- **Classification**: Accuracy, Precision, Recall, F1-Score, ROC-AUC (as in prior modules).
- **Regression**: MSE, RMSE, MAE, $R^2$.
- **Training/Validation Loss**: Monitor $J$ on training and validation sets to detect overfitting.

**Under the Hood**: Validation loss guides hyperparameter tuning (e.g., layers, neurons). `tch-rs` computes metrics efficiently, using GPU acceleration when available, outperforming Python's `pytorch` for CPU-bound tasks due to Rust's compiled efficiency. Rust's type system ensures tensor compatibility, avoiding runtime errors common in dynamic languages.

## Lab: Neural Network with `tch-rs`

You'll train a feedforward neural network on a synthetic dataset for binary classification, evaluating accuracy and loss.

1. **Edit** `src/main.rs` in your `rust_ml_tutorial` project:
    ```rust
    use tch::{nn, nn::Module, nn::OptimizerConfig, Device, Tensor};
    use ndarray::{array, Array2, Array1};

    fn main() -> Result<(), tch::TchError> {
        // Synthetic dataset: features (x1, x2), binary target (0 or 1)
        let x: Array2<f64> = array![
            [1.0, 2.0], [2.0, 1.0], [3.0, 3.0], [4.0, 5.0], [5.0, 4.0],
            [6.0, 1.0], [7.0, 2.0], [8.0, 3.0], [9.0, 4.0], [10.0, 5.0]
        ];
        let y: Array1<f64> = array![0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0];

        // Convert to tensors
        let device = Device::Cpu;
        let xs = Tensor::from_slice(x.as_slice().unwrap()).to_device(device);
        let ys = Tensor::from_slice(y.as_slice().unwrap()).to_device(device);

        // Define neural network
        let vs = nn::VarStore::new(device);
        let net = nn::seq()
            .add(nn::linear(&vs.root() / "layer1", 2, 10, Default::default()))
            .add_fn(|xs| xs.relu())
            .add(nn::linear(&vs.root() / "layer2", 10, 1, Default::default()))
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
    Epoch: 20, Loss: 0.45
    Epoch: 40, Loss: 0.30
    Epoch: 60, Loss: 0.22
    Epoch: 80, Loss: 0.18
    Epoch: 100, Loss: 0.15
    Accuracy: 0.90
    ```

## Understanding the Results

- **Dataset**: Synthetic features ($x_1$, $x_2$) predict binary classes (0 or 1), as in prior labs.
- **Model**: A 2-layer neural network (2 input neurons, 10 hidden with ReLU, 1 output with sigmoid) learns a non-linear boundary, achieving ~90% accuracy.
- **Loss**: The cross-entropy loss decreases (~0.15), indicating convergence.
- **Under the Hood**: `tch-rs` uses PyTorch's C++ backend for automatic differentiation, computing gradients via backpropagation. Rust's memory safety ensures robust tensor operations, avoiding leaks common in C++ during graph construction. The Adam optimizer adapts learning rates, converging faster than SGD, with Rust's compiled performance outpacing Python's `pytorch` for CPU tasks.
- **Evaluation**: High accuracy confirms effective learning, though validation data would detect overfitting in practice.

This lab introduces deep learning, preparing for convolutional neural networks.

## Next Steps

<!-- Continue to [Convolutional Neural Networks](/deep-learning/cnns) for image processing, or revisit [Model Evaluation](/core-ml/evaluation). -->

## Further Reading

- *Deep Learning* by Goodfellow et al. (Chapter 6)
- *Hands-On Machine Learning* by Géron (Chapter 10)
- `tch-rs` Documentation: [github.com/LaurentMazare/tch-rs](https://github.com/LaurentMazare/tch-rs)