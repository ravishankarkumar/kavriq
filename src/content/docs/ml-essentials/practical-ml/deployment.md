---
title: Model Deployment
description: In-depth exploration of deploying machine learning models
---
# Model Deployment

Model deployment brings machine learning (ML) models into production, enabling real-time predictions via APIs or batch processing. This section provides a comprehensive exploration of model serialization, API design, and inference optimization, with a Rust lab using `actix-web` and `linfa`. We'll dive into performance optimization, inference efficiency, and Rust's advantages, concluding the Practical ML Skills module.

## Theory

Deployment involves serving a trained model to handle new data, balancing latency, scalability, and reliability. For a model with parameters $\boldsymbol{\theta}$, inference computes predictions $\hat{\mathbf{y}} = f(\mathbf{x}, \boldsymbol{\theta})$ for input $\mathbf{x}$. Key considerations include:

- **Serialization**: Saving $\boldsymbol{\theta}$ to disk for portability.
- **API Design**: Exposing predictions via RESTful endpoints.
- **Optimization**: Minimizing inference time and resource usage.

### Model Serialization

Serialization converts a model's parameters into a format (e.g., JSON, binary) for storage and loading. For a logistic regression model with weights $\mathbf{w} \in \mathbb{R}^n$ and bias $b$, the serialized form is:
$$
\boldsymbol{\theta} = [\mathbf{w}, b]
$$
Deserialization reconstructs $\boldsymbol{\theta}$ for inference.

**Under the Hood**: Serialization requires efficient I/O operations, costing $O(n)$ for $n$ parameters. `linfa` uses `serde` for JSON serialization, leveraging Rust's zero-copy deserialization for speed, unlike Python's `joblib`, which may incur memory copying overhead. Rust's type safety ensures correct parameter parsing, avoiding C++'s manual buffer errors.

### API Design

A RESTful API serves predictions via HTTP endpoints (e.g., POST `/predict`). For input $\mathbf{x} \in \mathbb{R}^n$, the API returns $\hat{\mathbf{y}}$. The latency model is:
$$
T_{\text{total}} = T_{\text{network}} + T_{\text{deserialize}} + T_{\text{inference}} + T_{\text{serialize}}
$$
where $T_{\text{inference}}$ depends on model complexity (e.g., $O(n)$ for logistic regression).

**Under the Hood**: API performance hinges on request handling and concurrency. `actix-web` uses Rust's asynchronous runtime (`tokio`) for high-throughput request processing, outperforming Python's `FastAPI` for CPU-bound tasks due to Rust's compiled efficiency. Rust's memory safety prevents race conditions in concurrent requests, unlike C++'s manual thread management.

### Inference Optimization

Inference time is optimized by:

- **Batch Inference**: Processing multiple inputs $\mathbf{X} \in \mathbb{R}^{b \times n}$ (batch size $b$) leverages vectorized operations, reducing $T_{\text{inference}}$ to $O(b n)$ vs. $O(b \cdot n)$ for sequential processing.
- **Model Quantization**: Reducing parameter precision (e.g., float32 to int8) lowers memory and computation costs.
- **Hardware Acceleration**: Using GPUs or TPUs for matrix operations.

**Derivation**: For logistic regression, inference computes $\hat{y} = \sigma(\mathbf{w}^T \mathbf{x} + b)$, where $\sigma(z) = \frac{1}{1 + e^{-z}}$. The computational cost is:
$$
T_{\text{inference}} \approx c \cdot n
$$
where $c$ is a constant (e.g., FLOPs per operation). Batching amortizes overhead:
$$
T_{\text{batch}} \approx c \cdot b \cdot n / p
$$
where $p$ is the parallelism factor (e.g., GPU cores).

**Under the Hood**: Batch inference exploits SIMD instructions or GPU parallelism. `tch-rs` optimizes this with PyTorch's C++ backend, while `linfa` uses `ndarray` for CPU efficiency. Rust's compiled performance minimizes latency compared to Python's `pytorch`, and its type safety ensures correct tensor shapes, avoiding C++'s runtime errors.

## Evaluation

Deployment performance is evaluated with:

- **Latency**: Time from request to response ($T_{\text{total}}$).
- **Throughput**: Requests per second, $\frac{1}{T_{\text{total}}}$.
- **Accuracy**: Consistency with training performance (e.g., accuracy, MSE).
- **Resource Usage**: CPU, memory, or GPU consumption.

**Under the Hood**: Low latency and high throughput are critical for real-time applications. `actix-web` optimizes throughput with asynchronous handlers, leveraging Rust's `tokio` for non-blocking I/O, unlike Python's `FastAPI`, which may block under high load. Rust's memory safety prevents leaks during long-running services, a risk in C++.

## Lab: Model Deployment with `actix-web` and `linfa`

You'll deploy a logistic regression model as a RESTful API using `actix-web`, serving predictions on a synthetic dataset.

1. **Edit `src/main.rs` in your `rust_ml_tutorial` project:
    ```rust
    use actix_web::{web, App, HttpResponse, HttpServer};
    use linfa::prelude::*;
    use linfa_linear::LogisticRegression;
    use ndarray::{array, Array1, Array2};
    use serde::{Deserialize, Serialize};

    #[derive(Serialize, Deserialize)]
    struct PredictRequest {
        features: Vec<f64>,
    }

    #[derive(Serialize)]
    struct PredictResponse {
        prediction: f64,
    }

    async fn predict(
        req: web::Json<PredictRequest>,
        model: web::Data<LogisticRegression<f64>>,
    ) -> HttpResponse {
        let x = Array2::from_shape_vec((1, req.features.len()), req.features.clone()).unwrap();
        let pred = model.predict(&x)[0];
        HttpResponse::Ok().json(PredictResponse { prediction: pred })
    }

    #[actix_web::main]
    async fn main() -> std::io::Result<()> {
        // Synthetic training dataset
        let x: Array2<f64> = array![
            [1.0, 2.0], [2.0, 1.0], [3.0, 3.0], [4.0, 5.0], [5.0, 4.0],
            [6.0, 1.0], [7.0, 2.0], [8.0, 3.0], [9.0, 4.0], [10.0, 5.0]
        ];
        let y: Array1<f64> = array![0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let dataset = Dataset::new(x, y);

        // Train logistic regression
        let model = LogisticRegression::default()
            .l2_penalty(0.1)
            .max_iterations(100)
            .fit(&dataset)
            .unwrap();

        // Start HTTP server
        HttpServer::new(move || {
            App::new()
                .app_data(web::Data::new(model.clone()))
                .route("/predict", web::post().to(predict))
        })
        .bind("127.0.0.1:8080")?
        .run()
        .await
    }
    ```

2. **Ensure Dependencies**:
   - Verify `Cargo.toml` includes:
     ```toml
     [dependencies]
     actix-web = "4.4.0"
     linfa = "0.7.1"
     linfa-linear = "0.7.0"
     ndarray = "0.15.0"
     serde = { version = "1.0", features = ["derive"] }
     ```
   - Run `cargo build`.

3. **Run the Program**:
    ```bash
    cargo run
    ```
    - The server starts at `http://127.0.0.1:8080`.
    - Test the API with a `curl` command:
      ```bash
      curl -X POST -H "Content-Type: application/json" -d '{"features":[7.0,2.0]}' http://127.0.0.1:8080/predict
      ```
    **Expected Output** (approximate):
      ```json
      {"prediction":1.0}
      ```

## Understanding the Results

- **Dataset**: The logistic regression model, trained on synthetic features ($x_1$, $x_2$) and binary targets, is deployed as an API.
- **API**: The `/predict` endpoint accepts feature vectors and returns predictions (e.g., class 1 for input [7.0, 2.0]).
- **Under the Hood**: `actix-web` handles requests asynchronously, with `linfa` performing inference in $O(n)$ time for $n$ features. Rust's `tokio` runtime ensures high throughput, outperforming Python's `FastAPI` for concurrent requests due to compiled efficiency. `serde`'s zero-copy JSON parsing minimizes latency, unlike Python's `serde_json`, which may copy data. Rust's memory safety prevents request handling errors, unlike C++'s manual concurrency management, which risks race conditions.
- **Evaluation**: The API delivers correct predictions, with low latency and scalability, suitable for production use. Real-world deployment would require monitoring latency and throughput under load.

This lab concludes the Practical ML Skills module, preparing for advanced topics.

## Next Steps

<!-- Continue to [Natural Language Processing](/advanced/nlp) for advanced topics, or revisit [Hyperparameter Tuning](/practical-ml/tuning). -->

## Further Reading

- *Hands-On Machine Learning* by Géron (Chapter 2)
- `actix-web` Documentation: [actix.rs](https://actix.rs)
- `linfa` Documentation: [github.com/rust-ml/linfa](https://github.com/rust-ml/linfa)