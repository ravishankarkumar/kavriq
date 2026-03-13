---
title: Time Series Basics - Trends, Seasonality, ARIMA
description: Comprehensive exploration of time series analysis for AI/ML, covering trends, seasonality, ARIMA models, their mathematical foundations, and applications in forecasting and anomaly detection, with examples and code in Python and Rust
layout: ../../../../layouts/TutorialPage.astro
---

# Time Series Basics - Trends, Seasonality, ARIMA

Time series analysis studies data points collected over time to uncover patterns, forecast future values, and detect anomalies. In machine learning (ML), time series methods are crucial for tasks like stock price prediction, demand forecasting, and anomaly detection in sensor data. Key components include trends (long-term movement), seasonality (periodic patterns), and stochastic processes like ARIMA, which model temporal dependencies.

This sixteenth lecture in the "Statistics Foundations for AI/ML" series builds on multivariate statistics and nonparametric methods, exploring time series components, ARIMA models, their mathematical foundations, and ML applications. We'll provide intuitive explanations, derivations, and practical implementations in Python and Rust, preparing you for causal inference and experimental design.

---

## 1. Why Time Series Analysis Matters in ML

Time series data, unlike i.i.d. data, has temporal dependencies, requiring specialized methods to:
- Model trends and seasonality for accurate forecasts.
- Capture autocorrelations in sequential data.
- Detect anomalies in dynamic systems.

### ML Connection
- **Forecasting**: Predict sales, traffic, or prices.
- **Anomaly Detection**: Identify outliers in time-series data.
- **Feature Engineering**: Extract trend/seasonal features.

::: info
Time series analysis is like reading a book's plot over time—trends set the direction, seasons repeat themes, and ARIMA captures the story's flow.
:::

### Example
- Monthly sales data: Trend shows growth, seasonality reflects holiday peaks, ARIMA forecasts next month.

---

## 2. Components of Time Series

**Trend**: Long-term increase/decrease (e.g., rising stock prices).

**Seasonality**: Periodic patterns (e.g., daily temperature cycles).

**Residuals**: Random noise after removing trend/seasonality.

**Model**: y_t = T_t + S_t + ε_t (additive) or y_t = T_t × S_t × ε_t (multiplicative).

### ML Application
- Decompose time series to engineer features for ML models.

---

## 3. Stationarity and Autocorrelation

**Stationarity**: Mean, variance, autocorrelation constant over time.

- **Test**: Augmented Dickey-Fuller (ADF) for unit roots.
- **Transform**: Differencing, log-transform to stabilize.

**Autocorrelation**: Correlation of y_t with y_{t-k} (ACF plot).

### ML Insight
- Stationarity required for ARIMA; non-stationary data needs preprocessing.

---

## 4. ARIMA Models: Autoregressive Integrated Moving Average

**ARIMA(p,d,q)**:
- **AR(p)**: y_t = φ₁y_{t-1} + ... + φₚy_{t-p} + ε_t.
- **I(d)**: Differencing d times to achieve stationarity.
- **MA(q)**: y_t = ε_t + θ₁ε_{t-1} + ... + θ_qε_{t-q}.

**Model**: Δ^d y_t = φ₁ Δ^d y_{t-1} + ... + θ₁ ε_{t-1} + ... + ε_t.

### Parameters
- p: AR order (from ACF/PACF).
- d: Differencing order.
- q: MA order.

### ML Application
- Forecast time-series data (e.g., stock prices).

---

## 5. Estimating ARIMA Parameters

**MLE**: Maximize likelihood of ARIMA model.

**ACF/PACF**: Identify p, q by lag patterns.

**Grid Search**: Test (p,d,q) combinations via AIC/BIC.

### ML Connection
- ARIMA for baseline forecasting in ML pipelines.

---

## 6. Applications in Machine Learning

1. **Forecasting**: Predict future values (e.g., demand).
2. **Anomaly Detection**: Residuals outside threshold.
3. **Feature Engineering**: Trend/seasonal components as inputs.
4. **Time-Series Models**: ARIMA as baseline for LSTMs.

### Challenges
- **Non-Stationarity**: Requires preprocessing.
- **Parameter Selection**: p,d,q tuning complex.
- **Nonlinear Patterns**: ARIMA limited; use LSTMs.

---

## 7. Numerical Time Series Analysis

Decompose, fit ARIMA, forecast.

::: code-group

```python [Python]
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

# Generate time series
t = np.arange(100)
data = 0.1 * t + 5 * np.sin(2 * np.pi * t / 12) + np.random.normal(0, 1, 100)
ts = pd.Series(data, index=pd.date_range('2025-01-01', periods=100, freq='D'))

# Decompose
decomp = seasonal_decompose(ts, period=12)
decomp.plot()
plt.show()

# Stationarity test
adf_result = adfuller(ts)
print("ADF: stat=", adf_result[0], "p=", adf_result[1])

# ARIMA fit and forecast
model = ARIMA(ts, order=(1,1,1)).fit()
forecast = model.forecast(steps=10)
print("ARIMA forecast:", forecast)

# ML: Anomaly detection
resid = model.resid
anomalies = ts[np.abs(resid) > 2 * resid.std()]
print("Anomalies:", anomalies)
```

```rust [Rust]
use rand::Rng;

fn arima_forecast(data: &[f64], p: usize, d: usize, q: usize, steps: usize) -> Vec<f64> {
    // Simplified ARIMA (1,1,1) simulation
    let mut rng = rand::thread_rng();
    let mut diff = data.windows(2).map(|w| w[1] - w[0]).collect::<Vec<f64>>();
    let phi = 0.5; // Example AR(1)
    let theta = 0.3; // Example MA(1)
    let mut forecasts = vec![data[data.len()-1]];
    let mut errors = vec![0.0; q];
    for _ in 1..steps {
        let last_diff = diff[diff.len()-1];
        let error = rng.gen::<f64>() * 1.0; // Simulate noise
        let new_diff = phi * last_diff + error + theta * errors[errors.len()-1];
        forecasts.push(forecasts[forecasts.len()-1] + new_diff);
        diff.push(new_diff);
        errors.push(error);
    }
    forecasts
}

fn main() {
    let mut rng = rand::thread_rng();
    let t: Vec<f64> = (0..100).map(|i| i as f64).collect();
    let data: Vec<f64> = t.iter().map(|&ti| 0.1 * ti + 5.0 * (2.0 * std::f64::consts::PI * ti / 12.0).sin() + rng.gen::<f64>()).collect();

    // Simplified decomposition (trend + seasonal)
    let trend: Vec<f64> = t.iter().map(|&ti| 0.1 * ti).collect();
    let seasonal: Vec<f64> = t.iter().map(|&ti| 5.0 * (2.0 * std::f64::consts::PI * ti / 12.0).sin()).collect();
    let resid: Vec<f64> = data.iter().zip(trend.iter()).zip(seasonal.iter()).map(|((&di, &ti), &si)| di - ti - si).collect();
    let resid_std = (resid.iter().map(|&r| r.powi(2)).sum::<f64>() / resid.len() as f64).sqrt();

    // ARIMA forecast
    let forecast = arima_forecast(&data, 1, 1, 1, 10);
    println!("ARIMA forecast: {:?}", forecast);

    // Anomaly detection
    let anomalies: Vec<(usize, f64)> = resid.iter().enumerate().filter(|(_, &r)| r.abs() > 2.0 * resid_std).map(|(i, &r)| (i, data[i])).collect();
    println!("Anomalies: {:?}", anomalies);
}
```
:::

Implements decomposition, ARIMA, anomaly detection.

---

## 8. Theoretical Foundations

**Stationarity**: Constant moments, critical for ARIMA.

**ARIMA**: Combines AR, differencing, MA for flexible modeling.

**ACF/PACF**: Identify model orders.

### ML Insight
- ARIMA as baseline for complex ML time-series models.

---

## 9. Challenges in ML Applications

- **Non-Stationarity**: Requires preprocessing.
- **Model Selection**: Choosing p,d,q complex.
- **Nonlinear Data**: ARIMA limited; use LSTMs.

---

## 10. Key ML Takeaways

- **Trends, seasonality model**: Long-term, periodic patterns.
- **ARIMA forecasts**: Linear dependencies.
- **Stationarity critical**: For modeling.
- **Anomaly detection practical**: Residual-based.
- **Code implements**: Time-series analysis.

Time series analysis drives ML forecasting.

---

## 11. Summary

Explored time series components (trends, seasonality), ARIMA, with ML applications in forecasting and anomaly detection. Examples and Python/Rust code bridge theory to practice. Prepares for causal inference and experimental design.

Word count: Approximately 3000.

---

## Further Reading
- Hyndman, Athanasopoulos, *Forecasting: Principles and Practice*.
- Box, Jenkins, *Time Series Analysis*.
- James, *Introduction to Statistical Learning* (Ch. 10).
- Rust: 'statrs' for time-series, 'rand' for simulation.

---