---
title: Setup
description: Setting up the environment for AI/ML in Rust and Python
layout: ../../../layouts/TutorialPage.astro
---
# Setup

This guide helps you set up your environment for the AI/ML tutorials.  
We support **two paths**:  
- **Rust** for performance and safety.  
- **Python** for ecosystem richness and beginner-friendliness.  

Choose the language you prefer (or follow both for cross-learning).  
Both setups include installing the language, libraries, IDE, and verifying with a simple program.

---

## Setup with Rust

### Step 1: Install Rust
1. **Visit [rust-lang.org](https://www.rust-lang.org/tools/install)**.  
2. **Install rustup**:  
   On Unix-like systems (macOS/Linux):  
   ```bash
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   ```  
   On Windows, follow the instructions on the website.  
3. **Update Rust**:  
   ```bash
   rustup update
   ```  
4. **Verify installation**:  
   ```bash
   rustc --version
   cargo --version
   ```

### Step 2: IDE Setup
Use **Visual Studio Code (VS Code)**:  
- Install [VS Code](https://code.visualstudio.com).  
- Add the `rust-analyzer` extension.  

### Step 3: Create a Rust Project
```bash
cargo new rust_ml_tutorial
cd rust_ml_tutorial
cargo run
```
Expect “Hello, world!” output.

### Step 4: Install ML Libraries
Add dependencies in `Cargo.toml`:
```toml
[dependencies]
linfa = "0.7"
nalgebra = "0.32"
tch = "0.13"
polars = "0.32"
rust-bert = "0.24"
actix-web = "4"
plotters = "0.3"
ndarray = "0.15"
rand = "0.8"
statrs = "0.16"
```
Run:
```bash
cargo build
```

### Step 5: Verify Rust Setup
Test with a matrix:
```rust
use nalgebra::Matrix2;

fn main() {
    let matrix = Matrix2::new(1.0, 2.0, 3.0, 4.0);
    println!("Matrix:\n{}", matrix);
}
```
```bash
cargo run
```

---

## Setup with Python

### Step 1: Install Python
- Install Python 3.11+ from [python.org](https://www.python.org/downloads/) or use [Anaconda](https://www.anaconda.com/download).  
- Verify installation:  
  ```bash
  python --version
  pip --version
  ```

### Step 2: Create a Virtual Environment
```bash
python -m venv venv
source venv/bin/activate   # macOS/Linux
venv\Scripts\activate      # Windows
```

### Step 3: Install ML Libraries
```bash
pip install numpy pandas scikit-learn torch tensorflow matplotlib seaborn jupyter
```

### Step 4: IDE Setup
Use **VS Code** or **Jupyter Notebook**:  
- Install [VS Code](https://code.visualstudio.com).  
- Add the Python extension.  
- Or, run notebooks:  
  ```bash
  jupyter notebook
  ```

### Step 5: Verify Python Setup
```python
import numpy as np

matrix = np.array([[1, 2], [3, 4]])
print("Matrix:\n", matrix)
```
Run:
```bash
python main.py
```

---

## Troubleshooting

- **Rust issues**: reinstall via [rust-lang.org](https://www.rust-lang.org).  
- **Python issues**: ensure pip is updated (`pip install --upgrade pip`).  
- **Library conflicts**: use `cargo clean` (Rust) or `pip uninstall`/`pip freeze` (Python).  
- **IDE issues**: check extensions (`rust-analyzer`, Python).  

---

## Next Steps
- For Rust: continue to [Rust Basics](/ml-essentials/getting-started/rust-basics).  
- For Python: continue to [Python Basics](/ml-essentials/getting-started/python-basics).  

## Further Reading
- Rust: [nalgebra.org](https://www.nalgebra.org), *Hands-On Machine Learning* (Ch. 2)  
- Python: [scikit-learn docs](https://scikit-learn.org), *Deep Learning with Python* by Chollet  
