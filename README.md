# Adam Optimizer from Scratch

A from-scratch implementation and experimental study of the **Adam optimizer** using NumPy, followed by training a small neural network on the XOR problem.

The goal of this project is to understand what happens inside an optimizer rather than relying on pre-built implementations such as `torch.optim.Adam`.

---

## 📌 Overview

Optimization is one of the fundamental components of neural network training.

In this project, I implemented and studied three optimization methods:

- Stochastic Gradient Descent (SGD)
- Momentum
- Adam

The project progresses from a simple mathematical optimization problem to a complete neural network trained from scratch.

### Main Workflow

```
Toy Optimization
      ↓
Adam from Scratch
      ↓
Forward Propagation
      ↓
Binary Cross-Entropy Loss
      ↓
Backpropagation
      ↓
Adam Optimization
      ↓
XOR Neural Network
      ↓
SGD vs Momentum vs Adam
      ↓
Learning Rate Experiment
```

---

## 🎯 Objectives

1. Understand gradient-based optimization.
2. Implement Momentum from scratch.
3. Implement Adam from scratch.
4. Understand first and second moment estimates.
5. Understand Adam's bias correction.
6. Implement a neural network using NumPy.
7. Implement backpropagation manually.
8. Train a neural network on XOR.
9. Compare SGD, Momentum, and Adam.
10. Study the effect of learning rate on Adam.

---

## 🧮 Part 1: Adam on a Toy Optimization Problem

Before using Adam on a neural network, I tested it on a simple objective function:

$$L(\theta) = \theta^2$$

The gradient is:

$$\frac{dL}{d\theta} = 2\theta$$

The initial parameter was $\theta_0 = 10$, and the goal was to minimize the function toward $\theta \rightarrow 0$.

### Adam Update Rule

Adam maintains two moving averages.

**First moment (mean of gradients):**

$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$$

**Second moment (uncentered variance of gradients):**

$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$

Because both moment estimates are initialized at zero, Adam applies bias correction:

$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}$$

$$\hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$

The final parameter update rule is:

$$\theta_t = \theta_{t-1} - \eta \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

### Hyperparameters

| Parameter     | Value  |
|---------------|--------|
| Learning rate | `0.1`  |
| β₁            | `0.9`  |
| β₂            | `0.999`|
| ε             | `1e-8` |

---

## 🧠 Part 2: Neural Network from Scratch

After testing Adam on a simple function, I implemented a small neural network using only NumPy.

### Architecture

```
2 Input Neurons
       ↓
4 Hidden Neurons (tanh)
       ↓
1 Output Neuron (sigmoid)
```

### Forward Propagation

**Hidden layer:**

$$Z_1 = X W_1 + b_1$$

$$A_1 = \tanh(Z_1)$$

**Output layer:**

$$Z_2 = A_1 W_2 + b_2$$

$$\hat{y} = \sigma(Z_2) = \frac{1}{1 + e^{-Z_2}}$$

---

## 📉 Loss Function

Binary Cross-Entropy was used for the XOR classification problem:

$$L = -\frac{1}{n} \sum_{i} \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right]$$

---

## 🔄 Backpropagation

Gradients were calculated manually using the chain rule:

```
dW1, db1, dW2, db2
```

### Training Loop

```
Forward Pass
     ↓
Calculate Loss
     ↓
Backpropagation
     ↓
Calculate Gradients
     ↓
Optimizer Step
     ↓
Update Weights & Biases
     ↓
Repeat
```

---

## 🚀 Training XOR with Adam

### Dataset

| Input    | Target |
|----------|-------:|
| `[0, 0]` |      0 |
| `[0, 1]` |      1 |
| `[1, 0]` |      1 |
| `[1, 1]` |      0 |

The network was trained for **5000 epochs** using Adam.

### Results

**Final Binary Cross-Entropy loss:** `0.000006`

**Raw predictions:**

```
[1.14003288e-07]
[9.99994716e-01]
[9.99990093e-01]
[8.68342740e-06]
```

**Rounded predictions vs. actual labels:**

| Predicted | Actual |
|-----------|--------|
| 0         | 0      |
| 1         | 1      |
| 1         | 1      |
| 0         | 0      |

The network successfully learned the XOR mapping. ✅

---

## ⚖️ SGD vs Momentum vs Adam

The same neural network initialization was used for all three optimizers to keep the comparison fair.

### Results After 5000 Epochs

| Optimizer | Final Loss |
|-----------|----------:|
| SGD       |  0.000970 |
| Momentum  |  0.000075 |
| Adam      |  0.000006 |

### How Each Optimizer Works

**SGD** uses only the current gradient:

$$\theta_{t+1} = \theta_t - \eta \, g_t$$

**Momentum** incorporates information from previous gradients:

$$v_t = \beta v_{t-1} + g_t, \qquad \theta_{t+1} = \theta_t - \eta \, v_t$$

**Adam** combines momentum-like first-moment tracking with second-moment adaptive scaling.

### Convergence Order (this experiment)

```
SGD       → Slowest convergence
Momentum  → Faster convergence
Adam      → Fastest convergence
```

> **Note:** This result is specific to the chosen architecture, initialization, hyperparameters, and dataset. It should not be interpreted as Adam being universally superior to every other optimizer.

---

## 📊 Learning Rate Experiment

The network was trained for **1000 epochs** using Adam with different learning rates.

| Learning Rate | Final Loss |
|--------------:|----------:|
|         0.001 |  0.411976 |
|          0.01 |  0.005772 |
|           0.1 |  0.000158 |
|           0.5 |  0.000061 |

### Observations

```
0.001 → Too slow — barely converged in 1000 epochs
0.01  → Better, but still slow
0.1   → Fast convergence
0.5   → Fastest among tested values
```

A learning rate that is too small causes very slow convergence. However, increasing the learning rate indefinitely is not beneficial — an excessively large value can cause unstable updates, oscillation, or divergence.

---

## 🛠️ Technologies Used

| Tool            | Purpose                         |
|-----------------|---------------------------------|
| Python          | Core language                   |
| NumPy           | All numerical computations      |
| Matplotlib      | Plotting and visualization      |
| Jupyter Notebook| Interactive development         |

No deep-learning framework was used. The project does **not** use `torch.optim.Adam` or any other pre-built optimizer.

---

## 📁 Project Structure

```
adam-optimizer-from-scratch/
│
├── Adam_Optimizer_From_Scratch.ipynb
├── README.md
└── images/
    ├── adam_theta_vs_iteration.png
    ├── adam_loss_vs_iteration.png
    ├── adam_xor_training_loss.png
    ├── optimizer_comparison.png
    └── learning_rate_comparison.png
```

---

## 🔬 Key Concepts Covered

- Gradient Descent & Stochastic Gradient Descent
- Momentum
- Adaptive optimization
- Adam: first moment, second moment, and bias correction
- Learning rates and their effect on convergence
- Forward propagation and backpropagation
- Chain rule
- Binary Cross-Entropy loss
- Neural network parameter updates
- Experimental comparison of optimizers

---

## 📈 Key Takeaways

### 1. Backpropagation and optimization are separate

Backpropagation computes $\nabla_\theta L$. The optimizer determines *how* those gradients are used to update parameters.

### 2. Momentum provides gradient memory

Instead of responding only to the current gradient, Momentum incorporates a weighted history of past gradients, smoothing the update direction.

### 3. Adam combines two ideas

Adam tracks both:
- the **direction** of recent gradients (first moment)
- the **magnitude** of recent gradients (second moment)

This enables adaptive per-parameter learning rates.

### 4. Hyperparameters matter

The learning rate can dramatically change the training behavior of an optimizer — too small means slow convergence, too large risks instability.

### 5. Implementing from scratch builds real understanding

Writing the optimizer manually makes the mathematical operations behind neural network training concrete and debuggable.

---

## ⭐ Summary

This project implements **Adam from scratch using NumPy** and demonstrates its use in training a neural network on XOR. The work spans the full pipeline — from the mathematical definition of Adam through forward propagation, manual backpropagation, and controlled experiments comparing **SGD, Momentum, and Adam** across loss values and learning rates.
