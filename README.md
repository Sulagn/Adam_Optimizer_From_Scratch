# Adam Optimizer from Scratch

A from-scratch implementation and experimental study of the **Adam optimizer using NumPy**, followed by training a small neural network on the XOR classification problem.

The purpose of this project is to understand the mathematics and mechanics behind optimization algorithms rather than relying on pre-built implementations such as `torch.optim.Adam`.

---

## 📌 Overview

Optimization is one of the fundamental components of neural network training.

In this project, I implemented and studied three optimization methods:

* Stochastic Gradient Descent (SGD)
* Momentum
* Adam

The project progresses from a simple mathematical optimization problem to a complete neural network trained entirely using NumPy.

### Main Workflow

```text
Toy Optimization Problem
          ↓
Adam from Scratch
          ↓
Neural Network from Scratch
          ↓
Forward Propagation
          ↓
Binary Cross-Entropy Loss
          ↓
Backpropagation
          ↓
Adam Optimization
          ↓
XOR Classification
          ↓
SGD vs Momentum vs Adam
          ↓
Learning Rate Experiment
```

---

## 🎯 Objectives

The main objectives of this project were:

1. Understand gradient-based optimization.
2. Implement Momentum from scratch.
3. Implement Adam from scratch.
4. Understand first and second moment estimates.
5. Understand Adam's bias correction.
6. Implement a neural network using NumPy.
7. Implement backpropagation manually.
8. Train a neural network on the XOR problem.
9. Compare SGD, Momentum, and Adam.
10. Investigate the effect of learning rate on Adam.

---

# 🧮 Part 1: Adam on a Toy Optimization Problem

Before using Adam on a neural network, I first tested it on a simple mathematical optimization problem.

The objective function is:

$$
L(\theta) = \theta^2
$$

The gradient of the objective function is:

$$
\frac{dL}{d\theta} = 2\theta
$$

The initial parameter was:

$$
\theta_0 = 10
$$

The goal is to minimize the objective function and move the parameter toward:

$$
\theta \rightarrow 0
$$

### Adam Update

Adam maintains two exponential moving averages of the gradients.

### First Moment

The first moment tracks the moving average of the gradients:

$$
m_t = \beta_1m_{t-1} + (1-\beta_1)g_t
$$

### Second Moment

The second moment tracks the moving average of the squared gradients:

$$
v_t = \beta_2v_{t-1} + (1-\beta_2)g_t^2
$$

Because both moment estimates are initialized to zero, they are biased toward zero during the early iterations. Adam therefore applies bias correction.

### Bias Correction

Corrected first moment:

$$
\hat{m}_t =
\frac{m_t}{1-\beta_1^t}
$$

Corrected second moment:

$$
\hat{v}_t =
\frac{v_t}{1-\beta_2^t}
$$

### Parameter Update

The final Adam update is:

$$
\theta_t =
\theta_{t-1}
------------

\eta
\frac{\hat{m}_t}
{\sqrt{\hat{v}_t}+\epsilon}
$$

### Hyperparameters

| Hyperparameter       |     Value |
| -------------------- | --------: |
| Learning rate $\eta$ |       0.1 |
| $\beta_1$            |       0.9 |
| $\beta_2$            |     0.999 |
| $\epsilon$           | $10^{-8}$ |

This experiment demonstrates how Adam uses both the direction and magnitude of recent gradients to adapt parameter updates during optimization.

---

# 🧠 Part 2: Neural Network from Scratch

After implementing Adam on a simple mathematical objective, I used it to train a small neural network on the XOR classification problem.

The entire neural network was implemented using **NumPy**, without using PyTorch, TensorFlow, or any pre-built optimizer.

## Network Architecture

The network consists of:

```text
2 Input Neurons
       ↓
4 Hidden Neurons
       ↓
1 Output Neuron
```

The hidden layer uses the hyperbolic tangent activation function:

$$
A_1 = \tanh(Z_1)
$$

The output layer uses the sigmoid activation function:

$$
\sigma(z) = \frac{1}{1+e^{-z}}
$$

---

# 🔄 Forward Propagation

The first layer computes:

$$
Z_1 = XW_1 + b_1
$$

The hidden activation is:

$$
A_1 = \tanh(Z_1)
$$

The second layer computes:

$$
Z_2 = A_1W_2 + b_2
$$

The final prediction is:

$$
\hat{y} = \sigma(Z_2)
$$

Therefore, the complete forward pass is:

```text
X
↓
Z₁ = XW₁ + b₁
↓
A₁ = tanh(Z₁)
↓
Z₂ = A₁W₂ + b₂
↓
ŷ = sigmoid(Z₂)
```

---

# 📉 Binary Cross-Entropy Loss

Since XOR is a binary classification problem, Binary Cross-Entropy was used as the loss function.

$$
L =
-\frac{1}{n}
\sum_{i=1}^{n}
\left[
y_i\log(\hat{y}_i)
+
(1-y_i)\log(1-\hat{y}_i)
\right]
$$

The purpose of the optimizer is to minimize this loss by updating the network parameters.

---

# 🔙 Backpropagation

Backpropagation was implemented manually using the chain rule.

The gradients calculated were:

```text
dW₁
db₁
dW₂
db₂
```

The gradient flow is:

```text
Loss
 ↓
Output Layer
 ↓
Hidden Layer
 ↓
Input Layer
```

More specifically:

$$
\text{Loss}
\rightarrow
Z_2
\rightarrow
A_1
\rightarrow
Z_1
\rightarrow
W_1,b_1
$$

Backpropagation calculates the gradients, while the optimizer determines how those gradients are used to update the parameters.

---

# 🚀 Part 3: Training XOR with Adam

The XOR dataset consists of four examples:

| Input    | Target |
| -------- | -----: |
| `[0, 0]` |      0 |
| `[0, 1]` |      1 |
| `[1, 0]` |      1 |
| `[1, 1]` |      0 |

The network was trained for **5000 epochs** using the manually implemented Adam optimizer.

### Training Result

Final Binary Cross-Entropy loss:

$$
0.000006
$$

### Predictions

The trained network produced:

```text
[1.14003288e-07]
[9.99994716e-01]
[9.99990093e-01]
[8.68342740e-06]
```

Rounded predictions:

```text
[0]
[1]
[1]
[0]
```

Actual labels:

```text
[0]
[1]
[1]
[0]
```

The neural network successfully learned the XOR mapping.

---

# ⚖️ Part 4: SGD vs Momentum vs Adam

To compare the optimizers fairly, all three experiments started from the **same network initialization**.

The architecture, dataset, loss function, and training budget were kept the same.

Only the optimization algorithm was changed.

## Optimization Methods

### SGD

Standard gradient descent uses:

$$
\theta_{t+1}
============

## \theta_t

\eta g_t
$$

SGD uses only the current gradient.

---

### Momentum

Momentum maintains a velocity:

$$
v_t =
\beta v_{t-1}
+
g_t
$$

and updates the parameters using:

$$
\theta_{t+1}
============

## \theta_t

\eta v_t
$$

Momentum introduces memory of previous gradients.

---

### Adam

Adam maintains both first and second moment estimates:

$$
m_t =
\beta_1m_{t-1}
+
(1-\beta_1)g_t
$$

$$
v_t =
\beta_2v_{t-1}
+
(1-\beta_2)g_t^2
$$

After bias correction:

$$
\hat{m}_t =
\frac{m_t}{1-\beta_1^t}
$$

$$
\hat{v}_t =
\frac{v_t}{1-\beta_2^t}
$$

The parameter update is:

$$
\theta_t =
\theta_{t-1}
------------

\eta
\frac{\hat{m}_t}
{\sqrt{\hat{v}_t}+\epsilon}
$$

---

## Experimental Results

After 5000 epochs:

| Optimizer | Final Loss |
| --------- | ---------: |
| SGD       |   0.000970 |
| Momentum  |   0.000075 |
| Adam      |   0.000006 |

### Observation

For this particular XOR experiment:

```text
SGD
 ↓
Slowest convergence

Momentum
 ↓
Faster convergence

Adam
 ↓
Fastest convergence
```

Adam achieved the lowest final loss under the chosen experimental conditions.

However, this does **not** imply that Adam is universally superior to SGD or Momentum. Optimizer performance depends on the dataset, architecture, initialization, learning rate, and other hyperparameters.

---

# 📊 Part 5: Learning Rate Experiment

The learning rate is one of the most important hyperparameters in optimization.

To investigate its effect, Adam was trained using four different learning rates:

$$
\eta \in {0.001,\ 0.01,\ 0.1,\ 0.5}
$$

The network initialization and other hyperparameters were kept fixed.

## Results

| Learning Rate | Final Loss after 1000 Epochs |
| ------------: | ---------------------------: |
|         0.001 |                     0.411976 |
|          0.01 |                     0.005772 |
|           0.1 |                     0.000158 |
|           0.5 |                     0.000061 |

### Observations

A very small learning rate can result in slow convergence.

For this experiment:

```text
0.001 → Slow convergence
0.01  → Improved convergence
0.1   → Fast convergence
0.5   → Fastest among tested values
```

However, increasing the learning rate indefinitely is not beneficial.

An excessively large learning rate can cause:

* Overshooting
* Oscillation
* Divergence
* Numerical instability
* `NaN` losses

Therefore, the learning rate must be chosen appropriately for the optimization problem.

---

# 📈 Experiments and Visualizations

The notebook includes visualizations for:

### 1. Adam Optimization on the Toy Function

Tracking:

$$
\theta
$$

and:

$$
L(\theta)
$$

over iterations.

### 2. XOR Training Loss

Tracking Binary Cross-Entropy loss over training epochs.

### 3. Optimizer Comparison

Comparing:

```text
SGD
Momentum
Adam
```

on the same XOR problem.

### 4. Learning Rate Comparison

Comparing Adam with:

```text
η = 0.001
η = 0.01
η = 0.1
η = 0.5
```

using a logarithmic loss scale.

---

# 🛠️ Technologies Used

* **Python**
* **NumPy**
* **Matplotlib**
* **Jupyter Notebook**

No deep-learning framework was used for the core implementation.

In particular, the project does not use:

```python
torch.optim.Adam
```

or any other pre-built optimizer.

---

# 📁 Project Structure

```text
Adam-Optimizer-From-Scratch/
│
├── Adam_Optimizer_From_Scratch.ipynb
├── README.md
└── images/
    ├── adam_theta_vs_iteration.png
    ├── adam_loss_vs_iteration.png
    ├── xor_training_loss.png
    ├── optimizer_comparison.png
    └── learning_rate_comparison.png
```

If the plots are not saved separately, the `images/` directory can be omitted and the results can simply remain inside the Jupyter Notebook.

---

# 🧠 Key Concepts Learned

This project provided hands-on experience with:

* Gradient Descent
* Stochastic Gradient Descent
* Momentum
* Adam
* First moment estimation
* Second moment estimation
* Bias correction
* Adaptive learning rates
* Forward propagation
* Backpropagation
* Chain rule
* Binary Cross-Entropy
* Neural network parameter updates
* Optimization convergence
* Learning-rate selection
* Experimental comparison of optimizers

---

# 💡 Key Takeaways

### 1. Backpropagation and optimization are different

Backpropagation calculates the gradient:

$$
\nabla_\theta L
$$

The optimizer uses that gradient to determine how the parameters should be updated.

---

### 2. Momentum provides memory

Instead of using only the current gradient, Momentum incorporates information from previous gradients.

---

### 3. Adam combines two ideas

Adam tracks:

* The moving average of gradients
* The moving average of squared gradients

This allows it to adapt parameter updates based on both direction and magnitude.

---

### 4. Learning rate matters

Even with the same optimizer, changing the learning rate can dramatically change convergence behavior.

---

### 5. Implementation from scratch builds deeper understanding

Implementing these algorithms manually makes the mathematical operations behind neural-network training much more concrete.

---


# ⭐ Summary

This project implements the **Adam optimizer from scratch using NumPy** and demonstrates its application to training a neural network on the XOR classification problem.

The project progresses from a simple mathematical optimization problem to a complete neural-network training pipeline involving:

$$
\boxed{
\text{Forward Propagation}
\rightarrow
\text{Loss}
\rightarrow
\text{Backpropagation}
\rightarrow
\text{Adam}
\rightarrow
\text{Parameter Update}
}
$$

The project also experimentally compares **SGD, Momentum, and Adam** and investigates how the **learning rate affects Adam's convergence**.

The main goal was not simply to make the model work, but to understand the mathematics and mechanics behind neural-network optimization by implementing the algorithms from scratch.
