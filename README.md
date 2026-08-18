# Adam Optimizer from Scratch

A from-scratch implementation and experimental study of the **Adam optimizer** using NumPy, followed by training a small neural network on the XOR problem.

The goal of this project is to understand what happens inside an optimizer rather than relying on pre-built implementations such as `torch.optim.Adam`.

---

## 📌 Overview

Optimization is one of the fundamental components of neural network training.

In this project, I implemented and studied three optimization methods:

* Stochastic Gradient Descent (SGD)
* Momentum
* Adam

The project progresses from a simple mathematical optimization problem to a complete neural network trained from scratch.

### Main workflow

```text
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

The main objectives of this project were:

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

[
L(\theta)=\theta^2
]

The gradient is:

[
\frac{dL}{d\theta}=2\theta
]

The initial parameter was:

[
\theta_0=10
]

The goal was to minimize the function and move:

[
\theta \rightarrow 0
]

### Adam update

Adam maintains two moving averages.

### First moment

[
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t
]

### Second moment

[
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2
]

Because the moment estimates start at zero, Adam uses bias correction:

[
\hat m_t=\frac{m_t}{1-\beta_1^t}
]

[
\hat v_t=\frac{v_t}{1-\beta_2^t}
]

The final parameter update is:

[
\theta_t=
\theta_{t-1}
------------

\eta
\frac{\hat m_t}
{\sqrt{\hat v_t}+\epsilon}
]

### Hyperparameters

```text
Learning rate = 0.1
β1 = 0.9
β2 = 0.999
ε = 1e-8
```

---

## 🧠 Part 2: Neural Network from Scratch

After testing Adam on a simple function, I implemented a small neural network using only NumPy.

### Architecture

```text
2 Input Neurons
       ↓
4 Hidden Neurons
       ↓
1 Output Neuron
```

The hidden layer uses:

[
\tanh(x)
]

and the output layer uses:

[
\sigma(x)=\frac{1}{1+e^{-x}}
]

### Forward propagation

The hidden layer:

[
Z_1=XW_1+b_1
]

[
A_1=\tanh(Z_1)
]

The output layer:

[
Z_2=A_1W_2+b_2
]

[
\hat y=\sigma(Z_2)
]

---

## 📉 Loss Function

For the XOR classification problem, Binary Cross-Entropy was used:

[
L=
-\frac{1}{n}
\sum_i
\left[
y_i\log(\hat y_i)
+
(1-y_i)\log(1-\hat y_i)
\right]
]

---

## 🔄 Backpropagation

The gradients were calculated manually using the chain rule.

The gradients calculated were:

```text
dW1
db1
dW2
db2
```

These gradients were then passed to the optimizer.

The training process was:

```text
Forward Pass
     ↓
Calculate Loss
     ↓
Backpropagation
     ↓
Calculate Gradients
     ↓
Optimizer
     ↓
Update Weights & Biases
     ↓
Repeat
```

---

## 🚀 Training XOR with Adam

The XOR dataset is:

| Input    | Target |
| -------- | -----: |
| `[0, 0]` |      0 |
| `[0, 1]` |      1 |
| `[1, 0]` |      1 |
| `[1, 1]` |      0 |

The network was trained for **5000 epochs** using Adam.

### Training result

Final Binary Cross-Entropy loss:

```text
0.000006
```

### Predictions

```text
[1.14003288e-07]
[9.99994716e-01]
[9.99990093e-01]
[8.68342740e-06]
```

Rounded:

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

The network successfully learned the XOR mapping.

---

## ⚖️ SGD vs Momentum vs Adam

The same neural network initialization was used for all three optimizers to make the comparison fair.

### Results after 5000 epochs

| Optimizer | Final Loss |
| --------- | ---------: |
| SGD       |   0.000970 |
| Momentum  |   0.000075 |
| Adam      |   0.000006 |

### Interpretation

SGD uses only the current gradient:

[
\theta_{t+1}=\theta_t-\eta g_t
]

Momentum incorporates information from previous gradients:

[
v_t=\beta v_{t-1}+g_t
]

Adam combines momentum-like first-moment tracking with second-moment adaptive scaling.

For this particular XOR experiment:

```text
SGD       → Slowest convergence
Momentum  → Faster convergence
Adam      → Fastest convergence
```

This result is specific to the chosen architecture, initialization, hyperparameters, and dataset. It should not be interpreted as Adam being universally superior to every other optimizer.

---

## 📊 Learning Rate Experiment

I also investigated how Adam's learning rate affects convergence.

The network was trained for 1000 epochs using different learning rates.

| Learning Rate | Final Loss |
| ------------: | ---------: |
|         0.001 |   0.411976 |
|          0.01 |   0.005772 |
|           0.1 |   0.000158 |
|           0.5 |   0.000061 |

### Observation

A learning rate that is too small can result in very slow convergence.

For this experiment:

```text
0.001 → Too slow
0.01  → Better
0.1   → Fast convergence
0.5   → Fastest among tested values
```

However, increasing the learning rate indefinitely is not beneficial. An excessively large learning rate can cause unstable updates, oscillation, divergence, or numerical problems.

---

## 🛠️ Technologies Used

* Python
* NumPy
* Matplotlib
* Jupyter Notebook

No deep-learning framework was used for the implementation.

In particular, the project does **not** use:

```python
torch.optim.Adam
```

or another pre-built optimizer.

---

## 📁 Project Structure

```text
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

The exact notebook filename can be changed to match the file uploaded to the repository.

---

## 🔬 Key Concepts Learned

Through this project, I worked through:

* Gradient Descent
* Stochastic Gradient Descent
* Momentum
* Adaptive optimization
* Adam
* First moment estimation
* Second moment estimation
* Bias correction
* Learning rates
* Forward propagation
* Backpropagation
* Chain rule
* Binary Cross-Entropy
* Neural network parameter updates
* Optimization convergence
* Experimental comparison of optimizers

---

## 📈 Key Takeaways

The main lessons from this implementation were:

### 1. Backpropagation and optimization are different

Backpropagation calculates:

[
\nabla_\theta L
]

The optimizer determines how those gradients are used to update the parameters.

### 2. Momentum provides gradient memory

Instead of responding only to the current gradient, Momentum incorporates previous gradients.

### 3. Adam combines two ideas

Adam tracks both:

* the direction of recent gradients
* the magnitude of recent gradients

This allows adaptive parameter updates.

### 4. Hyperparameters matter

The learning rate can dramatically change the training behavior of an optimizer.

### 5. Implementing algorithms from scratch improves understanding

Writing the optimizer manually makes the mathematical operations behind neural network training much more concrete.

---

## ⭐ Summary

This project implements **Adam from scratch using NumPy** and demonstrates its use in training a neural network on XOR.

The project moves from the mathematical definition of Adam to a complete neural-network training pipeline, followed by controlled experiments comparing **SGD, Momentum, and Adam** and investigating the effect of the learning rate.
