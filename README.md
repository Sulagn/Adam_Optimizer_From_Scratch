# Adam Optimizer from Scratch

A from-scratch implementation and experimental study of the **Adam optimizer using NumPy**, followed by training a small neural network on the XOR classification problem.

The purpose of this project is to understand the mathematics and mechanics behind optimization algorithms rather than relying on pre-built implementations such as `torch.optim.Adam`.

---

## 📌 Overview

Optimization is one of the fundamental components of neural network training.

In this project, I implemented and studied three optimization methods:

- Stochastic Gradient Descent (SGD)
- Momentum
- Adam

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
