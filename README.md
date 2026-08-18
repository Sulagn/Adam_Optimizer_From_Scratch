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

🎯 Objectives

The main objectives of this project were:

Understand gradient-based optimization.
Implement Momentum from scratch.
Implement Adam from scratch.
Understand first and second moment estimates.
Understand Adam's bias correction.
Implement a neural network using NumPy.
Implement backpropagation manually.
Train a neural network on the XOR problem.
Compare SGD, Momentum, and Adam.
Investigate the effect of learning rate on Adam.

🧮 Part 1: Adam on a Toy Optimization Problem

Before using Adam on a neural network, I first tested it on a simple mathematical optimization problem.

The objective function is:

L(θ)=θ
2

The gradient of the objective function is:

dθ
dL
	​

=2θ

The initial parameter was:

θ
0
	​

=10

The goal is to minimize the objective function and move the parameter toward:

θ→0
Adam Update

Adam maintains two exponential moving averages of the gradients.

First Moment

The first moment tracks the moving average of the gradients:

m
t
	​

=β
1
	​

m
t−1
	​

+(1−β
1
	​

)g
t
	​

Second Moment

The second moment tracks the moving average of the squared gradients:

v
t
	​

=β
2
	​

v
t−1
	​

+(1−β
2
	​

)g
t
2
	​


Because both moment estimates are initialized to zero, they are biased toward zero during the early iterations. Adam therefore applies bias correction.

Bias Correction

Corrected first moment:

m
^
t
	​

=
1−β
1
t
	​

m
t
	​

	​


Corrected second moment:

v
^
t
	​

=
1−β
2
t
	​

v
t
	​

	​

Parameter Update

The final Adam update is:

θ
t
	​

=θ
t−1
	​

−η
v
^
t
	​

	​

+ϵ
m
^
t
	​

	​

Hyperparameters
Hyperparameter	Value
Learning rate $\eta$	0.1
$\beta_1$	0.9
$\beta_2$	0.999
$\epsilon$	$10^{-8}$

This experiment demonstrates how Adam uses both the direction and magnitude of recent gradients to adapt parameter updates during optimization.
