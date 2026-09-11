---
title: 01 Linear Neural Network for Regression
parent: Permanent Notes
nav_order: 2
---

# 01 · Linear Neural Network for Regression

Builds on [00 · Basics](../00-basics/Notes.md).

## Summary

Linear Neural Network is the most primal form of neural networks. We can see it as a single neuron separated out from a bigger network. The complex neural networks contain more of this, all nicely linked together.

Described as a linear algebra function,

$$
\hat{y} = w_1 x_1 + w_2 x_2 + \cdots + w_n x_n + b
$$

If used as matrices,

$$
\hat{y} =
\begin{bmatrix} w_1 & w_2 & \cdots & w_n \end{bmatrix}
\begin{bmatrix} x_1 \\ x_2 \\ \vdots \\ x_n \end{bmatrix}
+ b
$$

where $x_1, x_2, \dots, x_n$ are input values and $w_1, w_2, \dots, w_n$ are weights describing how much importance we are giving to each corresponding input value while coming up with the prediction $\hat{y}$; $b$ is bias. With bias we give a bit of firm identity to each particular neuron. We assign each neuron its own parameter that is not dependent on any input or any other neuron.

The above function can also be viewed geometrically, as vectors in n-dimensional space.

## Practice files

Three implementations of the same linear regression, each one removing a layer of manual work. See [levels of abstraction](../Index.md#levels-of-abstraction).

### [L1_From_Scratch.py](https://github.com/DHRUVINMORADIYA/Learning_ML_01_ANN/blob/main/permanent-notes/01-linear-neural-network-for-regression/L1_From_Scratch.py){:target="_blank"}

A simple implementation without use of libraries, to build intuition around the inner logic.

Implemented concepts:

1. OOP in ML (Data, Model, Trainer)
2. Making use of synthetic data (Gaussian distribution)
3. Generalization and weight decay (L2 regularization)
4. Forward pass, MSE, gradients, SGD update

### [L2_PyTorch.py](https://github.com/DHRUVINMORADIYA/Learning_ML_01_ANN/blob/main/permanent-notes/01-linear-neural-network-for-regression/L2_PyTorch.py){:target="_blank"}

One level up on abstraction, where we let PyTorch do the low-level math. Use of autograd was one of the most important parts at this level.

### [L3_High_Level.py](https://github.com/DHRUVINMORADIYA/Learning_ML_01_ANN/blob/main/permanent-notes/01-linear-neural-network-for-regression/L3_High_Level.py){:target="_blank"}

High-level code, and I assume that's what people should be using in real life. We use Keras or similar frameworks that take care of everything. We just give data and configure parameters.
