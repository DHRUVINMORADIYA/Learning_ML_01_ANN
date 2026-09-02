# Permanent Notes — Index

## Topics

Distilled notes and the code written for each.

### Linear Neural Network for Regression

- [Notes](Linear%20Neural%20Network%20for%20Regression/Notes.md)
- Code:
  - [L1_From_Scratch.py](Linear%20Neural%20Network%20for%20Regression/L1_From_Scratch.py) — no libraries, gradients by hand
  - [L2_PyTorch.py](Linear%20Neural%20Network%20for%20Regression/L2_PyTorch.py) — autograd + `nn` + `optim`, own training loop
  - [L3_High_Level.py](Linear%20Neural%20Network%20for%20Regression/L3_High_Level.py) — Keras `.fit()`

## Concepts

Core ideas I unlocked, kept at a broad level here and cited from the topic notes.

### Levels of abstraction

When working with a model there are four layers of abstraction, from most manual to most hands-off. Each step hands one more responsibility to a library.

1. **Mathematical layer** — linear algebra, activation functions, loss functions, calculus. Gradients are derived by hand (chain rule).
2. **From scratch** — the derived formulas are taken as given and applied (e.g. plugging into the gradient expression). Still no framework.
3. **Framework, control kept** — a library (PyTorch, TensorFlow) computes gradients via autograd. How much to hand over is a choice: raw autograd (own parameter creation and updates) or `nn` + `optim` (library does those) while still writing the training loop.
4. **High level** — a framework (Keras, Lightning, fastai) runs everything, including the loop. Only data and parameters are configured.

Cited by: [Linear Neural Network for Regression](Linear%20Neural%20Network%20for%20Regression/Notes.md)
