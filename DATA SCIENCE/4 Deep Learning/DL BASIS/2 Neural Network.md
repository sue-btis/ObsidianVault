---
tags:
  - deep_learning
  - Python
aliases:
  - Perceptron
  - ANN
  - Artificial Neural Network
creation_date: 2025-07-24
---
[[3 Perceptron ScikitLearn]]
# 🎓 Neural Network

## 🎯 Objective
*What is a Neural Network and how does a Perceptron work?*

Understand the concept of a Neural Network, focusing on the `Perceptron` model, its internal logic, and how to implement one from scratch in Python.

---

## 🧠 The Core Idea (The Analogy)
*Imagine a light switch that decides when to turn on.*

A `Perceptron` is like a tiny decision-maker. It looks at some input values, multiplies them by "importance scores" (called weights), adds everything up, and if the result is big enough—it turns on (outputs 1). Otherwise, it stays off (outputs 0). A Neural Network is just a big team of these switches working together.

---

## ⚙️ How It Works: Step-by-Step

1. **Inputs:** Values from the outside world (e.g., pixels in an image).
2. **Weights:** Each input has a weight telling how important it is.
3. **Weighted Sum:** Multiply inputs by their weights and sum.
4. **Bias:** Add a number that helps shift the result up or down.
5. **Activation Function:** Decide whether to "activate" (output 1) or not (output 0).
6. **Output:** The final result (like a decision).

---

### 🔢 The Perceptron Formula

$$
y = 
\begin{cases}
1 & \text{if } w \cdot x + b > 0 \\
0 & \text{otherwise}
\end{cases}
$$

Where:

- $x$ are the inputs
- $w$ are the weights
- $b$ is the bias
- $w \cdot x$ means the dot product (multiply and sum)

The perceptron activates if the total input is strong enough.

---

## 🧪 My Own Perceptron in Python

```python
# Simple perceptron class from scratch
import numpy as np

class Perceptron:
    def __init__(self, n_inputs, lr=0.1):
        self.weights = np.random.rand(n_inputs)
        self.bias = np.random.rand(1)
        self.lr = lr  # learning rate

    def activation(self, x):
        # Step function: returns 1 if x > 0, else 0
        return 1 if x > 0 else 0

    def predict(self, x):
        total = np.dot(self.weights, x) + self.bias
        return self.activation(total)

    def train(self, X, y, epochs=10):
        for _ in range(epochs):
            for xi, target in zip(X, y):
                prediction = self.predict(xi)
                error = target - prediction
                # Update rule
                self.weights += self.lr * error * xi
                self.bias += self.lr * error

# Example: train to learn AND logic gate
X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([0, 0, 0, 1])  # AND truth table

p = Perceptron(n_inputs=2)
p.train(X, y)

# Test predictions
for xi in X:
    print(f"{xi} -> {p.predict(xi)}")
```

---

## 🧠 Final Thoughts

The `Perceptron` is the building block of modern Neural Networks. It’s a simple yet powerful idea: take inputs, weigh them, and decide. By connecting many perceptrons, we can teach machines to recognize patterns in data—from numbers to images and beyond.
