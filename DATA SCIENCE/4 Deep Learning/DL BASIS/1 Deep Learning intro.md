---
tags:
  - deep_learning
  - Python
aliases:
  - Deep Neural Networks
  - DL
creation_date: 2025-07-24
---
[[2 Neural Networks]]
# 🎓 Deep Learning

## 🎯 Objective
*What is Deep Learning and how does it work?*

Understand the basics of Deep Learning, how it differs from Machine Learning, and learn key concepts like `tensors`, `loss function`, `gradient descent`, and `backpropagation` through clear analogies and examples.

---

## 🧠 The Core Idea (The Analogy)
*Imagine your brain is a big city full of lights.*

Each light represents a tiny decision: turn on or off, connect or not. Deep Learning is like building an artificial city with many lights called `neurons`. These lights are organized in layers like floors in a building and learn to light up correctly by observing thousands of examples—just like learning from experience.

---

## ⚙️ How It Works: Step-by-Step

1. **Input Layer:** Data (e.g. an image) is turned into numbers and enters as a set of `tensors` (multidimensional arrays).
2. **Hidden Layers:** Each neuron performs a math operation (like multiply and sum) and passes the info to the next layer.
3. **Activation Function:** A function decides whether a neuron should "activate" or not. Common example: `ReLU`.
4. **Loss Function:** Measures how wrong the prediction was.
5. **Backpropagation:** Like rewinding to fix mistakes, this adjusts the network's connections.
6. **Gradient Descent:** A technique to improve the model by moving in the direction that reduces the error—like going downhill.

---

### 🧮 Math Behind It: Tensors & Functions

- **Tensor:** Think of it as a magical box that stores numbers. A tensor can be:
  - Scalar (number): $x = 7$
  - Vector (list): $v = [1, 2, 3]$
  - Matrix: 
    $$
    M = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}
    $$
  - 3D Tensor or more: Like a box full of paper sheets (each sheet is a matrix).

- **Loss Function:** Measures how bad the prediction was. For example, `Mean Squared Error (MSE)` is:

  $$
  L = \frac{1}{n} \sum_{i=1}^{n}(y_i - \hat{y}_i)^2
  $$

  Where $y_i$ is the actual value and $\hat{y}_i$ is the predicted value.

- **Gradient Descent:**

  $$ 
  \theta = \theta - \alpha \cdot \nabla_\theta L(\theta)
  $$

  This means: update your parameters $\theta$ in the opposite direction of the error to improve the model.

---

### 🔁 Backpropagation
This process moves backward through the network, calculating how much each neuron contributed to the final error. Then it adjusts the "weights" using `gradient descent`.

---

## 🎲 Stochastic & Variants of Gradient Descent

- **Batch Gradient Descent:** Uses all the data before adjusting weights. Slow.
- **Stochastic Gradient Descent (SGD):** Adjusts after each example. Faster, but noisy.
- **Mini-Batch Gradient Descent:** A balance—uses small data groups to combine speed and stability.

---

## 🤖 Difference Between Machine Learning & Deep Learning

| Concept               | Machine Learning (ML)                 | Deep Learning (DL)                                 |
|-----------------------|---------------------------------------|----------------------------------------------------|
| Requires Feature Engineering | ✅ Yes, manual effort            | ❌ Learns directly from raw data                    |
| Data Needed            | Small to medium                      | Large datasets (Big Data)                          |
| Interpretability       | High (easier to understand)          | Low (acts like a black box)                        |
| Model Examples         | Decision Tree, SVM, KNN              | CNN, RNN, Transformer                              |

---

## ⚠️ Precautions with Deep Learning

- ❗ **Needs lots of data:** Without enough, it may learn incorrectly.
- ❗ **Can overfit:** Learns by heart instead of generalizing.
- ❗ **Hard to interpret:** Not always clear why it made a decision.
- ❗ **Ethical concerns:** May reproduce bias if trained on unbalanced data.

---

### 💻 Code Example
*A basic neural network using `TensorFlow`.*

```python
# Simple neural network with TensorFlow and Keras
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Create a model with 1 hidden layer
model = Sequential([
    Dense(16, activation='relu', input_shape=(10,)),  # Hidden layer with 16 neurons
    Dense(1, activation='sigmoid')  # Output layer for binary classification
])

# Compile with loss function and optimizer
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Fake data for training
import numpy as np
X = np.random.rand(100, 10)  # 100 samples, 10 features
y = np.random.randint(0, 2, size=(100,))  # 0 or 1 labels

# Train the model
model.fit(X, y, epochs=5, batch_size=10)
```
