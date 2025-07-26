---
tags:
  - deep_learning
  - Python
aliases:
  - Perceptron with sklearn
  - Linear Classifier
creation_date: 2025-07-24
---

# 🎓 Perceptron with Scikit-learn

## 🎯 Objective
*How can we use `scikit-learn` to train a Perceptron for classification tasks?*

Learn how to apply the `Perceptron` model using the popular `scikit-learn` library, understand its parameters, and test it on real data.

## 🧠 The Core Idea (The Analogy)
*A Perceptron is like a strict gatekeeper.*

It looks at multiple input signals (features) and decides whether to open the gate (classify as 1) or keep it closed (classify as 0), depending on whether the total weighted signal passes a certain threshold.

## ⚙️ How It Works: Step-by-Step in Scikit-learn

1. **Import the model:** `from sklearn.linear_model import Perceptron`
2. **Load or prepare data:** Use NumPy arrays or datasets like Iris.
3. **Create the model:** Set parameters like learning rate or max iterations.
4. **Train the model:** Use `.fit(X, y)`
5. **Predict:** Use `.predict(X_new)`
6. **Evaluate:** Use accuracy score or confusion matrix.

## 📚 Parameters of `Perceptron`

- `penalty`: Regularization (`'l2'`, `'l1'`, etc.)
- `alpha`: Regularization strength
- `fit_intercept`: Whether to learn the bias
- `max_iter`: Number of epochs
- `eta0`: Learning rate
- `random_state`: Reproducibility

## 💻 Code Example: Perceptron on Iris Dataset

```python
# Train a Perceptron on Iris dataset using scikit-learn
from sklearn.datasets import load_iris
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load data
iris = load_iris()
X = iris.data[:, (0, 2)]  # Use only sepal length and petal length
y = (iris.target == 0).astype(int)  # Binary classification: class 0 vs others

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Create and train Perceptron
clf = Perceptron(max_iter=1000, eta0=1.0, random_state=42)
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# Evaluate accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))
```

## 🔍 Tips and Notes

- The Perceptron in `scikit-learn` only works for **linearly separable** data.
- You can use `StandardScaler` to normalize features before training.
- For multi-class problems, use other models like `LogisticRegression` or `MLPClassifier`.

## 🧠 Final Thoughts

Using `scikit-learn`, we can easily experiment with Perceptrons and see how they perform on real-world datasets. It’s a great way to explore the foundations of linear classification and prepare for more advanced models.
