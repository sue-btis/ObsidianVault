#Machine_Learning 
# 🔧 Hyperparameters in Machine Learning Models

## 📌 What are Hyperparameters?

A **hyperparameter** is a setting **defined before training** a machine learning model. Unlike parameters (e.g., weights in linear regression), hyperparameters **are not learned** from the data—they are **manually set** to influence how the model learns.

> 🎯 Think of them as the model's "dials" that we adjust to get better performance.

### 📍 Examples of Hyperparameters

- **k-nearest neighbors (`K`)**  
  Number of neighbors to consider.

- **Decision Trees**  
  - `max_depth`: max number of splits
  - `min_samples_split`: minimum samples required to split a node

- **Linear/Logistic Regression**  
  - `alpha`, `lambda`: regularization strength

---

## ⚖️ Hyperparameters vs Parameters

| Concept       | Hyperparameters                         | Parameters                      |
|---------------|------------------------------------------|---------------------------------|
| Set by        | The user                                 | Learned by the model            |
| Examples      | `k`, `max_depth`, `C`, `kernel`          | Coefficients, weights           |
| Change during training? | ❌ No                         | ✅ Yes                          |

---

## 🎢 The Bias-Variance Tradeoff

Choosing the right hyperparameters affects the **balance between underfitting and overfitting**:

- **High bias** → underfitting (too simple)
- **High variance** → overfitting (too complex)

We want a **compromise**:

$$
\text{Error}_{total} = \text{Bias}^2 + \text{Variance} + \text{Irreducible Error}
$$

---

## 🎯 What is Hyperparameter Tuning?

Hyperparameter tuning means trying different combinations of hyperparameter values to find the set that **optimizes model performance**.

### 🔄 Workflow

1. Choose model and metric (e.g., accuracy).
2. Select range of hyperparameter values.
3. Use training/validation split or cross-validation.
4. Compare performance across combinations.
5. Choose the best set and test on hold-out data.

---

## 🔍 Common Tuning Methods

### 1. 🧮 Grid Search

- **How it works:** Tries **all combinations** from a specified grid of values.
- **Pros:** Exhaustive
- **Cons:** Time-consuming for many parameters

```python
from sklearn.model_selection import GridSearchCV

param_grid = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
grid = GridSearchCV(SVC(), param_grid, cv=5)
grid.fit(X_train, y_train)

print(grid.best_params_)
```

---

### 2. 🎲 Random Search

- **How it works:** Samples **random combinations** from defined distributions.
- **Pros:** Faster than grid search
- **Cons:** May miss optimal settings if not enough iterations

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform

param_dist = {'C': uniform(0, 100), 'penalty': ['l1', 'l2']}
rand = RandomizedSearchCV(LogisticRegression(solver='liblinear'), param_distributions=param_dist, n_iter=10, cv=5)
rand.fit(X_train, y_train)

print(rand.best_params_)
```

---
## 3. 🧠 Bayesian Optimization

- **How it works:** Uses a probabilistic model to estimate performance and **intelligently explore** the hyperparameter space based on past results.
- **Tools:** `Optuna`, `scikit-optimize`, `Hyperopt`
- **Pros:** Sample-efficient, finds optimal configurations faster
- **Cons:** More complex to implement, steeper learning curve

```python
import optuna

def objective(trial):
    C = trial.suggest_loguniform('C', 1e-3, 1e2)
    clf = SVC(C=C)
    return cross_val_score(clf, X_train, y_train, cv=5).mean()

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)

print(study.best_params)
```

---

### 4. 🧬 Genetic Algorithms

- **How it works:** Inspired by natural selection—evolve hyperparameter sets through generations.
- **Tools:** `TPOT`, `DEAP`
- **Pros:** Good for complex spaces and non-continuous variables
- **Cons:** Computationally expensive and slower convergence

```python
from tpot import TPOTClassifier

tpot = TPOTClassifier(generations=5, population_size=20, verbosity=2)
tpot.fit(X_train, y_train)

print(tpot.fitted_pipeline_)
```
---

## 🔁 Cross-Validation in Tuning

When using methods like `GridSearchCV` or `RandomizedSearchCV`, cross-validation ensures that our evaluation is **robust and unbiased**.

```python
from sklearn.model_selection import GridSearchCV

grid = GridSearchCV(model, param_grid, cv=10)
grid.fit(X_train, y_train)

print(grid.best_estimator_)
print(grid.best_score_)
print(grid.best_params_)
```

> 🧠 Tip: Use `cv=5` or `cv=10` depending on dataset size.

---

## 📊 Evaluating Results

Both tuning methods share the same useful attributes:

- `.best_estimator_`: Best model
- `.best_params_`: Best hyperparameter combination
- `.best_score_`: Best validation score
- `.cv_results_`: Full history of all tries
