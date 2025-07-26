#Machine_Learning 
#Supervised_Learning
#Unsupervised_Learning
# Recommendation Systems with Surprise

Recommendation systems help users discover items they might like, such as movies, products, or books. One popular Python library for building recommender systems is `Surprise` (Simple Python Recommendation System Engine).

It provides easy-to-use tools for building collaborative filtering models, especially matrix factorization and neighborhood-based methods.

## 💡 Types of Recommendation Systems

There are three main types:

1. **Content-based filtering**: Recommends items similar to those the user liked in the past.
2. **Collaborative filtering**: Uses the preferences of similar users.
3. **Hybrid models**: Combine both approaches.

`Surprise` focuses on collaborative filtering.

---

## 🔧 Installing Surprise

To install the library, use:

```python
pip install scikit-surprise
```

---

## 📊 Dataset Structure

Surprise expects data in this format:

| userID | itemID | rating |
|--------|--------|--------|
| 1      | 101    | 4.0    |
| 2      | 103    | 5.0    |

You can load your own data or use built-in datasets like `MovieLens`.

---

## 📥 Loading Data

```python
from surprise import Dataset
from surprise import Reader

# Example: loading data from a DataFrame
import pandas as pd

ratings_dict = {
    "userID": [1, 1, 1, 2, 2],
    "itemID": ['A', 'B', 'C', 'A', 'B'],
    "rating": [4, 3, 5, 2, 4]
}

df = pd.DataFrame(ratings_dict)

reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(df[["userID", "itemID", "rating"]], reader)
```

---

## ✅ Training a Basic Model

Let’s use the **SVD (Singular Value Decomposition)** algorithm:

```python
from surprise import SVD
from surprise.model_selection import cross_validate

algo = SVD()
cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
```

---

## 📈 Predicting Ratings

```python
from surprise import train_test_split

# Split the data
trainset, testset = train_test_split(data, test_size=0.25)

# Train
algo.fit(trainset)

# Predict
predictions = algo.test(testset)

# Check a single prediction
print(predictions[0])
```

Output:

```
Prediction(uid=1, iid='B', r_ui=3.0, est=3.52, details={'was_impossible': False})
```

---

## 🎯 Making a Custom Prediction

You can predict the rating a specific user would give to an item:

```python
uid = '1'  # User ID
iid = 'C'  # Item ID

pred = algo.predict(uid, iid)
print(pred.est)
```

---

## 📌 Summary

- `Surprise` is a powerful library for collaborative filtering.
- You can train models like **SVD**, **KNN**, and **SVD++**.
- Built-in tools support evaluation using **RMSE** and **MAE**.
- Easy to integrate with custom data.