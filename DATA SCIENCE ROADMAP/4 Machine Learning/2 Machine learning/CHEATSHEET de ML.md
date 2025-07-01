
# 🧠 Machine Learning & Data Science Cheatsheet

Guía exhaustiva y pedagógica para aprender desde cero a nivel intermedio-avanzado en ML y DS.

---

## 📦 1. Exploración, Manipulación y Análisis de Datos con Pandas

```python
import pandas as pd
```

### 🧩 Estructuras de Datos

```python
# Serie
serie = pd.Series([10, 20, 30], index=['a', 'b', 'c'])

# DataFrame
datos = {'Nombre': ['Ana', 'Luis', 'Juan'], 'Edad': [23, 45, 34]}
df = pd.DataFrame(datos)
```

### 📥 Cargar y Crear DataFrames

```python
df = pd.read_csv('archivo.csv')
data = np.array([[1, 2, 3], [4, 5, 6]])
df = pd.DataFrame(data, columns=['A', 'B', 'C'])
```

### 🔍 Selección de Datos

```python
df['A']         # Columna
df.iloc[0]      # Por posición
df.loc[0, 'A']  # Por etiqueta
```

### ➕ Transformación y Filtrado

```python
df['D'] = df['A'] + df['B']
df['E'] = df['B'].apply(lambda x: x * 2)
df = df[df['A'] > 2]
df.drop('D', axis=1, inplace=True)
```

### 📊 Estadísticas Básicas

```python
df.mean(), df.median(), df.std()
df.describe()
df.corr(), df.cov()
```

### 🔄 Agrupamiento y Agregaciones

```python
df.groupby('A').agg({'B': ['mean', 'sum'], 'C': 'max'})
df.agg({'A': ['mean'], 'B': ['min', 'max']})
```

### 📈 Pivot Tables

```python
df.pivot_table(values='Sales', index='City', columns='Year', aggfunc='sum')
```

### 🔗 Merge y Join

```python
pd.merge(df1, df2, on='ID', how='inner')
pd.concat([df1, df2], axis=0, ignore_index=True)
```

---

## 🔢 2. Manipulación Numérica y Álgebra Lineal con NumPy

```python
import numpy as np
```

### 🧮 Crear Arrays

```python
v = np.array([1, 2, 3])
A = np.array([[1, 2], [3, 4]])
I = np.eye(3)
Z = np.zeros((2, 3))
```

### 🔄 Transformaciones

```python
v.reshape((3, 1))
A.T  # Transpuesta
```

### 🔗 Construcción y Broadcasting

```python
np.column_stack((col1, col2))
v + 5  # Broadcasting
np.sum(v)
```

### 💥 Operaciones Elementales

```python
A + B
A * B
np.dot(A, B)
A @ B  # Multiplicación matricial
```

### 📐 Álgebra Lineal

```python
np.linalg.inv(A)
np.linalg.det(A)
eig_vals, eig_vecs = np.linalg.eig(A)
np.linalg.solve(A, b)
```

---

## 🔄 Flujo Básico de un Proyecto de Machine Learning

1. **Recolección de datos**: desde CSV, APIs o bases de datos.
2. **Limpieza**: manejo de datos faltantes, duplicados, valores extremos.
3. **Análisis exploratorio**: `describe()`, `groupby()`, `value_counts()`.
4. **Visualización**: `matplotlib`, `seaborn` para entender la distribución.
5. **Preparación de datos**: codificación, escalado, división train/test.
6. **Entrenamiento del modelo**: scikit-learn (`LinearRegression`, `KNeighborsClassifier`, etc.).
7. **Evaluación del modelo**: métricas (accuracy, MSE, F1-score...).
8. **Ajuste y validación cruzada**: `GridSearchCV`, regularización.
9. **Despliegue**: exportar con `joblib`, usar en APIs o dashboards.

---

(En la siguiente parte se incluirán modelos de ML supervisado y no supervisado, métricas y visualización avanzada.)

---
## 👨‍🏫 Conceptos Clave

### 🎯 Variables X (features) e y (target)

- **X**: matriz de características (variables independientes)
- **y**: variable objetivo (lo que queremos predecir)

```python
X = df.drop("target", axis=1)
y = df["target"]
```

### ⚠️ Overfitting vs. Underfitting

- **Overfitting**: El modelo aprende demasiado bien los datos de entrenamiento y falla al generalizar.
- **Underfitting**: El modelo no logra captar la relación subyacente en los datos.

| Tipo de error    | Causa principal                    | Solución común                  |
|------------------|------------------------------------|----------------------------------|
| Overfitting       | Modelo demasiado complejo          | Regularización, más datos       |
| Underfitting      | Modelo demasiado simple            | Modelo más complejo             |

### ⚖️ Bias-Variance Tradeoff

- **Bias**: Error por suposiciones incorrectas del modelo.
- **Variance**: Sensibilidad del modelo a pequeñas variaciones en los datos.

Ideal: bajo bias y baja varianza → buen desempeño generalizado.

---

# 🔢 Parte 2.5: Preprocesamiento Numérico y Categórico

## 🔢 Transformaciones Numéricas

### 1. 🎯 Centering (Centrado)
$$
x_{centered} = x - \bar{x}
$$

```python
x_centered = x - np.mean(x)
```

---

### 2. ⚖️ Standard Scaler (Estandarización)
$$
x_{std} = \frac{x - \mu}{\sigma}
$$

```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x.reshape(-1, 1))
```

---

### 3. 📊 Min-Max Scaler
$$
x' = \frac{x - x_{min}}{x_{max} - x_{min}}
$$

```python
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
x_minmax = scaler.fit_transform(x.reshape(-1, 1))
```

---

### 4. 🧱 Binning (Discretización)

```python
x_binned = pd.cut(x, bins=3, labels=["bajo", "medio", "alto"])
```

---

### 5. 🔁 Transformaciones Logarítmicas
$$
x' = \log(x + 1)
$$

```python
x_log = np.log1p(x)
```

---

## 🧩 Codificación de Variables Categóricas

### 🔢 Ordinal Encoding

```python
from sklearn.preprocessing import OrdinalEncoder
encoder = OrdinalEncoder(categories=[['Excellent', 'New', 'Like New', 'Good', 'Fair']])
encoded = encoder.fit_transform(cars['condition'].values.reshape(-1,1))
```

---

### 🏷️ Label Encoding

```python
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
cars['color'] = encoder.fit_transform(cars['color'])
```

---

### 🟦 One-Hot Encoding

```python
ohe = pd.get_dummies(cars['color'])
cars = cars.join(ohe)
```

---

### 🔢 Binary Encoding

```python
from category_encoders import BinaryEncoder
colors = BinaryEncoder(cols=['color']).fit_transform(cars)
```

---

### 💠 Hashing Encoding

```python
from category_encoders import HashingEncoder
encoder = HashingEncoder(cols='color', n_components=5)
hash_results = encoder.fit_transform(cars['color'])
```

---

### 🎯 Target Encoding

```python
from category_encoders import TargetEncoder
encoder = TargetEncoder(cols='color')
encoded = encoder.fit_transform(cars['color'], cars['sellingprice'])
```

---

### ⏰ Encoding de Fechas

```python
cars['saledate'] = pd.to_datetime(cars['saledate'])
cars['month'] = cars['saledate'].dt.month
cars['dayofweek'] = cars['saledate'].dt.dayofweek
cars['yearbuild_sold'] = cars['saledate'].dt.year - cars['year']
```

---

### ✅ Conclusión

| Transformación      | ¿Cuándo usarla?                                |
|---------------------|------------------------------------------------|
| Centering           | PCA o modelos lineales                         |
| Standard Scaler     | Modelos sensibles a escala (SVM, regresión)    |
| Min-Max Scaler      | Redes neuronales, normalización entre [0, 1]  |
| Binning             | Modelos de reglas, simplificación              |
| Log Transform       | Reducción de asimetría                         |

| Codificación         | ¿Cuándo usarla?                               |
|----------------------|-----------------------------------------------|
| Ordinal              | Categorías ordenadas                          |
| Label                | Nominal simple                                |
| One-Hot              | Nominal sin orden                             |
| Binary               | Muchas categorías                             |
| Hashing              | Gran volumen, menos interpretabilidad         |
| Target               | Regresión, correlación con variable objetivo  |

---
# Parte 3 :📊 Tabla Global de Modelos de Machine Learning en Scikit-learn

Guía de referencia rápida organizada por tipo de modelo, con columnas clave para selección, preprocesamiento, interpretación y ajuste.

---

## ✅ Clasificación

| Modelo                  | Clase `scikit-learn`                          | Lineal | Escalado Necesario | Sensible a Outliers | Interpretable | Predice Probabilidades | Soporta Multiclase | Hiperparámetros Clave          |
|-------------------------|-----------------------------------------------|--------|---------------------|---------------------|----------------|------------------------|---------------------|-------------------------------|
| Logistic Regression     | `sklearn.linear_model.LogisticRegression`     | Sí     | ✅ Sí               | Alta                | ✅ Alta        | ✅ Sí                  | ✅ Sí               | `C`, `penalty`, `solver`      |
| K-Nearest Neighbors     | `sklearn.neighbors.KNeighborsClassifier`      | No     | ✅ Sí               | Alta                | Media          | ❌ No                 | ✅ Sí               | `n_neighbors`, `weights`      |
| Decision Tree           | `sklearn.tree.DecisionTreeClassifier`         | No     | ❌ No               | ✅ Baja             | Media          | ❌ No                 | ✅ Sí               | `max_depth`, `min_samples_split` |
| Random Forest           | `sklearn.ensemble.RandomForestClassifier`     | No     | ❌ No               | ✅ Baja             | ❌ Baja         | ✅ Sí                  | ✅ Sí               | `n_estimators`, `max_depth`   |
| Support Vector Machine  | `sklearn.svm.SVC`                              | Sí     | ✅ Sí               | Alta                | ❌ Baja         | ✅ Sí (`probability`)  | ✅ Sí               | `C`, `kernel`, `gamma`        |
| Gaussian Naive Bayes    | `sklearn.naive_bayes.GaussianNB`              | Sí     | Recomendado        | Alta                | ✅ Alta        | ✅ Sí                  | ✅ Sí               | -                             |

---

## 📈 Regresión

| Modelo                  | Clase `scikit-learn`                          | Lineal | Escalado Necesario | Sensible a Outliers | Interpretable | Soporta Multioutput | Hiperparámetros Clave           |
|-------------------------|-----------------------------------------------|--------|---------------------|---------------------|----------------|----------------------|-------------------------------|
| Linear Regression       | `sklearn.linear_model.LinearRegression`       | Sí     | ✅ Sí               | Alta                | ✅ Alta        | ✅ Sí                | -                             |
| Ridge                   | `sklearn.linear_model.Ridge`                  | Sí     | ✅ Sí               | Alta                | ✅ Alta        | ✅ Sí                | `alpha`                       |
| Lasso                   | `sklearn.linear_model.Lasso`                  | Sí     | ✅ Sí               | Alta                | ✅ Alta        | ✅ Sí                | `alpha`                       |
| K-Nearest Regressor     | `sklearn.neighbors.KNeighborsRegressor`       | No     | ✅ Sí               | Alta                | Media          | ✅ Sí                | `n_neighbors`, `weights`      |
| Decision Tree Regressor| `sklearn.tree.DecisionTreeRegressor`          | No     | ❌ No               | ✅ Baja             | Media          | ✅ Sí                | `max_depth`, `min_samples_split` |
| Random Forest Regressor| `sklearn.ensemble.RandomForestRegressor`      | No     | ❌ No               | ✅ Baja             | ❌ Baja         | ✅ Sí                | `n_estimators`, `max_depth`   |
| Support Vector Regressor| `sklearn.svm.SVR`                             | No     | ✅ Sí               | Alta                | ❌ Baja         | ❌ No                | `C`, `kernel`, `epsilon`      |

---

## 📊 Clustering (No Supervisado)

| Modelo                  | Clase `scikit-learn`                          | Escalado Necesario | Sensible a Outliers | Necesita k | Predice Cluster (`predict`) | Métricas comunes             |
|-------------------------|-----------------------------------------------|---------------------|---------------------|------------|------------------------------|------------------------------|
| KMeans                  | `sklearn.cluster.KMeans`                      | ✅ Sí               | ✅ Alta             | ✅ Sí     | ✅ Sí                        | Inertia, Silhouette Score    |
| DBSCAN                  | `sklearn.cluster.DBSCAN`                      | ✅ Sí               | ✅ Alta             | ❌ No     | ❌ No (`labels_`)            | Silhouette, Davies-Bouldin   |
| Agglomerative Clustering| `sklearn.cluster.AgglomerativeClustering`     | ✅ Sí               | ✅ Alta             | ✅ Sí     | ❌ No (`labels_`)            | Silhouette, Dendrogram       |
| Gaussian Mixture (EM)   | `sklearn.mixture.GaussianMixture`            | ✅ Sí               | Alta                | ✅ Sí     | ✅ Sí                        | Log-likelihood, AIC, BIC     |

---

## 🧰 Herramientas Complementarias

- `Pipeline`: Secuencia de pasos (`StandardScaler` → `Model`).
- `GridSearchCV`: Búsqueda exhaustiva de hiperparámetros.
- `cross_val_score`: Validación cruzada rápida.
- `train_test_split`: Separar `X_train`, `X_test`, `y_train`, `y_test`.
- `StandardScaler`, `MinMaxScaler`: Normalización / escalado.
- `OneHotEncoder`: Para variables categóricas.
- `SelectKBest`, `PCA`: Selección / reducción de características.

---

> 🧠 **Consejo:** Usa `Pipeline` junto con `GridSearchCV` para combinar normalización, selección de variables y modelo en un solo flujo reproducible.

---

# 🤖 Parte 4: Modelado Supervisado
## 📈 Regresión Lineal

Modelo básico con una variable:

$$
\hat{y} = mx + b
$$

### ✅ Supuestos

1. Linealidad
2. Homocedasticidad
3. Independencia de errores
4. Normalidad de los residuos

### 📌 Simple Linear Regression

```python
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

x = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
y = np.array([1.5, 2.0, 3.5, 3.7, 4.5])

model = LinearRegression()
model.fit(x, y)

y_pred = model.predict(x)
plt.scatter(x, y, label="Datos reales")
plt.plot(x, y_pred, color="red", label="Regresión")
plt.legend()
plt.title("Regresión Lineal Simple")
plt.show()
```

---

### 📌 Multiple Linear Regression

```python
data = pd.DataFrame({
    "horas_estudio": [1, 2, 3, 4, 5],
    "horas_sueño": [8, 7, 6, 5, 4],
    "nota": [60, 65, 70, 75, 80]
})
X = data[["horas_estudio", "horas_sueño"]]
y = data["nota"]

model = LinearRegression()
model.fit(X, y)
print("Coeficientes:", model.coef_)
print("Intercepto:", model.intercept_)
```

---

### 📊 Correlación

```python
import seaborn as sns
sns.heatmap(data.corr(), annot=True, cmap="coolwarm")
plt.title("Matriz de Correlación")
plt.show()
```

---

### 📐 Evaluación del Modelo

* MAE: $\frac{1}{n} \sum |y - \hat{y}|$
- MSE: $\frac{1}{n} \sum (y - \hat{y})^2$
- RMSE: $\sqrt{MSE}$
- R² Score: Varianza explicada por el modelo

```python
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

y_pred = model.predict(X)
print("MAE:", mean_absolute_error(y, y_pred))
print("MSE:", mean_squared_error(y, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y, y_pred)))
print("R²:", r2_score(y, y_pred))
```

---

## 🧮 OLS vs Gradient Descent

| Método              | OLS                          | Gradient Descent (GD)        |
|---------------------|-------------------------------|-------------------------------|
| Tipo                | Analítico                     | Iterativo                     |
| Exactitud           | Solución exacta               | Aproximación                  |
| Velocidad (pocos datos) | Rápido                     | Más lento                     |
| Velocidad (big data)    | Lento (álgebra matricial)  | Escalable                     |
| Requiere ajuste     | No                            | Sí (learning rate, epochs)    |

### OLS Fórmula:

$$
\beta = (X^T X)^{-1} X^T y
$$

### GD Fórmulas:

$$
m \leftarrow m - \eta \cdot \frac{\partial}{\partial m} \text{Loss},\quad
b \leftarrow b - \eta \cdot \frac{\partial}{\partial b} \text{Loss}
$$

---

## 🔁 Regresión Logística

### 📘 Definición

Predice una variable categórica (binaria) usando una transformación sigmoide.
$$
P(y = 1 | x) = \frac{1}{1 + e^{-(b + m_1 x_1 + \dots + m_n x_n)}}
$$

### 🧪 Implementación

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LogisticRegression(class_weight='balanced')
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

---

### 📊 Métricas y Matriz de Confusión

- Accuracy
- Precision, Recall
- F1 Score
- ROC-AUC

```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
```

> ⚠️ Usa `predict_proba` para obtener probabilidades y ajustar el threshold si es necesario.
---

### 📈 ROC Curve & AUC

```python
from sklearn.metrics import roc_curve, roc_auc_score

y_proba = model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_proba)
auc = roc_auc_score(y_test, y_proba)

plt.plot(fpr, tpr, label=f'AUC = {auc:.2f}')
plt.plot([0, 1], [0, 1], '--', color='gray')
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("ROC Curve")
plt.legend()
plt.show()
```

---

### ⚙️ Ajuste de Threshold

```python
y_pred_custom = (y_proba > 0.3).astype(int)
```

---

## 🧠 Consideraciones Avanzadas

- Usa `class_weight="balanced"` si hay desbalance.
- Revisa correlaciones para evitar multicolinealidad.
- Normaliza si las variables tienen escalas diferentes.
- Visualiza con curvas ROC y matriz de confusión.

---

## 🔍 K-Nearest Neighbors (KNN)

### 🧠 Idea Principal

KNN predice un resultado **basado en los K vecinos más cercanos** en el espacio de características.

- **KNN Clasificación**: vota la clase más común entre los vecinos ( 0 o 1).
- **KNN Regresión**: promedia el valor de salida de los vecinos( valores numéricos como estrellas  de una película) .

### 📏 Recomendaciones generales

- Siempre **escalar los datos** (StandardScaler o MinMaxScaler) antes de usar KNN.
- Seleccionar el valor óptimo de K es **crítico** (ni muy bajo ni muy alto).
- El parámetro `weights` puede cambiar significativamente el resultado:
  - `'uniform'`: todos los vecinos tienen el mismo peso.
  - `'distance'`: los vecinos más cercanos tienen más influencia.

---

### 📘 KNN para Clasificación

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

# Ajustar K automáticamente
param_grid = {'n_neighbors': range(1, 21), 'weights': ['uniform', 'distance']}
grid = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5)
grid.fit(X_train, y_train)

print("Mejor K:", grid.best_params_['n_neighbors'])
print("Pesos usados:", grid.best_params_['weights'])

# Evaluar
y_pred = grid.predict(X_test)
print(classification_report(y_test, y_pred))
```

---

### 📘 KNN para Regresión

```python
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error

reg = KNeighborsRegressor(n_neighbors=5, weights='distance')
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)

print("MSE:", mean_squared_error(y_test, y_pred))
```

---

### 🧪 Comparación de pesos en regresión

```python
for weight in ['uniform', 'distance']:
    model = KNeighborsRegressor(n_neighbors=5, weights=weight)
    model.fit(X_train, y_train)
    print(f"{weight} MSE:", mean_squared_error(y_test, model.predict(X_test)))
```

---

### 🧠 Elección de K

- K muy pequeño → sobreajuste (modelo muy flexible).
- K muy grande → subajuste (modelo demasiado general).
- Se recomienda probar con `GridSearchCV` o validación cruzada.

---

## 🌳 Árboles de Decisión (Decision Trees)

### 🧠 Idea Principal

Dividen el espacio de decisiones en **reglas tipo sí/no** según las características más importantes.

- Se crean nodos que maximizan la **ganancia de información** o reducen la **impureza de Gini**.

---

### 📘 Criterios de división

- **Gini Impurity**:
  $$
  Gini = 1 - \sum_{i=1}^C p_i^2
  $$

- **Entropy / Information Gain**:
  $$
  Entropy = - \sum_{i=1}^C p_i \log_2 p_i
  $$

  $$
  IG = Entropy_{parent} - \sum \left( \frac{n_{child}}{n_{total}} \cdot Entropy_{child} \right)
  $$

---

### 💻 Implementación

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

model = DecisionTreeClassifier(criterion='gini', max_depth=3)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(classification_report(y_test, y_pred))
plot_tree(model, filled=True, feature_names=X.columns)
plt.show()
```

---

### ⚙️ Hiperparámetros clave

- `criterion`: `'gini'` o `'entropy'`
- `max_depth`: profundidad máxima del árbol
- `min_samples_split`: mínimo de muestras para dividir un nodo
- `min_samples_leaf`: mínimo de muestras en una hoja

---

### ✅ Ventajas y Desventajas

| Pros                           | Contras                             |
|--------------------------------|--------------------------------------|
| Fácil de interpretar           | Tienden al sobreajuste               |
| No necesita normalización      | Sensibles a pequeños cambios         |
| Acepta variables categóricas   | Árboles muy grandes son difíciles de manejar |

---

> 🌱 Para mejorar los árboles individuales, se usan **Random Forests** y **Boosting**, que se explicarán más adelante.

