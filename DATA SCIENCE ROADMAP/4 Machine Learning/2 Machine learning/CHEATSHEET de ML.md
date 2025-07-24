
# 🧠 Machine Learning & Data Science Cheatsheet

Guía exhaustiva y pedagógica para aprender desde cero a nivel intermedio-avanzado en ML y DS.

---

# 📦 Parte 1: Exploración, Manipulación y Análisis de Datos con Pandas

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

## Manipulación Numérica y Álgebra Lineal con NumPy

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
# ➕Parte 2 : Metodos de Transformación
## 🔢1. Transformaciones Numéricas y  de Variables categóricas

### 🧩 Transformaciones Numéricas

#### 1. 🎯 Centering (Centrado)
$$
x_{centered} = x - \bar{x}
$$

```python
x_centered = x - np.mean(x)
```

---

#### 2. ⚖️ Standard Scaler (Estandarización)
$$
x_{std} = \frac{x - \mu}{\sigma}
$$

```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x.reshape(-1, 1))
```

---

#### 3. 📊 Min-Max Scaler
$$
x' = \frac{x - x_{min}}{x_{max} - x_{min}}
$$

```python
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
x_minmax = scaler.fit_transform(x.reshape(-1, 1))
```

---

#### 4. 🧱 Binning (Discretización)

```python
x_binned = pd.cut(x, bins=3, labels=["bajo", "medio", "alto"])
```

---

#### 5. 🔁 Transformaciones Logarítmicas
$$
x' = \log(x + 1)
$$

```python
x_log = np.log1p(x)
```

---

### 🧩 Transformaciones de Variables Categóricas

#### 🔢 Ordinal Encoding

```python
from sklearn.preprocessing import OrdinalEncoder
encoder = OrdinalEncoder(categories=[['Excellent', 'New', 'Like New', 'Good', 'Fair']])
encoded = encoder.fit_transform(cars['condition'].values.reshape(-1,1))
```

---

#### 🏷️ Label Encoding

```python
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
cars['color'] = encoder.fit_transform(cars['color'])
```

---

#### 🟦 One-Hot Encoding

```python
ohe = pd.get_dummies(cars['color'])
cars = cars.join(ohe)
```

---

#### 🔢 Binary Encoding

```python
from category_encoders import BinaryEncoder
colors = BinaryEncoder(cols=['color']).fit_transform(cars)
```

---

#### 💠 Hashing Encoding

```python
from category_encoders import HashingEncoder
encoder = HashingEncoder(cols='color', n_components=5)
hash_results = encoder.fit_transform(cars['color'])
```

---

#### 🎯 Target Encoding

```python
from category_encoders import TargetEncoder
encoder = TargetEncoder(cols='color')
encoded = encoder.fit_transform(cars['color'], cars['sellingprice'])
```

---

#### ⏰ Encoding de Fechas

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

## 🧬 2 : Reducción de Características

La selección de características permite eliminar variables irrelevantes o redundantes, mejorando la eficiencia y precisión del modelo.

---

### 🧠 Categorías Principales

| Método       | Basado en...             | ¿Usa modelo? | Características clave                          |
|--------------|--------------------------|--------------|------------------------------------------------|
| Filter       | Estadísticas individuales| ❌ No         | Rápido, independiente del modelo               |
| Wrapper      | Rendimiento del modelo   | ✅ Sí         | Evalúa subconjuntos, computacionalmente costoso|
| Embedded     | Aprendizaje interno del modelo | ✅ Sí   | Usa regularización o importancia automática    |

---

### 🔎 A. Filter Methods

Evalúan cada variable por separado, usando estadísticas para seleccionar las más relevantes.

#### 📌 Técnicas comunes:
- **Chi-cuadrado (χ²)** – para variables categóricas
- **ANOVA F-test** – para clasificación
- **Correlación de Pearson** – para regresión y datos numéricos

#### 📦 Ejemplo en Python:

```python
from sklearn.feature_selection import SelectKBest, chi2, f_classif

# Chi-cuadrado para clasificación
selector = SelectKBest(score_func=chi2, k=5)
X_new = selector.fit_transform(X, y)

# ANOVA F-test
selector = SelectKBest(score_func=f_classif, k=5)
X_new = selector.fit_transform(X, y)
```

---

### 🔁 B. Wrapper Methods

Evalúan **subconjuntos de características** entrenando modelos múltiples para encontrar combinaciones óptimas.

#### 📌 Métodos típicos:

| Método    | Descripción                                                    |
|-----------|----------------------------------------------------------------|
| **SFS**   | Agrega features uno a uno que más mejoran el modelo            |
| **BFS**   | Elimina features uno a uno hasta que el modelo empeore         |
| **SFFS**  | Forward con posibilidad de eliminar en pasos posteriores       |
| **SBFS**  | Backward con posibilidad de recuperar features eliminadas      |
| **RFE**   | Elimina recursivamente las menos importantes con un modelo     |

#### 📦 Ejemplo con RFE:

```python
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
selector = RFE(model, n_features_to_select=5)
X_selected = selector.fit_transform(X, y)
```

#### 📦 Ejemplo con SFS (`mlxtend`):

```python
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.linear_model import LogisticRegression

sfs = SFS(LogisticRegression(),
          k_features=5,
          forward=True,  # False para backward
          floating=False,  # True para SFFS/SBFS
          scoring='accuracy',
          cv=5)

sfs = sfs.fit(X, y)
print("Mejores features:", sfs.k_feature_names_)
```

---

### 🧲 C. Embedded Methods (con Regularización)

Incorporan la selección de características **dentro del proceso de entrenamiento** del modelo. Usan penalizaciones para reducir el impacto de variables menos útiles.

#### 🧠 ¿Qué es Regularización?

La regularización agrega una penalización al modelo para evitar sobreajuste y reducir la complejidad. Esto puede "forzar" a ciertos coeficientes a valores muy bajos o incluso cero.

#### 🔶 Lasso Regression (L1)

- Penaliza con la **suma de los valores absolutos** de los coeficientes.
- Puede forzar coeficientes a cero → selección automática de características.

$$
\text{Loss}_{L1} = RSS + \alpha \sum |w_i|
$$

```python
from sklearn.linear_model import LassoCV

model = LassoCV()
model.fit(X, y)
print(model.coef_)  # Coeficientes importantes se mantienen
```

---

#### 🔷 Ridge Regression (L2)

- Penaliza con la **suma de los cuadrados** de los coeficientes.
- Reduce la magnitud de los coeficientes, pero no los lleva a cero.
- Útil cuando hay **colinealidad** entre variables.

$$
\text{Loss}_{L2} = RSS + \alpha \sum w_i^2
$$

```python
from sklearn.linear_model import RidgeCV

ridge = RidgeCV(alphas=[0.1, 1.0, 10.0])
ridge.fit(X, y)
print(ridge.coef_)
```

---

#### 🔀 ElasticNet

- Combinación de **L1 (Lasso)** y **L2 (Ridge)**.
- Controla el equilibrio entre selección de variables y regularización suave.

$$
\text{Loss}_{ElasticNet} = RSS + \alpha (r \sum |w_i| + (1 - r) \sum w_i^2)
$$

```python
from sklearn.linear_model import ElasticNetCV

enet = ElasticNetCV()
enet.fit(X, y)
print(enet.coef_)
```

---

#### 🌲 Importancia basada en árboles

Modelos como Random Forest o XGBoost calculan la importancia de cada variable de forma automática.

```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()
model.fit(X, y)

importances = model.feature_importances_
```

---

## 🌈 3: Dimensionality Reduction: PCA y t-SNE

### 🧠 ¿Por qué reducir dimensiones?

- Visualización más clara de los datos.  
- Eliminar ruido o colinealidad.  
- Acelerar entrenamiento de modelos.  
- Mejorar generalización y evitar overfitting.  

---

### 🔍 ¿Por que son no supervisados?

Las técnicas de reducción de dimensionalidad **no supervisadas** (como PCA y t-SNE) **no** utilizan las etiquetas $(y$) de los datos al buscar sus nuevas representaciones. En lugar de aprender una función $(f: X \to y$), estas técnicas:

1. Analizan únicamente la **estructura interna** de las características $(X$).  
2. Encuentran combinaciones o proyecciones (componentes principales en PCA, distribución probabilística local en t-SNE) que **maximizan la retención de información** o **conservan distancias** entre puntos.  
3. Operan sin guía de salidas deseadas, extrayendo patrones de forma **exploratoria** y **descriptiva**.

---

### 📉 PCA (Principal Component Analysis)

**PCA** busca proyectar los datos en un nuevo espacio de menor dimensión **maximizando la varianza**.
#### ⚙️ Fundamento Matemático

1. Centrar los datos (media = 0).
2. Calcular la **matriz de covarianza**.
3. Calcular los **eigenvectors** y **eigenvalores**.
4. Elegir los componentes con mayor varianza.
5. Proyectar los datos en el nuevo subespacio.

$$
X_{pca} = X \cdot W_k
$$

- $X$: matriz original
- $W_k$: los $k$ autovectores principales

---

#### 📌 ¿Cuándo usar PCA?

- Datos **numéricos**.
- Relación **lineal** entre variables.
- Objetivo: **reducción**, **visualización**, o **descorrelación**.

---

#### 💻 Código PCA en Python

```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Datos simulados
X = df.drop('target', axis=1)
y = df['target']

# Escalado previo
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Visualización
plt.figure(figsize=(6, 5))
sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=y, palette='Set2')
plt.title("PCA - 2 Componentes")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()
```

---

#### 🧮 ¿Cuántos componentes elegir?

```python
pca = PCA().fit(X_scaled)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel("Número de componentes")
plt.ylabel("Varianza explicada acumulada")
plt.title("Varianza explicada por PCA")
plt.grid()
plt.show()
```

---

### 🎨 t-SNE (t-distributed Stochastic Neighbor Embedding)

**t-SNE** es una técnica **no lineal** que preserva relaciones locales entre puntos para **visualización** en 2D o 3D.

- Basado en **distribuciones de probabilidad**.
- Ideal para clusters no lineales.

---

#### 💻 Código t-SNE en Python

```python
from sklearn.manifold import TSNE

# t-SNE requiere datos escalados
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)

# Visualización
plt.figure(figsize=(6, 5))
sns.scatterplot(x=X_tsne[:,0], y=X_tsne[:,1], hue=y, palette='coolwarm')
plt.title("t-SNE - 2 Componentes")
plt.xlabel("Dim 1")
plt.ylabel("Dim 2")
plt.show()
```

---

### ✅ Conclusiones

#### 🧠 Diferencias vs PCA

| Característica       | PCA               | t-SNE                  |
|----------------------|-------------------|------------------------|
| Tipo                 | Lineal            | No lineal              |
| Uso principal        | Reducción general | Visualización          |
| Velocidad            | Rápido            | Lento (grandes datos)  |
| Interpretabilidad    | Alta              | Baja                   |
| Conserva distancias  | Global            | Local                  |

- Usa **PCA** si necesitas interpretación, velocidad, o reducción antes de modelos.
- Usa **t-SNE** si deseas una **visualización intuitiva** de la estructura interna de los datos.

> 🌟 A veces es útil combinar ambos: aplicar PCA para reducir a 50 dimensiones y luego t-SNE a 2D.


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

> 🧠 **Consejo:** Usa `Pipeline` junto con `GridSearchCV` para combinar normalización, selección de variables y modelo en un solo flujo reproducible.


---
# 🤖 Parte 5: Modelado Supervisado
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

---

## 🎯Ensemble methods


En Machine Learning, **los métodos de ensamble** combinan múltiples modelos (a menudo llamados *estimadores base*) para crear un modelo más robusto y preciso que cualquiera de sus partes por separado.

> 📌 *"Un conjunto de modelos débiles puede formar un modelo fuerte."*

---

### 🎯 ¿Por qué usar Ensembles?

- 🔁 Reducen **varianza** (overfitting).
- 🎯 Disminuyen **sesgo** (underfitting).
- 📈 Mejoran la **precisión y estabilidad** del modelo.
- 🛡️ Son más **resistentes al ruido** y a errores de muestreo.

---

### 🔗 Tipos Principales

| Tipo         | ¿Cómo funciona?                                             | Ejemplo típico                |
|--------------|-------------------------------------------------------------|-------------------------------|
| **Bagging**  | Entrena varios modelos **en paralelo** con muestras distintas (bootstrapped). | Random Forest                 |
| **Boosting** | Entrena modelos **en secuencia**, cada uno corrige los errores del anterior. | AdaBoost, Gradient Boosting  |
| **Stacking** | Combina modelos diferentes y usa otro modelo (meta-modelo) para hacer la predicción final. | Modelos de mezcla (blending) |

---

### 🔍 Comparación General

| Método     | Reduce varianza | Reduce sesgo | Paralelizable | Ejemplo                  |
|------------|------------------|---------------|----------------|---------------------------|
| Bagging    | ✅                | ❌             | ✅              | Random Forest             |
| Boosting   | ❌                | ✅             | ❌              | XGBoost, AdaBoost         |
| Stacking   | ✅✅              | ✅✅           | ❌              | Meta-ensemble personalizado |

---

### A) Bagging Methods

#### 🌲Random Forest
El **Random Forest** es un algoritmo de aprendizaje supervisado basado en árboles de decisión. Es parte de la familia de métodos de ensamble, específicamente del tipo **bagging**.

---

##### 🎯 ¿Qué es?

Un **Random Forest** entrena múltiples árboles de decisión sobre diferentes subconjuntos del conjunto de datos (mediante muestreo con reemplazo) y luego combina sus predicciones:

- En clasificación: toma la **votación mayoritaria**
- En regresión: toma el **promedio** de las predicciones

---

##### 🔍 ¿Por qué usar Random Forest?

- Reduce el **overfitting** de los árboles individuales
- Proporciona una buena estimación de la importancia de las características
- Robusto ante datos ruidosos y valores atípicos

---

##### 🔧 Principales Hiperparámetros

| Parámetro              | Descripción                                               |
|------------------------|-----------------------------------------------------------|
| `n_estimators`         | Número de árboles en el bosque                            |
| `max_depth`            | Profundidad máxima de cada árbol                          |
| `max_features`         | Número de features consideradas en cada split             |
| `min_samples_split`    | Mínimo de muestras requeridas para dividir un nodo        |
| `min_samples_leaf`     | Mínimo de muestras en una hoja                            |
| `bootstrap`            | Si se usan muestras con reemplazo (`True` por defecto)    |

---

##### 💻 Ejemplo en Python

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Datos de ejemplo
X, y = load_iris(return_X_y=True)

# División de datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Modelo
rf = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42)
rf.fit(X_train, y_train)

# Predicción y evaluación
y_pred = rf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
```

---

##### 🌿 Importancia de Características

Random Forest permite evaluar la **importancia relativa** de cada variable en las decisiones del modelo.

```python
import pandas as pd
import matplotlib.pyplot as plt

feature_importance = rf.feature_importances_
features = load_iris().feature_names
importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importance})
importance_df.sort_values(by="Importance", ascending=True).plot.barh(x='Feature', y='Importance')
plt.title("Importancia de Características")
plt.show()
```

---

#### ✅ Ventajas

- Funciona bien con datos no lineales y con muchas variables
- Poca necesidad de preprocesamiento
- Puede manejar datos faltantes (algunas implementaciones)

#### ⚠️ Desventajas

- Difícil de interpretar individualmente (modelo tipo "caja negra")
- Lento para predicción en tiempo real con muchos árboles
- Puede consumir mucha memoria

---

#### 🧠 Cuándo usarlo

- Problemas de clasificación o regresión donde se necesita robustez
- Cuando otros modelos individuales tienen alto **overfitting**
- Como baseline potente antes de probar modelos más complejos

---

#### 📌 Notas adicionales

- Si hay **desequilibrio de clases**, puedes usar `class_weight='balanced'`
- Para regresión, utiliza `RandomForestRegressor` con la misma lógica

---

### B) 🚀 Boosting Methods 

Boosting es una técnica de ensamble que **combina varios modelos débiles** (como árboles pequeños) para formar un modelo fuerte. A diferencia de bagging (como Random Forest), el entrenamiento es **secuencial**: cada modelo corrige los errores del anterior.

---

#### 🧠 Idea Principal

1. Entrena un modelo débil (e.g., árbol).
2. Evalúa los errores.
3. Entrena otro modelo **enfocado en los errores**.
4. Repite, combinando todos los modelos con **pesos**.

> Se usa mucho en tareas donde se necesita **alta precisión** (competencias, bancos, medicina).

---

#### 📦 Tipos de Boosting

##### 1. 🎯 AdaBoost (Adaptive Boosting)

- Asigna más peso a los errores.
- Usa modelos débiles (usualmente árboles de decisión con profundidad 1).
- Actualiza los pesos en cada iteración.

```python
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

ada = AdaBoostClassifier(
    base_estimator=DecisionTreeClassifier(max_depth=1),
    n_estimators=50,
    learning_rate=1.0
)
ada.fit(X_train, y_train)
```

---

##### 2. 🌳 Gradient Boosting

- Optimiza una función de pérdida usando gradientes.
- Más preciso que AdaBoost, pero más lento.
- Puede sobreajustar si no se regula bien.

```python
from sklearn.ensemble import GradientBoostingClassifier

gb = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3
)
gb.fit(X_train, y_train)
```

---

##### 3. ⚡ XGBoost (Extreme Gradient Boosting)

- Optimización de Gradient Boosting.
- Muy usado en Kaggle y producción.
- Rápido, regularizado, eficiente.

```python
from xgboost import XGBClassifier

xgb = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    use_label_encoder=False,
    eval_metric='logloss'
)
xgb.fit(X_train, y_train)
```

---

##### 4. 🧠 LightGBM

- Usa histogramas y hojas en lugar de niveles.
- Más rápido en datasets grandes.
- Requiere datos limpios y sin categorías codificadas mal.

```python
from lightgbm import LGBMClassifier

lgbm = LGBMClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3
)
lgbm.fit(X_train, y_train)
```

---

#### ✅ Ventajas

- Alta precisión.
- Reduce sesgo.
- Útil en datos tabulares.

#### ⚠️ Desventajas

- Lento si no se optimiza.
- Más propenso a sobreajuste.
- Difícil de interpretar (especialmente XGBoost).

---

#### 🧪 Consejos Prácticos

- Ajusta el número de estimadores y `learning_rate` juntos.
- Usa `early_stopping_rounds` en XGBoost o LightGBM para evitar overfitting.
- Visualiza la importancia de variables (`feature_importances_`).

---

#### 📊 Visualización de Importancia

```python
import matplotlib.pyplot as plt

importances = xgb.feature_importances_
plt.bar(range(len(importances)), importances)
plt.title("Importancia de Características (XGBoost)")
plt.show()
```

> ✅ Boosting es ideal cuando se busca precisión máxima.  
> 🔍 ¡Pero ojo con el tiempo de cómputo y el sobreajuste!

---

### D) 🧱 Stacking Methods


**Stacking (Stacked Generalization)** es una técnica de ensamblado donde múltiples modelos (denominados *base learners*) son entrenados y sus predicciones son usadas como entrada de un **modelo meta** (*meta learner*), que aprende a combinarlas de forma óptima.

---

#### 🧠 ¿Por qué usar Stacking?

- Aprovecha la **diversidad de modelos** para mejorar el rendimiento.
- El *meta-modelo* aprende de los errores y aciertos de cada *base learner*.
- Puede mejorar el rendimiento en comparación con métodos individuales o bagging/boosting.

---

#### ⚙️ Arquitectura de Stacking

```
X_train ─┬──────────> Model 1 ─┐
         ├──────────> Model 2 ─┤
         ├──────────> Model 3 ─┤──> Meta-model (e.g. Logistic Regression)
         └──────────> ...     ─┘
```

- Los modelos base se entrenan con el conjunto de entrenamiento.
- El modelo meta se entrena con las predicciones de los modelos base.
- Puede usarse validación cruzada para evitar sobreajuste.

---

#### 🔢 ¿Qué modelos usar?

- **Base learners**: modelos diversos (árboles, regresiones, KNN, etc.).
- **Meta learner**: modelo simple como regresión logística o ridge regression.

---

#### 💻 Ejemplo en Python con Scikit-learn

```python
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

# Dataset
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Modelos base
estimators = [
    ('dt', DecisionTreeClassifier(max_depth=3)),
    ('knn', KNeighborsClassifier(n_neighbors=5)),
    ('svm', SVC(probability=True))
]

# Modelo de ensamblado
stack = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression()
)

# Entrenar
stack.fit(X_train, y_train)
print("Accuracy:", stack.score(X_test, y_test))
```

---

#### 🧠 Recomendaciones

- Usa modelos base que generen predicciones distintas.
- Asegura que el *meta-modelo* no tenga acceso a los datos originales, solo a las predicciones.
- Usa validación cruzada interna para generar las predicciones de entrenamiento del *meta-modelo*.

---

#### ✅ Ventajas

| Beneficio                 | Descripción |
|---------------------------|-------------|
| Alta performance          | Puede superar a modelos individuales |
| Flexibilidad              | Puedes mezclar cualquier tipo de modelo |
| Aprovecha especialización| Cada modelo puede enfocarse en un tipo de patrón |

#### ⚠️ Limitaciones

| Desventaja               | Descripción |
|--------------------------|-------------|
| Complejidad computacional| Entrena múltiples modelos |
| Riesgo de overfitting    | Si no se aplica correctamente |
| Dificultad de interpretación | Difícil explicar decisiones |

---

#### 🧪 Variaciones

- **Blending**: variante más simple de stacking que usa un conjunto holdout en vez de validación cruzada.
- **Multilayer Stacking**: anidación de varios niveles de modelos base y meta.

---

## 🧠 Naive Bayes Classifier

`Naive Bayes` es una familia de algoritmos de clasificación basada en el **Teorema de Bayes**. Es simple pero muy eficaz para tareas como clasificación de texto (spam, análisis de sentimientos, etc.).

---

## 📐 Teorema de Bayes

El **Teorema de Bayes** permite calcular la probabilidad de un evento A dado un evento B:

$$
P(A \mid B) = \frac{P(B \mid A) \cdot P(A)}{P(B)}
$$

En palabras sencillas:

> ¿Cuál es la probabilidad de que algo sea **A** si observamos **B**?

---

## 🧠 Naive Bayes: ¿por qué "naive"?

Se le llama *naive* (ingenuo) porque **asume que las variables predictoras son independientes entre sí**, lo cual rara vez es cierto, pero en la práctica funciona muy bien.

### Aplicación clásica: clasificación de texto

Usamos `Naive Bayes` para predecir si un mensaje es **spam o no spam**, basándonos en las palabras que contiene.

---

## 🔤 CountVectorizer: convertir texto a números

Antes de entrenar el modelo, debemos convertir el texto en una **matriz numérica**. `CountVectorizer` convierte las palabras en conteos:

```python
from sklearn.feature_extraction.text import CountVectorizer

corpus = [
    "me encanta el machine learning",
    "el aprendizaje automático es fascinante",
    "odio el spam",
    "el spam es molesto"
]

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus)

print(vectorizer.get_feature_names_out())
print(X.toarray())
```

Cada fila representa un documento (frase) y cada columna una palabra.

---

## 🤖 Entrenar un modelo con MultinomialNB

El modelo `MultinomialNB` es una variante de Naive Bayes diseñada para datos **discretos** como el conteo de palabras.

```python
from sklearn.naive_bayes import MultinomialNB

# Etiquetas: 0 = ham, 1 = spam
y = [0, 0, 1, 1]

model = MultinomialNB()
model.fit(X, y)
```

---

## 🔍 Hacer predicciones

```python
test = ["me molesta el spam"]
X_test = vectorizer.transform(test)

pred = model.predict(X_test)
print(pred)  # Resultado: [1] => spam
```

---

## 📈 ¿Por qué usar Naive Bayes?

✅ Muy rápido y eficiente  
✅ Funciona bien con datos de texto  
✅ Ideal para datos grandes y dispersos  
⚠️ No captura relaciones complejas entre variables

---

## 🧪 Métricas para evaluación

Usa las mismas métricas que en clasificación:

- Accuracy
- Precision
- Recall
- F1-score

```python
from sklearn.metrics import classification_report

print(classification_report(y, model.predict(X)))
```

---

## 📌 Recomendaciones

- Funciona mejor cuando las características son independientes.
- Si los datos tienen muchas ceros (como en texto), `MultinomialNB` o `BernoulliNB` son buenas opciones.
- Para texto, combinar `CountVectorizer` o `TfidfVectorizer` con `Naive Bayes` suele dar buenos resultados.


---
# 🤖 Parte 6: Modelado No-Supervisado

## 🧠 K-Means Clustering

K-Means es un algoritmo de **aprendizaje no supervisado** utilizado para agrupar datos similares. Busca dividir los datos en **K grupos (clusters)**, donde cada punto pertenece al cluster con el centro más cercano (media).

---

### 🚀 ¿Cómo Funciona?

1. Se eligen **K centroides** aleatoriamente.
2. Se asigna cada punto al centroide más cercano.
3. Se recalculan los centroides como la media de los puntos asignados.
4. Se repite el proceso hasta que los centroides no cambien (o cambien muy poco).

---

### 📐 Fórmula de Distancia Euclidiana

$$
\text{dist}(x, \mu) = \sqrt{\sum_{i=1}^{n} (x_i - \mu_i)^2}
$$

Donde:
- $( x )$ es el punto de datos
- $( \mu )$ es el centroide

---

### 📦 Implementación en Python

```python
from sklearn.cluster import KMeans
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Datos de ejemplo
df = pd.DataFrame({
    'x': [1, 1.5, 3, 5, 3.5, 4.5, 3.5],
    'y': [1, 2, 4, 7, 5, 5, 4.5]
})

# K-Means con 2 clusters
kmeans = KMeans(n_clusters=2, random_state=42)
df['cluster'] = kmeans.fit_predict(df[['x', 'y']])

# Visualizar resultados
sns.scatterplot(data=df, x='x', y='y', hue='cluster', palette='viridis')
plt.scatter(*kmeans.cluster_centers_.T, s=200, c='red', label='Centroides')
plt.legend()
plt.title("K-Means Clustering")
plt.show()
```

---

### ❓ Cómo Elegir K (Número de Clusters)

#### 📉 Método del Codo (Elbow Method)

```python
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

inertia = []
K = range(1, 10)

for k in K:
    model = KMeans(n_clusters=k, random_state=42)
    model.fit(df[['x', 'y']])
    inertia.append(model.inertia_)

plt.plot(K, inertia, 'o-')
plt.xlabel('Número de Clusters K')
plt.ylabel('Inercia (Suma de errores)')
plt.title('Método del Codo')
plt.show()
```

---

### 🧪 Métricas de Evaluación

| Métrica            | Descripción |
|--------------------|-------------|
| **Inercia**         | Suma de distancias cuadradas entre puntos y su centroide |
| **Silhouette Score** | Mide la separación entre clusters (0 a 1, mejor si es cercano a 1) |

```python
score = silhouette_score(df[['x', 'y']], df['cluster'])
print("Silhouette Score:", score)
```

---

#### 🧠 Ventajas

- Simple y eficiente
- Funciona bien con clusters esféricos

### ⚠️ Desventajas

- Sensible a outliers
- No funciona bien con clusters de forma no circular
- Hay que definir K previamente

---

#### 🎯 Cuándo usar K-Means

- Cuando tienes datos sin etiquetas y buscas estructura interna.
- Cuando quieres una solución rápida de agrupamiento.

> 💡 Tip: Escalar los datos con `StandardScaler` antes de aplicar K-Means puede mejorar los resultados.


---

# 🎯 Parte 8: Recommender Systems

Los *Recommender Systems* (sistemas de recomendación) son algoritmos que predicen qué elementos pueden gustarle a un usuario, basándose en su historial o en las preferencias de otros usuarios. Son ampliamente usados en plataformas como **Netflix**, **Amazon**, **Spotify**, etc.

---

## 🧩 Tipos principales de Recommender Systems

| Tipo               | ¿Cómo funciona?                                                                 | Ejemplo                         |
|--------------------|----------------------------------------------------------------------------------|----------------------------------|
| Content-Based      | Analiza las características de los ítems que te han gustado y busca similares. | "Te gustó esta película de acción, prueba esta otra del mismo género." |
| Collaborative Filtering | Busca patrones entre usuarios con gustos similares.                          | "Personas como tú también vieron..." |
| Hybrid             | Combina los dos métodos anteriores.                                             | "Recomendación personalizada combinando gustos e historial de otros." |

---

## 🔧 Usando Surprise para Collaborative Filtering

La librería `surprise` es una herramienta sencilla para crear y evaluar modelos de filtrado colaborativo.

### Paso 1: Importar y cargar datos

```python
from surprise import Dataset, Reader, SVD
from surprise.model_selection import cross_validate

# Cargar dataset de ejemplo
data = Dataset.load_builtin('ml-100k')  # Ratings de películas
```

### Paso 2: Crear y evaluar el modelo

```python
algo = SVD()  # Singular Value Decomposition (modelo basado en factores)

# Evaluar usando validación cruzada
cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
```

---

## 🔍 Hacer predicciones

Una vez entrenado, puedes predecir qué calificación daría un usuario a un ítem específico:

```python
trainset = data.build_full_trainset()
algo.fit(trainset)

# Predecir calificación del usuario 196 para el ítem 302
pred = algo.predict(uid='196', iid='302')
print(pred.est)  # .est contiene la calificación estimada
```

---

## 📌 Recomendaciones finales

- Para sistemas más robustos puedes combinar modelos (*hybrid systems*).
- Existen otros enfoques como *deep learning*, *item-based filtering* o *context-aware*.
- Siempre valida tus modelos con datos reales y mide su impacto.

___
# 🧠 Parte 8: Optimización de Hiperparámetros (Hyperparameter Tuning)

Elegir correctamente los hiperparámetros puede marcar la diferencia entre un modelo mediocre y uno excelente. Aquí se presentan los métodos más utilizados para hacer tuning de manera eficaz.

---

## ⚙️ ¿Qué es un hiperparámetro?

Parámetros definidos antes del entrenamiento y no aprendidos directamente del modelo.

Ejemplos:
- `n_neighbors` en KNN
- `C`, `kernel` en SVM
- `alpha` en Lasso/Ridge
- `max_depth`, `min_samples_split` en árboles y Random Forest
- `learning_rate`, `n_estimators` en Boosting

---

## 🔍 Métodos de Optimización

### 🔄 1. Grid Search

Explora **todas las combinaciones posibles** dentro de un conjunto de valores definido.

```python
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

param_grid = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
grid = GridSearchCV(SVC(), param_grid, cv=5)
grid.fit(X_train, y_train)
```

✅ Exhaustivo  
⚠️ Costoso con muchos hiperparámetros

---

### 🎲 2. Random Search

Explora **combinaciones aleatorias** dentro de un espacio definido. Más eficiente en grandes espacios.

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
from sklearn.ensemble import RandomForestClassifier

param_dist = {'n_estimators': randint(50, 200)}
rand = RandomizedSearchCV(RandomForestClassifier(), param_dist, n_iter=10, cv=5)
rand.fit(X_train, y_train)
```

✅ Más rápido que Grid  
⚠️ Puede omitir combinaciones óptimas

---

### 📈 3. Bayesian Optimization (con `optuna`)

Modelo probabilístico que **aprende del pasado** para proponer mejores combinaciones.

```python
import optuna
from sklearn.model_selection import cross_val_score

def objective(trial):
    C = trial.suggest_loguniform("C", 1e-3, 10)
    kernel = trial.suggest_categorical("kernel", ["linear", "rbf"])
    model = SVC(C=C, kernel=kernel)
    return cross_val_score(model, X, y, cv=5).mean()

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)

print("Best params:", study.best_params)
```

✅ Inteligente, eficiente  
⚠️ Más complejo de implementar

---

### 🧬 4. Algoritmos Genéticos (con `TPOT`)

Simulan evolución biológica para encontrar hiperparámetros óptimos (y a veces estructuras de modelo).

```python
from tpot import TPOTClassifier

tpot = TPOTClassifier(generations=5, population_size=50, verbosity=2)
tpot.fit(X_train, y_train)
```

✅ Explora soluciones creativas  
⚠️ Lento, uso de CPU intensivo

---

### 🔧 Otros métodos posibles

| Método              | Descripción breve                            |
|---------------------|-----------------------------------------------|
| Hyperband           | Variante de random search con early stopping |
| Successive Halving  | Evalúa muchas configuraciones y elimina pronto las peores |

---

## 📊 Evaluación de Resultados

Después de usar `GridSearchCV` o `RandomizedSearchCV`:

```python
model = grid  # o rand, etc.

print(model.best_estimator_)  # Mejor modelo completo
print(model.best_params_)     # Mejores hiperparámetros
print(model.best_score_)      # Mejor puntuación de validación cruzada
print(model.cv_results_)      # Resultados completos de todas las combinaciones
```

✅ Usa `.cv_results_` para análisis personalizados o gráficas de rendimiento.

---

## ✅ Resumen Comparativo

| Método          | Velocidad | Eficiencia | Complejidad | Escenarios ideales                     |
|-----------------|-----------|------------|-------------|----------------------------------------|
| Grid Search     | Lenta     | Alta       | Baja        | Pocos parámetros, espacio pequeño      |
| Random Search   | Rápida    | Media      | Baja        | Muchos parámetros, tuning general      |
| Bayesian Opt.   | Media     | Muy alta   | Media/Alta  | Optimización precisa y eficiente       |
| Genético (TPOT) | Lenta     | Alta       | Alta        | AutoML, problemas grandes o complejos  |
