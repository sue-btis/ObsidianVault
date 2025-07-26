---
tags: [python, programming, deep_learning]
aliases: ["Redes neuronales profundas", "Aprendizaje profundo"]
creation_date: "2025-07-24"
---

# 🎓 Deep Learning

## 🎯 Objective
*¿Qué es el Deep Learning y cómo funciona a nivel básico?*

Aprenderás qué es el Deep Learning, cómo se relaciona con el Machine Learning, y entenderás los conceptos clave como `tensors`, `loss function`, `gradient descent` y `backpropagation` con ejemplos y metáforas claras.

---

## 🧠 The Core Idea (The Analogy)
*Imagina que tu cerebro es una gran ciudad llena de luces.*

Cada luz representa una pequeña decisión: encender o apagar, conectar o no conectar. El Deep Learning es como construir una ciudad artificial con muchas luces llamadas `neurons`. Estas luces se conectan en capas como pisos de un edificio y aprenden a iluminarse correctamente observando miles de ejemplos, como si estuvieran entrenando con experiencias pasadas.

---

## ⚙️ How It Works: Step-by-Step

1. **Input Layer (Capa de entrada):** Los datos (por ejemplo, una imagen) se convierten en números y entran como un conjunto de `tensors` (matrices multidimensionales).
2. **Hidden Layers (Capas ocultas):** Cada neurona hace una operación matemática (como multiplicar y sumar) y pasa la información a la siguiente capa.
3. **Activation Function:** Se usa una función para decidir si la neurona "se activa" o no. Ejemplo común: `ReLU`.
4. **Loss Function:** Calculamos el error de la red, es decir, cuánto se equivocó la predicción.
5. **Backpropagation:** Como si retrocedieras para corregir errores, este paso ajusta las conexiones neuronales.
6. **Gradient Descent:** Una técnica para encontrar la mejor forma de corregir el error, como bajar por una montaña buscando el punto más bajo.

---

### 🧮 Math Behind It: Tensors & Functions

- **Tensor:** Es como una caja mágica que guarda números. Un tensor puede ser:
  - Escalar (número): $x = 7$
  - Vector (lista): $v = [1, 2, 3]$
  - Matriz: 
    $$
    M = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}
    $$
  - Tensor 3D o más: Piensa en una caja con muchas hojas de papel (cada hoja es una matriz).

- **Loss Function:** Mide qué tan mal predijo la red. Por ejemplo, la `Mean Squared Error (MSE)` es:
  
  $$
  L = \frac{1}{n} \sum_{i=1}^{n}(y_i - \hat{y}_i)^2
  $$

  Donde $y_i$ es el valor real y $\hat{y}_i$ es el valor predicho.

- **Gradient Descent:**

  $$ 
  \theta = \theta - \alpha \cdot \nabla_\theta L(\theta)
  $$

  Significa: ajusta tus parámetros $\theta$ en la dirección contraria del error para mejorar.

---

### 🔁 Backpropagation
Este proceso recorre la red hacia atrás, calculando cuánto influyó cada neurona en el error. Luego, ajusta los "pesos" (las conexiones) usando `gradient descent`.

---

## 🎲 Stochastic & Variants of Gradient Descent

- **Batch Gradient Descent:** Usa todos los datos antes de ajustar pesos. Muy lento.
- **Stochastic Gradient Descent (SGD):** Ajusta después de cada ejemplo. Más rápido, pero ruidoso.
- **Mini-Batch Gradient Descent:** Mezcla lo mejor de ambos: usa pequeños grupos para mejorar velocidad y estabilidad.

---

## 🤖 Difference Between Machine Learning & Deep Learning

| Concepto              | Machine Learning (ML)                   | Deep Learning (DL)                                  |
|-----------------------|-----------------------------------------|-----------------------------------------------------|
| Requiere Feature Engineering | ✅ Sí, manualmente                     | ❌ Aprende directamente de los datos                 |
| Datos necesarios       | Pocos a medianos                       | Muchos datos (Big Data)                            |
| Interpretabilidad      | Alta (más fácil de entender)           | Baja (es una "caja negra")                          |
| Ejemplos de modelos    | Decision Tree, SVM, KNN                | Convolutional Neural Network, RNN, Transformer      |

---

## ⚠️ Precautions with Deep Learning

- ❗ **Requiere muchos datos:** Si no tienes suficientes, puede aprender mal.
- ❗ **Puede sobreajustarse (overfitting):** Aprende de memoria en lugar de generalizar.
- ❗ **Difícil de interpretar:** No siempre sabrás “por qué” tomó una decisión.
- ❗ **Uso ético:** Puede replicar sesgos si se entrena con datos no balanceados.

---

### 💻 Code Example
*Ejemplo básico de una red neuronal usando `TensorFlow`.*

```python
# Red neuronal simple con TensorFlow y Keras
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Creamos una red con 1 capa oculta
model = Sequential([
    Dense(16, activation='relu', input_shape=(10,)),  # Capa oculta con 16 neuronas
    Dense(1, activation='sigmoid')  # Capa de salida para clasificación binaria
])

# Compilamos con función de pérdida y optimizador
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Simulamos entrenamiento con datos ficticios
import numpy as np
X = np.random.rand(100, 10)  # 100 muestras, 10 características
y = np.random.randint(0, 2, size=(100,))  # 0 o 1

# Entrenamos la red
model.fit(X, y, epochs=5, batch_size=10)
