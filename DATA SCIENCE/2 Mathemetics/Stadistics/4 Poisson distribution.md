#Math #Stadistics 
# 🌟 Distribución de Poisson

La **Distribución de Poisson** describe la probabilidad de que ocurran exactamente \( k \) eventos en un intervalo fijo (tiempo, espacio, etc.), cuando:


- Los eventos son **independientes** entre sí.

- Ocurren a una **tasa promedio constante** \($$\lambda$$

- No pueden ocurrir múltiples eventos al mismo instante exacto.
## 📌 Propiedades

- **Esperanza (media):**
  $$\mathbb{E}(X) = \lambda$$

- **Varianza:**  
  $$\mathrm{Var}(X) = \lambda$$
## 🧪 Ejemplo en Python

  

```python
from scipy.stats import poisson

lam = 3  # tasa promedio de eventos

expectation = poisson.mean(lam)

variance = poisson.var(lam)

  

print("Esperanza:", expectation)

print("Varianza:", variance)
```

  

> 🧠 Si en promedio ocurren 3 eventos por intervalo, entonces:

> - Se espera observar 3 eventos.

> - La dispersión de los datos también es 3.


---

# 🪙 Distribución Binomial

La **Distribución Binomial** modela el número de **éxitos** en \( n \) ensayos independientes, cada uno con una probabilidad fija \( p \) de éxito.

## 📌 Propiedades

- **Esperanza:**
 $$\mathbb{E}(X) = n \cdot p$$
- **Varianza:**
  $$\mathrm{Var}(X) = n \cdot p \cdot (1 - p)$$
## 🧪 Ejemplo en Python

```python
n = 10      # número de ensayos

p = 0.6     # probabilidad de éxito

expectation = n * p

variance = n * p * (1 - p)

print("Esperanza:", expectation)

print("Varianza:", variance)

```

  

> 🧠 Ejemplo: Si lanzas una moneda 10 veces y la probabilidad de obtener cara es 0.6, se espera obtener 6 caras con una cierta varianza.
---
# 📊 Propiedades de la Esperanza y Varianza


- **Linealidad de la esperanza:**
  $$\mathbb{E}(aX + b) = a \cdot \mathbb{E}(X) + b$$

- **Varianza con transformación lineal:**

$$\mathrm{Var}(aX + b) = a^2 \cdot \mathrm{Var}(X)$$

- **Suma de variables independientes:**
  $$\mathbb{E}(X + Y) = \mathbb{E}(X) + \mathbb{E}(Y)$$
  $$\mathrm{Var}(X + Y) = \mathrm{Var}(X) + \mathrm{Var}(Y)$$