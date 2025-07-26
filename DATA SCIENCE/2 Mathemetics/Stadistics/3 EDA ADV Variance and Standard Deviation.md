#Math #Stadistics 
[[3.3 EDA Detailed Guide]]
[[3.1 EDA ADV Quantiles & Boxplots]]


## 🌟 Introducción

El Análisis Exploratorio de Datos (EDA) no solo se trata de visualizar los datos, sino también de cuantificar su comportamiento. Dos métricas fundamentales para entender la dispersión de los datos son la **varianza** y la **desviación estándar**.

### 📝 Objetivos

1. Calcular varianza y desviación estándar en Python con NumPy.
    
2. Interpretar el significado de ambas métricas.
    
3. Relacionar estos conceptos con la distribución de los datos.
    

---

## 📐 ¿Qué es la Varianza?

- La **varianza** mide la dispersión de un conjunto de datos con respecto a su media.
    
- Fórmula:
    
- Un valor alto de varianza indica que los datos están más dispersos.
    

### Cálculo con NumPy

```Python
import numpy as np
x = [10, 20, 15, 25]
varianza = np.var(x)
print("Varianza:", varianza)
```

### Varianza Muestral

Por defecto, `np.var()` calcula la varianza poblacional. Para varianza **muestral**, se debe usar `ddof=1`:

```Python
var_muestral = np.var(x, ddof=1)
print("Varianza muestral:", var_muestral)
```

---

## 📏 ¿Qué es la Desviación Estándar?

- La **desviación estándar** es la raíz cuadrada de la varianza.
    
- Indica cuánto se alejan, en promedio, los datos de su media.
    

### Cálculo con NumPy

```Python
desviacion = np.std(x)
print("Desviación estándar:", desviacion)
```

### Desviación Estándar Muestral

```Python
desv_muestral = np.std(x, ddof=1)
print("Desviación muestral:", desv_muestral)
```

---

## 📉 Interpretación

- **Baja desviación estándar**: los datos están concentrados cerca de la media.
    
- **Alta desviación estándar**: los datos están dispersos.
    
- Útil para detectar **outliers**, **consistencia** y comparar **dispersión entre grupos**.
    

---

## 🔍 Comparación rápida

|Métrica|Descripción|Cálculo con NumPy|
|---|---|---|
|Varianza poblacional|Dispersión respecto a la media (con `n`)|`np.var(x)`|
|Varianza muestral|Varianza usando `n-1`|`np.var(x, ddof=1)`|
|Desviación estándar|Raíz cuadrada de la varianza|`np.std(x)`|
|Desv. muestral|Con corrección de Bessel (`n-1`)|`np.std(x, ddof=1)`|

---