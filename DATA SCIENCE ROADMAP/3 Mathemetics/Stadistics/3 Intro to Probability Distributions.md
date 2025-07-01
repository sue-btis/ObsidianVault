#Math #Stadistics 
[[4 Poisson distribution]]
## 🌟 Introducción

Las distribuciones de probabilidad describen cómo se distribuyen los valores de una variable aleatoria. Son fundamentales en estadística para modelar fenómenos aleatorios tanto en contextos discretos como continuos.

### 📝 Objetivos

1. Comprender los conceptos básicos de distribuciones de probabilidad.
    
2. Diferenciar entre variables aleatorias continuas y discretas.
    
3. Aplicar funciones de masa de probabilidad (PMF) y funciones de densidad de probabilidad (PDF).
    
4. Utilizar funciones de distribución acumulada (CDF) en cálculos probabilísticos.
    

---

## 🎲 Random Variables

- Una **variable aleatoria** es una variable cuyos valores están sujetos a variabilidad aleatoria.
    
- Se pueden clasificar en:
    
    - **Discretas:** Toman un conjunto finito o numerable de valores.
        
    - **Continuas:** Pueden tomar cualquier valor dentro de un rango.
        

**Ejemplo en Python con** `**numpy**`**:**

```Python
import numpy as np
random_choice = np.random.choice([1, 2, 3, 4], size=5)
print("Variable aleatoria discreta:", random_choice)
```

---

## 📏 Discrete and Continuous Random Variables

### Discrete:

- Los valores posibles son específicos y contables.
    
- Ejemplo: Número de caras al lanzar una moneda 10 veces.
    

### Continuous:

- Pueden tomar cualquier valor en un intervalo.
    
- Ejemplo: Tiempo de espera en una cola.
    

---

## 📐 Probability Mass Functions (PMF)

- La PMF asigna probabilidades a cada valor discreto de una variable aleatoria.
    
- **Ejemplo con distribución binomial:**
    

```Python
from scipy.stats import binom
n, p, k = 10, 0.5, 3
pmf_value = binom.pmf(k, n, p)
print("PMF (k=3, n=10, p=0.5):", pmf_value)
```

### Calculating Probabilities of Exact Values

- Utiliza la PMF cuando quieras saber la probabilidad exacta de un valor.
    

---

## 📈 Cumulative Distribution Function (CDF)

- La CDF proporciona la probabilidad acumulada hasta un punto dado.
    
- **Ejemplo con distribución binomial:**
    

```Python
cdf_value = binom.cdf(k, n, p)
print("CDF (k=3, n=10, p=0.5):", cdf_value)
```

### Calculating Probabilities of a Range

- Utiliza la CDF cuando quieras conocer la probabilidad acumulada en un rango de valores.
    

---

## 🧮 Probability Density Functions (PDF)

- La PDF se utiliza para variables continuas, representando la densidad de probabilidad en un punto.
    
- Ejemplo: Distribución normal.
    

**Ejemplo en Python:**

```Python
from scipy.stats import norm
x = 0
pdf_value = norm.pdf(x, loc=0, scale=1)
print("PDF (x=0, μ=0, σ=1):", pdf_value)
```

---

## 🔗 PDF vs CDF

- La **PDF** describe la densidad de probabilidad en un valor específico.
    
- La **CDF** describe la probabilidad acumulada hasta un valor.
    

**Relación:** La CDF es la integral de la PDF sobre el intervalo de interés.

---

## 🧮 ## 📚 Buenas prácticas

1. Identificar si la variable es continua o discreta antes de seleccionar la función.
    
2. Utilizar PMF para variables discretas y PDF para continuas.
    
3. Verificar el uso de funciones en librerías como `numpy` y `scipy`.
    

---

## 🔗 Recursos adicionales

- Curso: "Probability for Data Science" en Coursera.
    
- Documentación de `numpy.random`: Numpy Docs
    
- Documentación de `scipy.stats`: SciPy Docs