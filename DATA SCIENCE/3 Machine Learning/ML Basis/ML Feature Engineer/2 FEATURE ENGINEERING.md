#Machine_Learning 

[[2.1 NUM AND CATEG TRANSFORMATIONS]]
[[2.2 DIMENSIONAL TRANSFORMATIONS]]
[[2.3 METHODS FOR FEATURE REDUCTION]]




<div style="text-align: center;">
  <img src="999. IMG FOLDER/image-25.png" alt="Mi Imagen" width="600">
</div>

# 🛠️ Introducción a Feature Engineering

## 📌 ¿Qué es Feature Engineering?

Feature Engineering es el proceso de preparar, transformar, seleccionar o crear variables (llamadas _features_) para mejorar el desempeño de un modelo de aprendizaje automático.

Un _feature_ es cualquier propiedad medible que puede servir como entrada para un modelo. Por ejemplo:

- Temperatura, humedad, altitud para predecir precipitaciones.
    
El proceso de Feature Engineering se vuelve esencial cuando:

- No sabemos qué variables usar.
    
- Hay demasiadas variables.
    
- Algunas están mal formateadas o son redundantes.

## 🎯 ¿Por qué necesitamos Feature Engineering?

Para asegurar que un modelo sea:

1. **Eficaz (Performance)**: Que prediga bien.
    
2. **Rápido (Runtime)**: Que no sea lento en producción.
    
3. **Interpretativo (Interpretability)**: Que dé información útil y comprensible.
    
4. **Generalizable (Generalizability)**: Que funcione con nuevos datos.

---

## 🔄 ¿Dónde encaja en el flujo de Machine Learning?

Aunque se enseña como un paso intermedio entre EDA y modelado, en la práctica **Feature Engineering ocurre antes, durante y después del modelado**.

Se divide en tres grandes categorías:

### 1. 🔧 Métodos de Transformación

Aplican transformaciones a las variables para hacerlas más aptas para los modelos.

Ejemplos:

- Normalización / Escalamiento
    
- One-hot encoding
    
- Binning
    
- Transformaciones logarítmicas
    
- Hashing
    

**Objetivo**: mejorar desempeño, velocidad y comprensión.

---

### 2. 📉 Métodos de Reducción de Dimensionalidad

Reducen el número de _features_ conservando (o mejorando) el rendimiento.

Ejemplos:

- PCA (Análisis de Componentes Principales)
    
- LDA (Análisis Discriminante Lineal)
    

Estas técnicas generan nuevos _features_ matemáticos no directamente interpretables pero muy eficientes computacionalmente.

**También se llaman métodos de Extracción de Características.**

---

### 3. 🎯 Métodos de Selección de Características

Permiten elegir cuáles _features_ usar directamente.

#### i. Filter Methods

Usan estadísticas simples, sin modelo:

- Correlación (Pearson, Spearman)
    
- Chi-cuadrado ($\chi^2$)
    
- ANOVA
    
- Información mutua
    

#### ii. Wrapper Methods

Usan modelos para probar subconjuntos de variables:

- Forward Selection
    
- Backward Elimination
    
- Sequential Floating
    

#### iii. Embedded Methods

Seleccionan _features_ durante el entrenamiento del modelo:

- Regularización (Lasso, Ridge)
    
- Importancia de variables en árboles (Random Forest, XGBoost)
    

---

## 🧠 Resumen Rápido

|Tipo de Técnica|Mejora|¿Cuándo se usa?|
|---|---|---|
|Transformación de Features|Performance, runtime, interpretación|Antes del modelo|
|Reducción de Dimensionalidad|Performance, runtime|Antes/durante el modelado|
|Selección de Características|Interpretabilidad, generalización|Antes o durante el modelado|

> 📦 _Feature Engineering es la etapa donde la intuición humana y la técnica se combinan para construir modelos robustos y eficientes._

