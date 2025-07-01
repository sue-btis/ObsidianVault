#Python #Pandas
# 📊 Pandas: Common Statistics Methods, Aggregate Functions, and Pivot Tables

## 🌟 Introducción

Pandas ofrece una variedad de métodos para realizar cálculos estadísticos, agregar datos y generar tablas dinámicas (pivot). Estas funcionalidades son fundamentales para el análisis y resumen de datos en DataFrames.

---
## 📈 Métodos Estadísticos Comunes

Pandas proporciona métodos para calcular estadísticas descriptivas de manera rápida.

### 📊 Métodos Básicos

```python
import pandas as pd
import numpy as np

df = pd.DataFrame({
    'A': [10, 20, 30, 40],
    'B': [5, 15, 25, 35],
    'C': [1, 2, 3, 4]
})

# Media
print(df.mean())

# Mediana
print(df.median())

# Desviación estándar
print(df.std())

# Mínimo y máximo
print(df.min())
print(df.max())

# Suma
print(df.sum())

# Conteo
print(df.count())
```

### 📑 Métodos Adicionales

- **describe()**: Muestra estadísticas resumidas.
    

```python
print(df.describe())
```

- **corr()**: Muestra la correlación entre columnas.
    

```python
print(df.corr())
```

- **cov()**: Muestra la covarianza entre columnas.
    

```python
print(df.cov())
```

---

## 🔄 Aggregate Functions

Las funciones de agregación permiten aplicar operaciones estadísticas agrupando los datos.

### 📂 Ejemplo de agregación

```python
grouped = df.groupby('A').agg({'B': ['mean', 'sum'], 'C': 'max'})
print(grouped)
```

### 🌟 Funciones de agregación comunes

- **mean()**: Promedio de los valores.
    
- **sum()**: Suma de los valores.
    
- **min()**, **max()**: Valor mínimo y máximo.
    
- **count()**: Número de elementos.
    
- **std()**: Desviación estándar.
    
- **var()**: Varianza.
    
### 💡 Aplicación con varias columnas

```python
agg_df = df.agg({'A': ['mean', 'sum'], 'B': ['min', 'max']})
print(agg_df)
```

---

## 🔄 Tablas Pivot (Pivot Tables)

Las tablas pivot en Pandas permiten reorganizar y resumir conjuntos de datos grandes y complejos.

### 📂 Ejemplo básico

```python
df_pivot = pd.DataFrame({
    'City': ['A', 'B', 'A', 'B'],
    'Sales': [100, 200, 150, 250],
    'Year': [2020, 2020, 2021, 2021]
})

pivot = df_pivot.pivot_table(values='Sales', index='City', columns='Year', aggfunc='sum')
print(pivot)
```

### 🔧 Usando múltiples funciones

```python
pivot_multi = df_pivot.pivot_table(values='Sales', index='City', columns='Year', aggfunc=['sum', 'mean'])
print(pivot_multi)
```

### 📊 Pivot con múltiples índices

```python
multi_index_pivot = df_pivot.pivot_table(values='Sales', index=['City', 'Year'], aggfunc='mean')
print(multi_index_pivot)
```

---

## 🚩 Buenas prácticas

1. Usa `describe()` para obtener una visión general rápida de los datos.
    
2. Aplica `groupby()` para agregar datos según categorías.
    
3. Utiliza tablas pivot para explorar relaciones multidimensionales.
    

---

## 🔗 Referencias

- [Pandas Documentation](https://pandas.pydata.org/)
    
- Pivot Table Guide