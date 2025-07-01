#Python #Pandas


[[1 PANDAS]]
[[1 SEABORN]]
[[2 Aggregate Functions & Pivot]]
[[3 Merge & Concat]]
# 📊 Introduction to Pandas and NumPy

## 🌟 ¿Qué son Pandas y NumPy?

Pandas y NumPy son bibliotecas fundamentales en Python para el análisis de datos y cálculos numéricos:

- **Pandas:** Manejo y análisis de datos tabulares y estructurados (como hojas de cálculo).
    
- **NumPy:** Manipulación eficiente de arrays y operaciones matemáticas de alto rendimiento.
    

---

## 🔑 Estructuras de Datos en Pandas

### 🧩 Series

Una Serie en Pandas es una estructura unidimensional similar a un array con etiquetas para cada elemento.

``` python
import pandas as pd
serie = pd.Series([10, 20, 30], index=['a', 'b', 'c'])
print(serie)
```

**Salida:**

```python
a    10
b    20
c    30
dtype: int64
```

### 🗃️ DataFrames

Un DataFrame es una estructura bidimensional con etiquetas en filas y columnas.

``` python
datos = {'Nombre': ['Ana', 'Luis', 'Juan'], 'Edad': [23, 45, 34]}
df = pd.DataFrame(datos)
print(df)
```

**Salida:**

```python
  Nombre  Edad
0    Ana    23
1   Luis    45
2   Juan    34
```

---

## 🗂️ Crear, Cargar y Seleccionar Datos

### 📂 Creación de DataFrames

```python
import numpy as np
data = np.array([[1, 2, 3], [4, 5, 6]])
df = pd.DataFrame(data, columns=['A', 'B', 'C'])
print(df)
```

### 📥 Cargar datos desde CSV

```python
df = pd.read_csv('archivo.csv')
print(df.head())
```

### 🔍 Selección de Datos

#### Por columna:

- **Directo:**
    
```python
print(df['A'])
```

- **Con notación de punto:**
    
```python
print(df.A)
```

#### Por índice (iloc):

```python
print(df.iloc[0])
print(df.iloc[:, 1])
```

#### Por etiquetas (loc):

```python
print(df.loc[0, 'A'])
print(df.loc[:, 'B'])
```

### ➕ Agregar columnas

#### Directo:

```python
df['D'] = df['A'] + df['B']
print(df)
```

#### Con función lambda:

```python
df['E'] = df['B'].apply(lambda x: x * 2)
print(df)
```

### 🔗 Uso de Lambda para transformar columnas

Las funciones `lambda` se pueden usar con `apply()` para realizar transformaciones rápidas en una columna:

```python
df['Doble'] = df['A'].apply(lambda x: x * 2)
print(df)
```

#### Ejemplo avanzado:

```python
df['Rango'] = df['Edad'].apply(lambda x: 'Adulto' if x >= 18 else 'Menor')
print(df)
```

---

## 📑 Manipulación y Filtrado de Datos

### 🔄 Selección condicional

```python
print(df[df['A'] > 2])
```

### ➕ Agregar columnas

```python
df['D'] = df['A'] + df['B']
print(df)
```

### 🗑️ Eliminar columnas

```python
df = df.drop('D', axis=1)
```

---

## 📈 Buenas prácticas

1. Utiliza `head()` y `info()` para inspeccionar los datos rápidamente.
    
2. Aprovecha el uso de `iloc` y `loc` para acceder a filas y columnas de forma eficiente.
    
3. Utiliza NumPy para realizar operaciones matemáticas rápidas y vectorizadas.
    

---

## 🔗 Referencias

- Pandas Documentation
    
- [NumPy Documentation](https://numpy.org/)