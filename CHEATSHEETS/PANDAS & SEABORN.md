# 📝 Pandas and Seaborn Cheatsheet

## 🌟 Introducción

Este cheatsheet contiene los comandos más utilizados en Pandas y Seaborn para análisis y visualización de datos. Ideal para tenerlo siempre a mano al trabajar en proyectos de ciencia de datos.

---

## 📊 Pandas

### 🚀 Importar librería

```PYTHON
import pandas as pd
```

### 📂 Crear DataFrame

```PYTHON
df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
```

### 💾 Cargar y guardar datos

```PYTHON
# Leer CSV
df = pd.read_csv('archivo.csv')

# Guardar CSV
df.to_csv('archivo.csv', index=False)
```

### 🔍 Inspección de datos

```PYTHON
df.head()           # Primeras 5 filas
df.tail()           # Últimas 5 filas
df.info()           # Información general
df.describe()       # Estadísticas descriptivas
df.shape            # Dimensiones
df.columns          # Nombres de columnas
df.dtypes           # Tipos de datos
```

### 🔄 Filtrado y selección

```PYTHON
# Filtrar por valor
df[df['A'] > 2]

# Selección por columna
df['A']

# Selección por posición
df.iloc[0, 1]
```

### 🔗 Merge y concatenación

```PYTHON
pd.concat([df1, df2], axis=0)         # Concatenar filas
pd.merge(df1, df2, on='ID', how='inner')  # Merge
```

### 🔧 Limpieza de datos

```PYTHON
df.dropna()               # Eliminar filas con NA
df.fillna(0)              # Rellenar NA con 0
df.duplicated()           # Detectar duplicados
df.drop_duplicates()      # Eliminar duplicados
```

### 📊 Estadísticas descriptivas

```PYTHON
df.mean()         # Media
df.median()       # Mediana
df.std()          # Desviación estándar
df.min()          # Mínimo
df.max()          # Máximo
df.corr()         # Correlación
```

### 📈 Agrupación y agregación

```PYTHON
df.groupby('A').mean()       # Agrupar por 'A' y calcular media
df.agg({'A': ['mean', 'sum'], 'B': 'max'})
```

---

## 🎨 Seaborn

### 🚀 Importar librería

```PYTHON
import seaborn as sns
import matplotlib.pyplot as plt
```

### 📊 Gráficos básicos

```PYTHON
sns.histplot(data=df, x='A')      # Histograma
sns.scatterplot(data=df, x='A', y='B')  # Diagrama de dispersión
sns.boxplot(data=df, y='A')       # Boxplot
```

### 🌟 Gráficos multivariados

```PYTHON
sns.pairplot(df)             # Gráficos de pares
sns.heatmap(df.corr(), annot=True)  # Mapa de calor
sns.jointplot(data=df, x='A', y='B') # Distribución conjunta
```

### 🎨 Personalización de gráficos

```PYTHON
sns.set(style='whitegrid')
plt.title('Mi Gráfico')
plt.xlabel('X Label')
plt.ylabel('Y Label')
```

### 🔧 Ajuste de apariencia

```PYTHON
sns.set_palette('viridis')  # Cambiar paleta de colores
sns.despine()               # Eliminar bordes
```

---

## 🚩 Buenas prácticas

1. Carga y explora los datos antes de realizar gráficos.
    
2. Usa `sns.set()` para configurar la apariencia global.
    
3. Personaliza los gráficos con `matplotlib` para mejorar la presentación.
    

---

## 🔗 Referencias

- [Pandas Documentation](https://pandas.pydata.org/)
    
- Seaborn Documentation
    
- [Matplotlib Documentation](https://matplotlib.org/)