#Python #Seaborn
[[1 PANDAS]]

# 🎨 Seaborn 

## 🌟 Introducción

Seaborn es una biblioteca de visualización de datos en Python basada en Matplotlib. Proporciona una interfaz de alto nivel para crear gráficos estadísticos atractivos y personalizables.

### 🚀 Instalación

```Python
pip install seaborn
```

### 📦 Importación

```Python
import seaborn as sns
import matplotlib.pyplot as plt
```

---
## 📊 Gráficos Básicos

### 📈 Gráfico de Líneas

Muestra tendencias a lo largo del tiempo.

```Python
sns.lineplot(x='fecha', y='valor', data=df)
plt.title('Tendencia de Valor a lo largo del tiempo')
plt.show()
```

### 📊 Gráfico de Barras

Comparación de categorías.

```Python
sns.barplot(x='categoria', y='valor', data=df)
plt.title('Comparación de Categorías')
plt.show()
```

### 📦 Gráfico de Cajas (Boxplot)

Muestra la distribución y detecta outliers.

```Python
sns.boxplot(x='categoria', y='valor', data=df)
plt.title('Distribución por Categoría')
plt.show()
```

### 📑 Gráfico de Violín

Combinación de Boxplot y KDE.

```Python
sns.violinplot(x='categoria', y='valor', data=df)
plt.title('Distribución de Valor por Categoría')
plt.show()
```

---

## 🔗 Gráficos Bivariados

### 📉 Gráfico de Dispersión (Scatterplot)

Para analizar la relación entre dos variables.

```Python
sns.scatterplot(x='x_var', y='y_var', hue='grupo', data=df)
plt.title('Relación entre X y Y')
plt.show()
```

### 📊 Mapa de Calor (Heatmap)

Visualiza la correlación entre variables.

```Python
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Mapa de Calor de Correlación')
plt.show()
```

### 📏 Gráfico de Pares (Pairplot)

Muestra relaciones bivariadas entre múltiples variables.

```Python
sns.pairplot(df, hue='grupo')
plt.show()
```

---

## 📅 Gráficos Multivariados

### 🗺️ Jointplot

Combina un gráfico de dispersión y histogramas.

```Python
sns.jointplot(x='x_var', y='y_var', data=df, kind='reg')
plt.show()
```

### 💡 Gráfico de Enjambre (Swarmplot)

Muestra la distribución categórica sin superposición.

```Python
sns.swarmplot(x='categoria', y='valor', data=df)
plt.title('Distribución sin superposición')
plt.show()
```

---

## 📏 Ajuste de Apariencia

### 🌈 Paleta de Colores

```Python
sns.set_palette('pastel')
```

### 📐 Estilo de Gráfico

```Python
sns.set_style('darkgrid')
```

### 📑 Etiquetas y Títulos

```Python
plt.title('Título del Gráfico')
plt.xlabel('Etiqueta X')
plt.ylabel('Etiqueta Y')
```

---

## 🎛️ Personalización Avanzada

### 💬 Anotaciones

```Python
sns.barplot(x='categoria', y='valor', data=df)
plt.title('Gráfico de Barras')
for index, value in enumerate(df['valor']):
    plt.text(index, value, str(value))
plt.show()
```

### 📏 Ajuste de Tamaño

```Python
plt.figure(figsize=(10, 5))
sns.barplot(x='categoria', y='valor', data=df)
plt.show()
```

### 🔁 Gráficos en Subplots

```Python
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.histplot(df['col1'], ax=axes[0])
sns.histplot(df['col2'], ax=axes[1])
plt.show()
```

---

## 🚩 Buenas Prácticas

1. Ajusta el tamaño del gráfico para mejorar la legibilidad.
    
2. Usa paletas consistentes para gráficos similares.
    
3. Personaliza títulos y etiquetas para cada gráfico.
    
4. Elige el tipo de gráfico adecuado para el análisis.
    

---

## 🔗 Referencias

- Seaborn Documentation
    
- Seaborn Tutorial
    
- [Matplotlib Documentation](https://matplotlib.org/)