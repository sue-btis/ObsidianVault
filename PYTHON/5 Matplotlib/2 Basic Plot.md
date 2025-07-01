#Python #Matplotlib
[[3 Other plots]]
# 📊 Matplotlib Basics

## 🌟 Introducción

Matplotlib es una biblioteca de visualización de datos en Python que permite crear gráficos estáticos, animados e interactivos. Es especialmente útil para representar datos de forma clara y atractiva.

### 📝 Objetivos

1. Crear gráficos de líneas básicos y múltiples gráficos simultáneamente.
    
2. Personalizar el estilo de las líneas.
    
3. Etiquetar ejes, definir rangos y agregar títulos.
    
4. Utilizar subplots para organizar gráficos múltiples.
    
5. Añadir leyendas y personalizar los ticks del eje.
    

---

## 📈 Basic Line Plot

- Crear un gráfico de líneas básico:
    

``` Python
import matplotlib.pyplot as plt
x = [1, 2, 3, 4]
y = [10, 20, 15, 25]
plt.plot(x, y)
plt.title("Basic Line Plot")
plt.show()
```

### Multiple Line Plots Simultaneously

- Graficar múltiples líneas en el mismo gráfico:
    

```Python
plt.plot(x, y, label='Line 1')
plt.plot(x, [15, 25, 10, 20], label='Line 2')
plt.title("Multiple Line Plots")
plt.legend()
plt.show()
```

---

## 🖌️ Line Styling

- Personalizar el estilo de las líneas (color, grosor, tipo de línea):
    

```Python
plt.plot(x, y, color='blue', linewidth=2, linestyle='--')
plt.title("Styled Line Plot")
plt.show()
```

- Ejemplos de estilos:
    
    - Colores: 'blue', 'green', 'red', etc.
        
    - Tipo de línea: '-', '--', '-.', ':'
        
    - Marcadores: 'o', 's', '^', '*'
        

---

## 🏷️ Axis and Labels

- Añadir nombres a los ejes y establecer el rango:
    

```Python
plt.plot(x, y)
plt.xlabel("Eje X")
plt.ylabel("Eje Y")
plt.xlim(0, 5)
plt.ylim(5, 30)
plt.title("Gráfico con Ejes Nombrados")
plt.show()
```

---

## 🗺️ Subplots

- Crear múltiples gráficos en la misma figura:
    

```Python
import matplotlib.pyplot as plt
x = [1, 2, 3, 4]
y = [1, 2, 3, 4]

# First Subplot
plt.subplot(1, 2, 1)
plt.plot(x, y, color='green')
plt.title('First Subplot')

# Second Subplot
plt.subplot(1, 2, 2)
plt.plot(x, y, color='steelblue')
plt.title('Second Subplot')

# Display both subplots
plt.show()
```

### Ajuste de subplots

- Utilizar `plt.subplots_adjust()` para modificar el espacio entre gráficos:
    

```Python
plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, hspace=0.4, wspace=0.4)
```

---

## 📝 Legends

- Añadir leyendas para identificar cada línea:
    

```Python
plt.plot(x, y, label='Línea 1')
plt.plot(x, [20, 10, 30, 15], label='Línea 2')
plt.legend(loc='upper left')
plt.title("Gráfico con Leyendas")
plt.show()
```

- Posiciones comunes:
    
    - 'upper right', 'upper left', 'lower right', 'lower left', 'center'
        

---

## 📏 Ticks

- Personalizar las marcas de los ejes:
    

```Python
plt.plot(x, y)
plt.xticks([1, 2, 3, 4], ['A', 'B', 'C', 'D'])
plt.yticks([10, 15, 20, 25], ['Bajo', 'Medio', 'Alto', 'Máximo'])
plt.title("Gráfico con Ticks Personalizados")
plt.show()
```

---

## 📚 Buenas prácticas

1. Utilizar colores y estilos consistentes en múltiples gráficos.
    
2. Etiquetar claramente los ejes y añadir leyendas cuando se tracen varias líneas.
    
3. Utilizar subplots para comparar múltiples gráficos en la misma figura.