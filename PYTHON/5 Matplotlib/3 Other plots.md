#Python #Matplotlib
[[4 HOW TO PICK A CHART]]
# 📊 Matplotlib Advanced Plots

## 🌟 Introducción

Matplotlib permite crear gráficos avanzados como barras, gráficos de pastel, gráficos de área y más. Estos gráficos son ideales para representar datos categóricos, distribuciones y comparar múltiples conjuntos de datos.

### 📝 Objetivos

1. Crear gráficos de barras simples, agrupados y apilados.
    
2. Visualizar errores usando barras de error.
    
3. Crear gráficos de área con `fill_between`.
    
4. Utilizar gráficos de pastel para representar proporciones.
    
5. Generar histogramas de una o múltiples distribuciones.
    

---

## 📊 Bar Chart

### Simple Bar Chart

```Python
import matplotlib.pyplot as plt

drinks = ["cappuccino", "latte", "chai", "americano", "mocha", "espresso"]
sales = [91, 76, 56, 66, 52, 27]
plt.bar(range(len(drinks)), sales)
ax = plt.subplot()
ax.set_xticks(range(len(drinks)))
ax.set_xticklabels(drinks)
plt.title("Simple Bar Chart")
plt.show()
```

### Side-By-Side Bars

```Python
drinks = ["cappuccino", "latte", "chai", "americano", "mocha", "espresso"]
sales1 = [91, 76, 56, 66, 52, 27]
sales2 = [65, 82, 36, 68, 38, 40]

n = 1  # First dataset
t = 2  # Number of datasets
d = 6  # Number of sets of bars
w = 0.8 # Width of each bar
store1_x = [t * element + w * n for element in range(d)]
plt.bar(store1_x, sales1, label='Store 1')

n = 2  # Second dataset
store2_x = [t * element + w * n for element in range(d)]
plt.bar(store2_x, sales2, label='Store 2')

plt.title("Side-By-Side Bar Chart")
plt.legend()
plt.show()
```

### Stacked Bars

```Python
drinks = ["cappuccino", "latte", "chai", "americano", "mocha", "espresso"]
sales1 = [91, 76, 56, 66, 52, 27]
sales2 = [65, 82, 36, 68, 38, 40]
plt.bar(range(6), sales1)
plt.bar(range(6), sales2, bottom=sales1)
plt.title("Stacked Bar Chart")
plt.show()
```

---

## 📏 Error Bars

```Python
ounces_of_milk = [6, 9, 4, 0, 9, 0]
error = [0.6, 0.9, 0.4, 0, 0.9, 0]
plt.bar(range(len(ounces_of_milk)), ounces_of_milk, yerr=error, capsize=5)
plt.title("Error Bars")
plt.show()
```

---

## 📈 Fill Between

```Python
months = range(12)
month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
revenue = [16000, 14000, 17500, 19500, 21500, 21500, 22000, 23000, 20000, 19500, 18000, 16500]
plt.plot(months, revenue)
ax = plt.subplot()
ax.set_xticks(months)
ax.set_xticklabels(month_names)
y_upper = [i + (i * 0.10) for i in revenue]
y_lower = [i - (i * 0.10) for i in revenue]
plt.fill_between(months, y_lower, y_upper, alpha=0.2)
plt.title("Revenue with Uncertainty")
plt.show()
```

---

## 🥧 Pie Chart

```Python
payment_method_names = ["Card Swipe", "Cash", "Apple Pay", "Other"]
payment_method_freqs = [270, 77, 32, 11]
plt.pie(payment_method_freqs, labels=payment_method_names, autopct="%0.1f%%")
plt.axis('equal')
plt.title("Payment Method Distribution")
plt.show()
```

---

## 📊 Histogram

### One Histogram

```Python
import numpy as np
x = np.random.randn(1000)
plt.hist(x, bins=30)
plt.title("Histogram of Random Data")
plt.show()
```

### Multiple Histograms

```Python
x1 = np.random.randn(1000)
x2 = np.random.randn(1000) + 2
plt.hist([x1, x2], bins=30, label=['Data 1', 'Data 2'], alpha=0.7)
plt.title("Overlayed Histograms")
plt.legend()
plt.show()
```

---

## 📚 Buenas prácticas

1. Utilizar etiquetas y leyendas para mejorar la interpretación.
    
2. Ajustar el tamaño de los gráficos según el número de elementos.
    
3. Usar gráficos de barras agrupados para comparaciones entre categorías.