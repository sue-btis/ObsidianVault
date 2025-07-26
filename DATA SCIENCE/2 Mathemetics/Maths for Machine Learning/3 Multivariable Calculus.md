#Math #Calculus

## Course: #Mathematics_for_Machine_Learning
## Platform: Coursera

# 🌱 Introducción a las Funciones (Pre-Cálculo)



Una función es como una **máquina matemática**: metes un valor (o varios), la máquina hace algo con ellos, y te devuelve un resultado.

> 📦 Entrada → Función → 📤 Salida

Por ejemplo, imagina una función que predice la temperatura en una habitación dependiendo del lugar y la hora:

$$
T(x, y, z, t) \Rightarrow \text{Temperatura en } (x, y, z) \text{ a tiempo } t
$$

Aquí:
- $(x, y, z)$ = coordenadas espaciales (ubicación)
- $t$ = tiempo
- $T$ = temperatura resultante


---

## 🧠 ¿Por Qué la Notación Matemática Es Tan Rara?

Aunque la idea es simple, la **forma en que escribimos funciones puede ser confusa**, especialmente al inicio.

### 📘 Notación típica:

$$
f(x) = x^2 + 3
$$

Significa: “$f$ es una función que toma $x$ y devuelve el cuadrado de $x$ más 3”.

> ❗ ¡No significa que estés multiplicando $f$ por $x$!

### 📚 Cuando hay más cosas en los paréntesis:

$$
f(x) = \frac{g(x)}{h + a}
$$

¿$g$ es una función? ¿$h$ y $a$ son constantes? ¿Variables? ¿Funciones?  
🤔 ¡Depende del contexto! Como en cualquier idioma, necesitas aprender la “gramática” matemática para descifrarlo.

---
# 🚗 Cálculo, Derivadas y Gráficas:  Introducción Intuitiva

##  Gráfica de Velocidad vs Tiempo

<div style="text-align: center;">
  <img src="999. IMG FOLDER/image-21.png" alt="Mi Imagen" width="500">
</div>

Imagina un coche cuyo movimiento está representado por una gráfica de velocidad contra tiempo.

- Si la velocidad es constante: línea horizontal → aceleración = 0
    
- Si la velocidad sube: la pendiente es positiva → aceleración > 0
    
- Si la velocidad baja: pendiente negativa → aceleración < 0
    

Podemos representar la aceleración como la **pendiente local** de la curva de velocidad:

$$
a(t) = \frac{d}{dt}v(t)
$$

Dibujando tangentes en cada punto de la gráfica de velocidad, podemos construir una nueva gráfica: **aceleración vs tiempo**.

> Donde la velocidad tiene su máximo o mínimo (pendiente = 0), la aceleración también es 0.


---

## 🔄 Derivada como Límite

El **gradiente** o pendiente entre dos puntos se aproxima a la derivada si esos puntos están muy cercanos. Esta es la definición formal de derivada:

$$
f'(x) = \lim_{\Delta x \to 0} \frac{f(x + \Delta x) - f(x)}{\Delta x}
$$

Este "cambio instantáneo" de la función es la herramienta clave del cálculo.

> No se trata de dividir entre cero, sino de acercarse lo más posible sin llegar.

---

## 🔁 Antiderivada e Integral

La operación inversa a derivar se llama **antiderivación** o **integración**. Si conocemos la velocidad, podemos hallar la distancia:

$$
\text{distancia} = \int v(t) \, dt
$$

Porque:

$$
\frac{d}{dt}(\text{distancia}) = v(t)
$$

---
## 🧪 Regla del Poder

Para funciones del tipo:

$$f(x) = ax^b$$

Su derivada es:

$$f'(x) = abx^{b - 1}$$

> Muy útil para polinomios. La base baja un grado y se multiplica por el exponente anterior.
---
## ✨ Funciones Especiales

### 1. Recíproca: $$f(x) = \frac{1}{x}$$

- Su derivada es: $$f'(x) = -\frac{1}{x^2}$$
- Tiene una **discontinuidad** en $$x = 0$$ (no definida).

### 2. Exponencial: $$f(x) = e^x$$

- $$f'(x) = e^x$$
- Se deriva infinitas veces sin cambiar.
- $e \approx 2.718$, aparece en todo tipo de procesos naturales.

### 3. Trigonométricas:

$$
\frac{d}{dx}(\sin x) = \cos x \\
\frac{d}{dx}(\cos x) = -\sin x
$$

> Se repiten en ciclos de 4 derivadas, como una rueda que vuelve al inicio.

---

## 📓 Conclusión: El Verdadero Sentido del Cálculo

Aunque aún no hemos formalizado todo, ya sabes que:

- Derivar es **medir la pendiente local** de una función.
- Puedes analizar funciones simples y predecir su comportamiento.
- Pronto verás cómo **diferenciar funciones complejas** sin hacer tanto álgebra.

El cálculo es una forma de ver el mundo en movimiento y entender cómo cambian las cosas, punto por punto.

> 🔹 *"El cálculo no es solo una técnica, es una forma de pensar."*

---
# ⏱️ Reglas que Ahorran Tiempo en Cálculo

A medida que las funciones se vuelven más complejas, calcular derivadas desde la definición formal puede volverse tedioso. Por eso, los matemáticos han desarrollado reglas que simplifican el proceso. Aquí reunimos las **cuatro reglas clave** que conforman tu "caja de herramientas" en cálculo:

---

## 🔗 Regla del Producto
<div style="text-align: center;">
  <img src="999. IMG FOLDER/image-23.png" alt="Mi Imagen" width="300">
</div>
### 🧠 Intuición
Imagina un rectángulo donde los lados son $f(x)$ y $g(x)$. El área $A(x) = f(x)g(x)$.
Cuando cambias $x$, ambos lados cambian, y por lo tanto, también el área.

<div style="text-align: center;">
  <img src="999. IMG FOLDER/image-22.png" alt="Mi Imagen" width="300">
</div>

La derivada del área (cambio en el área respecto a $x$) es:

$$
\frac{d}{dx}(f(x)g(x)) = f(x)g'(x) + g(x)f'(x)
$$

> Multiplica cada función por la derivada de la otra y súmalas.

---

## ⛓️ Regla de la Cadena

<div style="text-align: center;">
  <img src="999. IMG FOLDER/image-24.png" alt="Mi Imagen" width="300">
</div>

### 🧠 Intuición
Se usa cuando una función está **dentro** de otra, como $h(p(m))$.

Ejemplo:
- $h(p)$ = felicidad según pizzas
- $p(m)$ = pizzas según dinero
- Entonces, $h(p(m))$ = felicidad según dinero

La derivada total:

$$
\frac{dh}{dm} = \frac{dh}{dp} \cdot \frac{dp}{dm}
$$

> Deriva cada parte por separado y multiplícalas.

### 🧪 Ejemplo con funciones reales
- $h(p) = -\frac{1}{3}p^2 + p + \frac{1}{5}$
- $p(m) = e^{m - 1}$

Entonces:
$$
\frac{dh}{dm} = \left(1 - \frac{2}{3}p\right) \cdot e^m = \frac{1}{3} e^m (5 - 2e^m)
$$

---

## 🧪 Aplicación Combinada de las Reglas

Considera:
$$
f(x) = \frac{\sin(2x^5 + 3x)}{e^{7x}}
$$

### 1. Reescribir como producto:

$$
f(x) = \sin(2x^5 + 3x) \cdot e^{-7x}
$$

### 2. Derivada de $g(x) = \sin(2x^5 + 3x)$ con regla de cadena:

```python
u(x) = 2x**5 + 3x
# derivada de sin(u(x))
g'(x) = cos(2x**5 + 3x) * (10x**4 + 3)
```

### 3. Derivada de $h(x) = e^{-7x}$:

```python
h'(x) = -7 * e^{-7x}
```

### 4. Aplicar la regla del producto:

```python
f'(x) = g(x) * h'(x) + h(x) * g'(x)
```

---


## 🌐 Cálculo Multivariable y Derivadas Parciales

### 🔢 Variables, Constantes y Parámetros
- **Variables independientes**: se controlan directamente (ej. tiempo)
- **Variables dependientes**: cambian en función de otras (ej. velocidad)
- **Constantes**: no cambian (ej. $\pi$)
- **Parámetros**: se comportan como constantes en un caso, pero pueden variar en otro contexto

---

## 📐 Derivada Parcial Total (Total Derivative)

Cuando tienes una función multivariable como:

$$
f(x, y, z) = \sin(x) e^{y z^2}
$$

...y cada una de esas variables ($x$, $y$, $z$) **depende de otra variable** (como $t$), entonces $f$ **también depende de $t$** de forma compuesta.


### Paso 1: Derivadas parciales de $f$

Asumimos que $x$, $y$ y $z$ son **independientes** al calcular:

- $$\frac{\partial f}{\partial x} = \cos(x) e^{y z^2}$$
- $$\frac{\partial f}{\partial y} = \sin(x) e^{y z^2} \cdot z^2$$
- $$\frac{\partial f}{\partial z} = \sin(x) e^{y z^2} \cdot 2yz$$
### Paso 2: Variables dependientes de $t$

Supón que:

- $x = t - 1$
- $y = t^2$
- $z = \frac{1}{t}$

Entonces $f$ se convierte en $f(x(t), y(t), z(t))$

###  Paso 3: Derivadas de $x$, $y$, $z$ con respecto a $t$

- $$\frac{dx}{dt} = 1$$
- $$\frac{dy}{dt} = 2t$$
- $$\frac{dz}{dt} = -\frac{1}{t^2}$$
### Paso 4: Derivada Total

Combinamos las derivadas parciales con cómo cambian sus variables:

$$
\frac{df}{dt} = 
\frac{\partial f}{\partial x} \cdot \frac{dx}{dt} +
\frac{\partial f}{\partial y} \cdot \frac{dy}{dt} +
\frac{\partial f}{\partial z} \cdot \frac{dz}{dt}
$$

Esta expresión te dice **cómo cambia $f$ con el tiempo considerando TODAS las variables intermedias**.

---

## 📊 ¿Por qué es útil?

En el mundo real:

- Las variables dependen entre sí.
- Cambios en una afectan a otras.
- La derivada total nos da la **tasa de cambio completa**.

Ejemplos:
- Física: fuerzas que dependen de posición y velocidad.
- IA: funciones de pérdida que dependen de pesos y entradas.
- Economía: producción que depende de capital, trabajo y tiempo.

---

## 🧩 Analogía

Imagina que estás en un globo aerostático ($f$) y estás subiendo porque:

- Cambia la altitud ($x$)
- Cambia la temperatura ($y$)
- Cambia el viento ($z$)

Todo esto cambia con el tiempo ($t$). Entonces:

**La derivada total  te dice qué tan rápido sube el globo AHORA MISMO considerando todos esos factores.**

---

## 🧰 Reglas en tu Caja de Herramientas

1. **Regla de la Suma**:
   $$\frac{d}{dx}(f + g) = f' + g'$$

2. **Regla del Poder**:
   $$\frac{d}{dx}(x^n) = nx^{n-1}$$

3. **Regla del Producto**:
   $$\frac{d}{dx}(fg) = f g' + g f'$$

4. **Regla de la Cadena**:
   $$\frac{d}{dx}(f(g(x))) = f'(g(x)) \cdot g'(x)$$

5. **Derivadas Parciales**:
   $$\frac{\partial f}{\partial x},\ \frac{\partial f}{\partial y}, \ldots$$

6. **Derivada Total**:
   $$\frac{df}{dt} = \sum_i \frac{\partial f}{\partial x_i} \cdot \frac{dx_i}{dt}$$

---