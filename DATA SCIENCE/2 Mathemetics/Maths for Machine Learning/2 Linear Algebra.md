#Math #Linear_Algebra
[[2 Linear Algebra with Py]]

## Course: #Mathematics_for_Machine_Learning
## Platform: Coursera

# 📐 Vectores 
----
## 📊 Representar Datos como Vectores

*  Representación como vector columna
* En matemáticas y programación, se suele representar así:

$$
\mathbf{r} = \begin{bmatrix} a \\ b \end{bmatrix}
$$

* Esto hace más fácil calcular y visualizar en álgebra lineal y computadoras.

### Ejemplo: Alturas en una población

<div style="text-align: center;">
  <img src="999. IMG FOLDER/image-10.png" alt="Mi Imagen" width="500">
</div>

- Se agrupan por rangos de 2.5 cm para formar un histograma.
- Esto se transforma en un vector de frecuencias:

$$
\mathbf{f} = \begin{bmatrix}
f_{150.0,152.5} \\
f_{152.5,155.0} \\
f_{155.0,157.5} \\
f_{157.5,160.0} \\
f_{160.0,162.5} \\
\vdots
\end{bmatrix}
$$

- Cada componente indica cuántas personas hay en ese rango.

---

## 📈 Modelar Datos con Distribución Normal

### Qué es una curva normal (o gaussiana)

- Una curva en forma de campana que describe cómo se distribuyen los datos.
- ML, se usa para modelar probabilidades o hacer supuestos sobre la forma de los datos.
- Tiene dos parámetros clave:
  - $\mu$: media (centro de la curva)
  - $\sigma$: desviación estándar (qué tan ancha es la curva)

$$
g(x) = \frac{1}{\sigma\sqrt{2\pi}} e^{-\frac{(x-\mu)^2}{2\sigma^2}}
$$

### Vector de parámetros:

$$
\mathbf{p} = \begin{bmatrix} \mu \\ \sigma \end{bmatrix}
$$

### Predicción de frecuencias:

* Usamos una **distribución normal** con parámetros $\mu$ y $\sigma$ para generar una predicción de cómo se distribuirían los datos si siguieran esa curva.

$$
\mathbf{g}_\mathbf{p} = \begin{bmatrix}
g_{150.0,152.5} \\
g_{152.5,155.0} \\
\vdots
\end{bmatrix}
$$

---

## 📏 Medida de Ajuste del Modelo

- Se compara lo que predice el modelo con los datos reales.
- **Residual** = diferencia entre datos reales y predicción.
- Se calcula el **SSR** (suma de residuos al cuadrado):

$$
SSR(p) = \|\mathbf{f} - \mathbf{g}_\mathbf{p}\|^2
$$

🔍 Objetivo: ajustar $\mu$ y $\sigma$ para que SSR sea lo más pequeño posible.

---

## 🧭 Mapas de Contorno para Optimización

<div style="text-align: center;">
  <img src="999. IMG FOLDER/image-11.png" alt="Mi Imagen" width="500">
</div>

- Cada punto $\mathbf{p} = [\mu, \sigma]$ genera una curva diferente.
- Se crea una superficie de valores SSR.
- En un mapa de contorno, las líneas indican niveles de SSR.
- El mejor modelo está en el **mínimo global** de esa superficie.

$$
\Delta\mathbf{p} = \text{dirección que mejora el ajuste del modelo}
$$

---
## 🔁 Operaciones Básicas con Vectores

### **Suma de vectores**

$$
\mathbf{r} = [3, 2], \quad \mathbf{s} = [1, 4]
$$
$$
\mathbf{r} + \mathbf{s} = [3 + 1, 2 + 4] = [4, 6]
$$

### **Multiplicación por un número (escalar)**

$$
\mathbf{r} = [3, 2], \quad a = 2
$$
$$
a\mathbf{r} = 2 \cdot [3, 2] = [6, 4]
$$
### **Resta de vectores**
$$
\mathbf{r} = [3, 2], \quad \mathbf{s} = [1, 4]
$$
$$
\mathbf{r} - \mathbf{s} = [3 - 1, 2 - 4] = [2, -2]
$$
También se puede expresar como:

$$
\mathbf{r} - \mathbf{s} = \mathbf{r} + (-1) \cdot \mathbf{s} = [3, 2] + [-1, -4] = [2, -2]
$$
---

## 📏 Longitud (o Magnitud) de un Vector

Ejemplo físico: la **velocidad de un coche** en línea recta puede representarse con un vector. Su longitud es la rapidez total, sin importar la dirección.

<div style="text-align: center;">
  <img src="999. IMG FOLDER/image-12.png" alt="Mi Imagen" width="250">
</div>
Si:
$$
\mathbf{r} = a\hat{i} + b\hat{j}
$$
Entonces:
$$
\|\mathbf{r}\| = \sqrt{a^2 + b^2}
$$

📌 Esto se deriva del teorema de Pitágoras. Aplica incluso si las componentes tienen unidades distintas (ej. tiempo, dinero, distancia).

---
## ✴️ Producto Punto (Dot Product)

El **producto punto** es una forma de "multiplicar" vectores que **devuelve un número escalar**.

📌 **¿Qué mide?**  
Mide cuánto **uno de los vectores contribuye en la dirección del otro**. Si los vectores fueran fuerzas o velocidades, el producto punto te dice **cuánta fuerza o movimiento va en la misma dirección**.

### 🧠 Ejemplo intuitivo:
Imagina que caminas con viento:

- Si el viento va en la **misma dirección** que tú ⇒ te empuja (dot positivo).
- Si el viento sopla **de lado** ⇒ no te ayuda ni estorba (dot = 0).
- Si el viento viene **de frente** ⇒ te frena (dot negativo).

### 🧮 Fórmula:
En 2D:

$$
\mathbf{r} \cdot \mathbf{s} = r_1s_1 + r_2s_2
$$

Cuando haces el dot product de un vector consigo mismo:

$$
\mathbf{r} \cdot \mathbf{r} = \|\mathbf{r}\|^2
$$

📌 Útil para:
- Obtener magnitudes sin usar raíz cuadrada hasta el final.
- Detectar vectores nulos (si dot = 0).
---
## 📌 Propiedades del Producto Punto

1. **Conmutativo**:  
   $$ \mathbf{r} \cdot \mathbf{s} = \mathbf{s} \cdot \mathbf{r} $$
2. **Distributivo sobre suma**:  
   $$ \mathbf{r} \cdot (\mathbf{s} + \mathbf{t}) = \mathbf{r} \cdot \mathbf{s} + \mathbf{r} \cdot \mathbf{t} $$
3. **Asociativo con escalares**:  
   $$ \mathbf{r} \cdot (a\mathbf{s}) = a(\mathbf{r} \cdot \mathbf{s}) $$
⚙️ Estas propiedades lo hacen **fácil de usar en álgebra lineal, programación y simulaciones físicas**.

---
## 🧠 Ángulo entre Vectores

<div style="text-align: center;">
  <img src="999. IMG FOLDER/image-14.png" alt="Mi Imagen" width="200">
</div>

La conexión con el ángulo entre vectores:
$$
\mathbf{r} \cdot \mathbf{s} = \|\mathbf{r}\| \cdot \|\mathbf{s}\| \cdot \cos(\theta)
$$
### 🎯 ¿Qué nos dice?

- Si $( \theta = 0° )$: **idéntica dirección** → positivo
- Si $( \theta = 90° )$: **perpendiculares** → dot = 0
- Si $( \theta = 180° )$: **dirección opuesta** →  negativo

📌 Se usa para saber si dos movimientos, fuerzas o direcciones **se ayudan, se ignoran o se oponen**.

---
## 🔦 Proyección Escalar y Vectorial

- 🔢 Proyección escalar → te dice “cuánto de $\mathbf{r}$” hay en la dirección de $\mathbf{b}_1$​” → útil para cambiar de base.
    
- ➡️ Proyección vectorial → te da directamente “la sombra de $\mathbf{r}$” sobre $\mathbf{b}_1$​” → útil si necesitas sumar los vectores proyectados.
<div style="text-align: center;">
  <img src="999. IMG FOLDER/image-16.png" alt="Mi Imagen" width="300">
</div>
### 🔹 Proyección escalar
$$
\text{proj}_{\mathbf{r}}(\mathbf{s}) = \frac{\mathbf{r} \cdot \mathbf{s}}{\|\mathbf{r}\|}
$$

---
### 🔸 Proyección vectorial
$$
\text{Proj}_{\mathbf{r}}(\mathbf{s}) = \left( \frac{\mathbf{r} \cdot \mathbf{s}}{\mathbf{r} \cdot \mathbf{r}} \right) \mathbf{r}
$$
---
## 🧩 Intuición Final

🔍 El producto punto no es solo una fórmula:

- Es una herramienta para **medir alineación**.
- Sirve para saber **qué tanto dos vectores trabajan juntos o se cancelan**.
- Es clave para **machine learning**, **física**, **3D graphics**, y más.

💡 **Cuando haces un dot product, estás colapsando un vector sobre otro**. Te ayuda a **comparar direcciones, extraer componentes útiles** y entender cómo interactúan dos efectos.

---
# 🧭 Cambios de Base, Proyecciones y Espacios Vectoriales

## 📌 ¿Por qué importa la base?

Cuando usamos vectores, usualmente lo hacemos dentro de un **sistema de coordenadas**, definido por un conjunto de **vectores base**. Estos vectores base nos dicen cómo movernos en el espacio. Pero:

🧠 **El vector existe independientemente de la base**. Solo cambia cómo lo describimos (sus "coordenadas").

---

## 🔁 Cambiar de Base (de $\mathbf{e}$ a $\mathbf{b}$)

Supón que un vector $\mathbf{r} = [3, 4]$ está expresado con la base estándar $\{\hat{e}_1, \hat{e}_2\}$.

Ahora queremos reescribirlo usando otra base $\{\mathbf{b}_1, \mathbf{b}_2\}$, donde:

$\mathbf{b}_1 = [2, 1], \quad \mathbf{b}_2 = [-2, 4]$

✅ Verificamos que $\mathbf{b}_1 \perp \mathbf{b}_2$:

$\mathbf{b}_1 \cdot \mathbf{b}_2 = 2(-2) + 1(4) = -4 + 4 = 0 \Rightarrow \text{Son ortogonales}$

Entonces podemos usar **proyecciones**:

$r_{b_1} = \frac{\mathbf{r} \cdot \mathbf{b}_1}{\|\mathbf{b}_1\|^2} = \frac{3(2) + 4(1)}{2^2 + 1^2} = \frac{10}{5} = 2$

$r_{b_2} = \frac{\mathbf{r} \cdot \mathbf{b}_2}{\|\mathbf{b}_2\|^2} = \frac{3(-2) + 4(4)}{(-2)^2 + 4^2} = \frac{10}{20} = 0.5$

🔄 Entonces:

$\mathbf{r} = 2 \cdot \mathbf{b}_1 + 0.5 \cdot \mathbf{b}_2$

👉 Hemos convertido la representación de $\mathbf{r}$ desde la base $\mathbf{e}$ a la base $\mathbf{b}$.

---

## 📐 ¿Qué es una Base?

Un **conjunto de vectores linealmente independientes** que:
- No pueden escribirse unos en función de otros.
- Juntos generan todo el espacio (span).

🔢 El número de vectores base = **dimensión del espacio**.

- 1 vector independiente → línea (1D)
- 2 independientes → plano (2D)
- 3 independientes → espacio (3D)

👉 Si añades un vector y **no es combinación lineal** de los anteriores, creas una dimensión nueva.

---

## 🧠 Intuición Aplicada a Datos

<div style="text-align: center;">
  <img src="999. IMG FOLDER/image-17.png" alt="Mi Imagen" width="300">
</div>
Imagina que tienes puntos de datos en 2D que **caen casi sobre una línea**.

✅ Podrías definir:
- Un eje nuevo "a lo largo de la línea"
- Otro eje "perpendicular" (que mida la distancia desde la línea)

👉 Esto se parece mucho a **reducción de dimensionalidad** (como en PCA):

- El eje de la línea representa la **información importante**
- El eje perpendicular mide el **ruido** o error

En redes neuronales:
- Las bases podrían representar "rasgos latentes" como **forma de nariz**, **tono de piel**, etc.
- El modelo aprende una nueva base útil para representar los datos.

---

## 🔄 Base Natural vs. Base Aprendida

- Base natural: $[1, 0], [0, 1]$
- Base transformada: puede ser cualquier par de vectores independientes (aunque no ortogonales)

👉 Cambiar la base puede:
- Alinear los datos para análisis más simple
- Reducir ruido
- Identificar patrones ocultos

---

# 🍎Matriz

Qué tienen en común manzanas, plátanos y matrices?

 ¿cuánto cuesta cada fruta si solo sabemos el total de la cuenta? Si compras 2 manzanas y 3 plátanos por €8, y otro día compras 10 manzanas y 1 plátano por €13, puedes plantear el problema como un sistema de ecuaciones. Pero lo más poderoso es que **también puedes resolverlo con matrices**.

---
## 🧮 Representando el problema como matriz

La forma matricial del sistema es:

$$
\begin{bmatrix} 2 & 3 \\ 10 & 1 \end{bmatrix} \begin{bmatrix} a \\ b \end{bmatrix} = \begin{bmatrix} 8 \\ 13 \end{bmatrix}
$$

Esta ecuación resume la idea de que una **matriz puede actuar sobre un vector** y devolver otro vector. Aquí, los precios $(a, b)$ de las frutas se transforman en los totales €8 y €13.

---

## 🧭 ¿Qué hace realmente una matriz?

Visualmente, una matriz **transforma el espacio**: toma vectores base (como $\hat{e}_1 = [1, 0]$ y $\hat{e}_2 = [0, 1]$) y los lleva a nuevas posiciones. En este ejemplo:

<div style="text-align: center;">
  <img src="999. IMG FOLDER/image-18.png" alt="Mi Imagen" width="300">
</div>

$$
\text{Matriz} = \begin{bmatrix} 2 & 3 \\ 10 & 1 \end{bmatrix} \Rightarrow \begin{cases} \hat{e}_1 \to [2, 10] \\ \hat{e}_2 \to [3, 1] \end{cases}
$$

Esto significa que el espacio se ha estirado, rotado o deformado, y cualquier vector será una combinación de esas nuevas direcciones.

---

## 🎯 Propiedades clave de las matrices

¿Por qué es útil esto? Porque estas operaciones **respetan la estructura lineal**:

- Escalado: $A(n \mathbf{r}) = n A(\mathbf{r})$
- Suma: $A(\mathbf{r} + \mathbf{s}) = A(\mathbf{r}) + A(\mathbf{s})$

Esto asegura que **la combinación lineal de vectores** se transforma en la misma combinación de sus transformados.

---

## 🔄 Tipos de transformaciones

Las matrices permiten **describir visual y funcionalmente** los cambios espaciales. Cada tipo modifica el espacio de forma distinta:

| Tipo            | Matriz                                   | Qué hace                                                            |
|-----------------|-------------------------------------------|---------------------------------------------------------------------|
| Identidad       | $\begin{bmatrix}1 & 0 \\ 0 & 1\end{bmatrix}$     | No cambia nada                                                      |
| Escalado        | $\begin{bmatrix}3 & 0 \\ 0 & 2\end{bmatrix}$     | Estira los ejes (x3 en x, x2 en y)                                 |
| Reflejo         | $\begin{bmatrix}-1 & 0 \\ 0 & 1\end{bmatrix}$    | Refleja sobre eje y                                                 |
| Inversión total | $\begin{bmatrix}-1 & 0 \\ 0 & -1\end{bmatrix}$   | Refleja ambos ejes (giro de 180°)                                   |
| Cizalla         | $\begin{bmatrix}1 & k \\ 0 & 1\end{bmatrix}$      | Desplaza filas paralelamente (paralelogramo)                        |
| Rotación        | $\begin{bmatrix}\cos\theta & -\sin\theta \\ \sin\theta & \cos\theta\end{bmatrix}$ | Gira todo el espacio                                                 |

Cada transformación es una herramienta para modificar un objeto o conjunto de datos sin perder estructura.

---

## 🧰 Composición de transformaciones

Al aplicar varias transformaciones, **el orden importa**. Si aplicas primero una rotación y luego un reflejo, obtendrás un resultado diferente a hacerlo al revés.

Ejemplo:

- Rotación $90^\circ$ CCW:
$$ A_1 = \begin{bmatrix} 0 & -1 \\ 1 & 0 \end{bmatrix} $$
- Reflejo vertical:
  $$ A_2 = \begin{bmatrix} -1 & 0 \\ 0 & 1 \end{bmatrix} $$
- Composición:
  $$ A_2 A_1 \neq A_1 A_2 $$

Esto nos lleva a un concepto clave: **la multiplicación de matrices no es conmutativa**.

---
## 🧠 ¿Por qué transformar vectores con matrices?

Aunque los **vectores representan datos**, las **matrices nos permiten cambiar la forma en que los observamos**.

Transformar datos no cambia su esencia, **cambia su perspectiva**. Al aplicar una matriz, podemos:

- 🔍 **Descubrir patrones** ocultos al rotar o proyectar los datos.
    
- 📐 **Reducir dimensiones**, conservando lo más importante (como en PCA).
    
- 🧠 **Prepararlos para modelos** que aprenden mejor en espacios específicos.
    
- 📊 **Descorrelacionar variables** y facilitar el análisis.
    

> Es como ver una escultura desde otro ángulo: **es la misma**, pero entiendes mejor su forma.

---


## 🔁 Introducción a la matriz inversa

La **matriz inversa** $A^{-1}$ cumple:

$$
A^{-1} \cdot A = I
$$

Donde $I$ es la **matriz identidad**, es decir:

$$
I = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}
$$

Esta matriz deja todo igual al multiplicar: $I \cdot \vec{x} = \vec{x}$. Es el equivalente al número 1 en la multiplicación escalar.

Si logramos encontrar $A^{-1}$, podemos resolver:

$$
\vec{r} = A^{-1} \cdot \vec{s}
$$

Esto permite encontrar $\vec{r}$ para **cualquier** vector de salida $\vec{s}$.

## ✳️ Eliminación y sustitución hacia atrás

En lugar de calcular directamente la inversa, podemos resolver mediante:

1. **Eliminación de filas (reducción a forma escalonada o Echelon)**
2. **Sustitución hacia atrás (Back-substitution)**

Ejemplo con sistema ampliado:

$$
\begin{bmatrix} 1 & 1 & 3 \\ 1 & 2 & 4 \\ 1 & 1 & 2 \end{bmatrix} \cdot \begin{bmatrix} a \\ b \\ c \end{bmatrix} = \begin{bmatrix} 15 \\ 21 \\ 13 \end{bmatrix}
$$

Mediante operaciones entre filas se reduce a:

$$
\begin{bmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 1 \end{bmatrix} \cdot \begin{bmatrix} a \\ b \\ c \end{bmatrix} = \begin{bmatrix} 5 \\ 4 \\ 2 \end{bmatrix}
$$

➡️ Resultado: manzana = \$5, banana = \$4, zanahoria = \$2

## 🔄 Método para encontrar la inversa

1. Se toma la matriz $A$ y se **acompaña** con la identidad del mismo tamaño:

$$
\left[ A \mid I \right]
$$

2. Se aplican **operaciones fila** para transformar $A$ en $I$.
3. Al terminar, la parte derecha se habrá convertido en $A^{-1}$.

Ejemplo con matriz $3 \times 3$:

$$
\left[\begin{array}{ccc|ccc}
1 & 1 & 3 & 1 & 0 & 0 \\
1 & 2 & 4 & 0 & 1 & 0 \\
1 & 1 & 2 & 0 & 0 & 1
\end{array}\right]
\Rightarrow \cdots \Rightarrow
\left[\begin{array}{ccc|ccc}
1 & 0 & 0 & 1 & -1 & 2 \\
0 & 1 & 0 & -2 & 1 & 1 \\
0 & 0 & 1 & 1 & 0 & -1
\end{array}\right]
$$

La parte derecha es la matriz inversa de $A$.

---
### 🧠 ¿Por qué es útil encontrar la matriz inversa?

#### 1. **Resolver múltiples sistemas con la misma matriz**

Imagina que tienes una tienda y usas la misma **estructura de productos** (por ejemplo, precios de manzanas, bananas y zanahorias). Cada cliente compra diferentes cantidades, es decir, diferentes vectores $\vec{s}$ (como facturas).

La ecuación es:

$$
 A \cdot \vec{r} = \vec{s}
$$

Donde:

- $A$ representa los precios fijos por producto.
    
- $\vec{r}$ es la cantidad que buscas.
    
- $\vec{s}$ es el total de la compra.
    

Si **ya tienes la inversa $A^{-1}$ calculada**, puedes resolver **cualquier** factura (cualquier $\vec{s}$ nuevo) con solo:

$$
\vec{r} = A^{-1} \cdot \vec{s}
$$
🔁 Esto **evita repetir** todo el proceso de eliminación fila por fila para cada nuevo caso. Solo aplicas la fórmula.

---

#### 2. **Transformaciones de espacio y datos**

Cuando una matriz $A$ actúa sobre un vector $\vec{r}$, **lo transforma**: lo estira, rota, refleja o lo traslada dentro del espacio vectorial.

Con la **inversa** $A^{-1}$ puedes:

- Recuperar el vector **original** (deshacer la transformación).
    
- Entender **cómo cambian los datos** cuando pasas de una representación a otra (por ejemplo, de coordenadas normales a coordenadas de componentes principales en PCA).
    
- Analizar cómo diferentes entradas afectan salidas, o viceversa.

---

## 🧰 ¿Qué es el determinante?
El **determinante** de una matriz cuadrada mide cuánto se **escala el espacio** cuando aplicamos esa matriz como una transformación lineal.

Mide la **escala** de transformación del espacio (área, volumen, etc.)

- Para una matriz $2 \times 2$:
  $$
  A = \begin{bmatrix} a & b \\ c & d \end{bmatrix}
  $$
  El determinante se calcula como:
  $$
  \det(A) = ad - bc
  $$

## 📐 Interpretación geométrica

### 🔲 Caso 1: Matriz diagonal (escala directa)
Si:
$$
A = \begin{bmatrix} a & 0 \\ 0 & d \end{bmatrix}
$$

Transforma los vectores base así:
- $\hat{e}_1 = [1, 0] \rightarrow [a, 0]$
- $\hat{e}_2 = [0, 1] \rightarrow [0, d]$

Esto forma un rectángulo de área $a \cdot d = \det(A)$.

## 📉 ¿Qué ocurre si $\det(A) = 0$?

Cuando:
- Los vectores transformados están alineados (uno es múltiplo del otro)
- El área del paralelogramo es 0
- El espacio se **colapsa a una línea**

Entonces:
- La matriz **no tiene inversa**
- Se pierde una dimensión
- La información es irrecuperable
--- 

# ✍️ Convención de Sumatoria de Einstein y Multiplicación de Matrices
La **Convención de Sumatoria de Einstein** es una notación elegante y compacta para expresar operaciones con matrices y vectores, especialmente útil en programación, álgebra lineal y física. Esta notación:

> **Asume una suma sobre cualquier índice que aparece repetido en una expresión.**  
> No se necesita escribir el símbolo ∑ explícitamente.

---

## 📐 Multiplicación de Matrices en Notación de Einstein

Dado que:

- $A$ es una matriz de tamaño $n \times n$ con elementos $A_{ij}$.
- $B$ es una matriz de tamaño $n \times n$ con elementos $B_{jk}$.
- Entonces, su producto $C = AB$ tiene elementos:

$$
C_{ik} = \sum_j A_{ij} B_{jk}
$$

Bajo la **convención de Einstein**, se omite el símbolo de suma:

$$
C_{ik} = A_{ij} B_{jk}
$$

Se sobreentiende la suma sobre el índice **repetido** $j$.


---

## 🔄 Producto Punto como Multiplicación de Matrices

Dado dos vectores columna $u_i$ y $v_i$, su **producto punto** es:

$$
u \cdot v = \sum_i u_i v_i = u_i v_i
$$

Este producto es **equivalente** a una multiplicación de matrices:

$$
u^\top v = u_i v_i
$$

---

## 📊 Proyección y Simetría del Producto Punto

Supón que $\hat{u}$ es un vector unitario con componentes $u_1, u_2$, y los vectores base canónicos son:

$$
\hat{e}_1 = \begin{bmatrix}1 \\ 0\end{bmatrix}, \quad \hat{e}_2 = \begin{bmatrix}0 \\ 1\end{bmatrix}
$$

### Proyección de $\hat{u}$ sobre $\hat{e}_1$:

La proyección es simplemente $u_1$.

### Proyección de $\hat{e}_1$ sobre $\hat{u}$:

Geométricamente, se obtiene una proyección **idéntica** en magnitud. Esto refleja que:

$$
\hat{u} \cdot \hat{e}_1 = \hat{e}_1 \cdot \hat{u}
$$

> El producto punto es **simétrico**, y la proyección también.

---

## 🧩 Multiplicación de Matrices No Cuadradas

Se puede multiplicar una matriz $A$ de $m \times n$ por otra $B$ de $n \times k$:
Debe tener mismo numero de columnas matriz $A$ que de filas la matriz $B$

- $A_{ij}$ con $i = 1,\dots,m$, $j = 1,\dots,n$
- $B_{jk}$ con $j = 1,\dots,n$, $k = 1,\dots,k$

Producto resultante $C_{ik}$:

$$
C_{ik} = A_{ij} B_{jk}
$$

**Resultado:** matriz de tamaño $m \times k$

---

# 🔄 Cambio de Base y Transformaciones con Matrices

## 🧭 **Definiciones Clave**

- Las **columnas de una matriz de transformación** representan los **vectores base** del nuevo sistema en coordenadas del sistema original.
- Transformar un vector de un sistema a otro requiere **multiplicación matricial**.

--- 
### 🧱 1. **Columnas de una matriz de transformación = Vectores base del nuevo sistema**

Imagina que tienes dos mundos:

- 🌍 Tu mundo: usa los ejes estándar (horizontal = x, vertical = y).
    
- 🐼 Mundo de Panda: usa ejes inclinados o deformados (por ejemplo, 3 a la derecha y 1 arriba).
    
Cuando quieres representar el mundo de Panda en tu sistema, **necesitas saber cómo se ven los ejes de Panda en tu mundo**. Y eso se logra colocando sus vectores base como **columnas** en una **matriz de transformación**:
## 🧭 Ejemplo Conceptual

- En el mundo de Panda, sus ejes son:

$$
\hat{e}_1 = \begin{bmatrix} 1 \\ 0 \end{bmatrix}, \quad
\hat{e}_2 = \begin{bmatrix} 0 \\ 1 \end{bmatrix}
$$

- Pero en tu mundo, esos ejes **no se ven igual**. Tú ves los vectores base de Panda así (  matriz de transformación):

$$
B = \begin{bmatrix} 3 & 1 \\ 1 & 1 \end{bmatrix}
$$

Esto significa:

- El eje 1 de Panda se ve como $(3, 1)$ en tu sistema.
- El eje 2 de Panda se ve como $(1, 1)$ en tu sistema.

> Cada **columna** = un **vector base de Panda** en tu sistema.
> **B es cómo se ve el sistema de Panda desde tu sistema.**


---
## 🔁 ¿Qué hacen B y B⁻¹?

### **1. Para convertir de la base de Panda → a tu sistema:**

Multiplicas por **B**:

$$
\text{Vector en tu sistema} = B \cdot \text{Vector en base de Panda}
$$

> Traduces un vector **expresado en la base de Panda** a **tu sistema estándar**.

#### 🐼 Ejemplo:
Supongamos que Panda usa los siguientes vectores base (escritos en tu sistema):

$$
B = \begin{bmatrix} 3 & 1 \\ 1 & 1 \end{bmatrix}
$$

Y en su sistema, el vector es:

$$
\vec{v}_\text{Panda} = \begin{bmatrix} 1 \\ 0 \end{bmatrix}
$$

Entonces, en tu sistema, ese vector se ve como:

$$
\vec{v}_\text{tuyo} = B \cdot \vec{v}_\text{Panda} = 
\begin{bmatrix} 3 & 1 \\ 1 & 1 \end{bmatrix}
\begin{bmatrix} 1 \\ 0 \end{bmatrix}
=
\begin{bmatrix} 3 \\ 1 \end{bmatrix}
$$

---

### **2. Para convertir de tu sistema → a la base de Panda:**

Multiplicas por **la inversa de B**:

$$
\text{Vector en base de Panda} = B^{-1} \cdot \text{Vector en tu sistema}
$$

> Estás expresando un vector **de tu sistema** en **términos de la base de Panda**.

#### 🐼 Ejemplo:
Tomamos el mismo vector en tu sistema:

$$
\vec{v}_\text{tuyo} = \begin{bmatrix} 3 \\ 1 \end{bmatrix}
$$

La inversa de $B$ es:

$$
B^{-1} = \frac{1}{2}
\begin{bmatrix} 1 & -1 \\ -1 & 3 \end{bmatrix}
$$

Entonces:

$$
\vec{v}_\text{Panda} = B^{-1} \cdot \vec{v}_\text{tuyo} =
\frac{1}{2}
\begin{bmatrix} 1 & -1 \\ -1 & 3 \end{bmatrix}
\begin{bmatrix} 3 \\ 1 \end{bmatrix}
=
\frac{1}{2}
\begin{bmatrix} 2 \\ 0 \end{bmatrix}
=
\begin{bmatrix} 1 \\ 0 \end{bmatrix}
$$

Y así comprobamos que el vector $(3, 1)$ en tu sistema es realmente el vector $(1, 0)$ en la base de Panda.

---
<div style="text-align: center;">
  <img src="999. IMG FOLDER/image-19.png" alt="Mi Imagen" width="600">
</div>
## 🟧 Base Ortogonal de Panda

Cuando la base es **ortonormal** (longitud 1 y ortogonalidad), la transformación inversa es más simple:

Una **base ortonormal** es un conjunto de vectores que cumplen **dos condiciones clave**:

1. **Ortogonalidad** → todos los vectores son **perpendiculares** entre sí.
    
2. **Norma unitaria** → todos los vectores tienen **longitud 1**.
    
Visualmente, son como tus ejes clásicos $x$ y $y$, pero **rotados o reflejados**.

- Base de Panda ortonormal:
  $$
  B = \frac{1}{\sqrt{2}}
  \begin{bmatrix}
  1 & -1 \\
  1 & 1
  \end{bmatrix}
  $$


En general, para transformar de tu sistema al sistema de Panda necesitas:

$$
\vec{v}_\text{Panda} = B^{-1} \cdot \vec{v}_\text{tuyo}
$$

Pero como **B es ortonormal**, entonces:

$$
B^{-1} = B^T
$$

Es decir, ¡la inversa es simplemente la **transpuesta**!

**La transpuesta de una matriz $B$ se obtiene intercambiando filas por columnas.**

$$
B = \frac{1}{\sqrt{2}}
\begin{bmatrix}
1 & -1 \\
1 & 1
\end{bmatrix}
$$
Entonces su transpuesta (e inversa) es:

$$
B^T = \frac{1}{\sqrt{2}}
\begin{bmatrix}
1 & 1 \\
-1 & 1
\end{bmatrix}
$$


---

## 📐 Proyecciones como Transformación

<div style="text-align: center;">
  <img src="999. IMG FOLDER/proyeccion_ortonormal-1.png" alt="Mi Imagen" width="400">
</div>

Cuando los vectores base forman una **base ortonormal**, podemos evitar el uso de matrices para cambiar de base. En su lugar, usamos **producto punto**.

Para una base ortonormal, se puede evitar usar matrices:
	
* Para cada vector base ortonormal $\hat{b}_i$, calculamos cuánto del vector $\vec{v}$ "apunta" en esa dirección usando el producto punto:
  $$
  c_i = \vec{v} \cdot \hat{b}_i
  $$

- Vector en nueva base:
  $$
  \vec{v}_\text{nueva} = [c_1, c_2, ..., c_n]
  $$

> ⚠️ Solo funciona si la base es ortonormal. No funciona si los vectores base no son ortogonales.

### 🧠 Ejemplo numérico

Supón que:

$$
\vec{v} = \begin{bmatrix} 1 \\ 3 \end{bmatrix}, \quad
\hat{b}_1 = \frac{1}{\sqrt{2}} \begin{bmatrix} 1 \\ 1 \end{bmatrix}, \quad
\hat{b}_2 = \frac{1}{\sqrt{2}} \begin{bmatrix} -1 \\ 1 \end{bmatrix}
$$

Entonces:

- $c_1 = \vec{v} \cdot \hat{b}_1 = \frac{1}{\sqrt{2}}(1 + 3) = 2\sqrt{2}$
- $c_2 = \vec{v} \cdot \hat{b}_2 = \frac{1}{\sqrt{2}}(-1 + 3) = \sqrt{2}$

Vector en la nueva base:

$$
\vec{v}_\text{nueva} = \begin{bmatrix} 2\sqrt{2} \\ \sqrt{2} \end{bmatrix}
$$

---

## 🌀 Transformaciones dentro de Bases No Ortogonales

Digamos que:
- Tienes una base "rara" como la de Panda (no ortonormal, por ejemplo $B = \begin{bmatrix} 3 & 1 \ 1 & 1 \end{bmatrix}$).
    
- Quieres aplicar una transformación, por ejemplo, una **rotación de 45°**.
    
- Pero... **esa rotación está escrita en tu base estándar (la ortonormal)**.
    
Entonces surge el problema:

> ❓ ¿Cómo aplicar una transformación escrita en tu mundo al vector que vive en el mundo de Panda?

### 🧠 La solución en 3 pasos

#### 🔹 1. Convierte el vector de Panda a tu sistema

Este paso te permite **interpretar el vector de Panda usando tus coordenadas**:

$$
\vec{v} ' = B \cdot \vec{v}_\text{Panda}
$$

Aquí $\vec{v}_\text{Panda}$ es el vector descrito con los ejes de Panda, y $B$ lo traduce a tu mundo.

---

#### 🔹 2. Aplica la transformación (por ejemplo, rotación $R$)

Ahora que tienes el vector en tu sistema, aplicas la transformación normalmente:

$$
R \cdot \vec{v}'
$$

Por ejemplo, una rotación de 45° en tu base es:

$$
R = \frac{1}{\sqrt{2}} \begin{bmatrix} 1 & -1 \\ 1 & 1 \end{bmatrix}
$$

---

#### 🔹 3. Regresa el resultado al mundo de Panda

Una vez que el vector ha sido transformado, lo conviertes de vuelta a la base de Panda:

$$
\vec{v}_\text{rotada} = B^{-1} \cdot (R \cdot \vec{v}')
$$

Esto expresa el vector final **en términos de la base de Panda**.


> Esto se resume como:

$$
\vec{v}_\text{rotada} = B^{-1} R B \cdot \vec{v}_\text{Panda}
$$
### 📌 Intuición

- $B$ lleva **de Panda a ti**  
- $R$ aplica la transformación **en tu mundo**  
- $B^{-1}$ lleva **de vuelta a Panda**

Panda no sabe qué es una rotación de 45° porque su base no es ortonormal.  
Tú haces la rotación en tu lenguaje y luego se la traduces de regreso.

---
## Ejemplo: Transformación en Base No Ortonormal

### 🐼 Base de Panda (no ortonormal):

$$
B = \begin{bmatrix} 3 & 1 \\ 1 & 1 \end{bmatrix}
$$
 Vector en la base de Panda:

$$
\vec{v}_\text{Panda} = \begin{bmatrix} 2 \\ 1 \end{bmatrix}
$$
### 🔁 Paso 1: Llevar el vector al sistema estándar

Multiplicamos por $B$:

$$
\vec{v}' = B \cdot \vec{v}_\text{Panda} = \begin{bmatrix} 3 & 1 \\ 1 & 1 \end{bmatrix} \cdot \begin{bmatrix} 2 \\ 1 \end{bmatrix} = \begin{bmatrix} 7 \\ 3 \end{bmatrix}
$$


### 🔄 Paso 2: Aplicar una rotación de 45° en el sistema estándar

Matriz de rotación:

$$
R = \frac{1}{\sqrt{2}} \begin{bmatrix} 1 & -1 \\ 1 & 1 \end{bmatrix}
$$

Aplicamos la rotación:

$$
\vec{v}_\text{rotado}' = R \cdot \vec{v}' = \begin{bmatrix} 2.8284 \\ 7.0711 \end{bmatrix}
$$

### 🔃 Paso 3: Regresar a la base de Panda

Inversa de $B$:

$$
B^{-1} = \begin{bmatrix} 1 & -1 \\ -1 & 3 \end{bmatrix} \cdot \frac{1}{2}
$$

Aplicamos la transformación inversa:

$$
\vec{v}_\text{rotada} = B^{-1} \cdot \vec{v}_\text{rotado}' = \begin{bmatrix} 2.0 \\ 1.0 \end{bmatrix}
$$
Transpuesta, Inversa y Matrices Ortogonales

### 🔄 ¿Qué es la transpuesta?

La **transpuesta** de una matriz $A$ es una matriz $A^T$ donde se **intercambian filas por columnas**. Formalmente:

$$
(A^T)_{ij} = A_{ji}
$$

Ejemplo:

$$
A = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix} 
\Rightarrow 
A^T = \begin{bmatrix} 1 & 3 \\ 2 & 4 \end{bmatrix}
$$

---
## 🔁 Transpuesta  Matrices Ortogonales

### 🔄 ¿Qué es la transpuesta?

La **transpuesta** de una matriz $A$ es una matriz $A^T$ donde se **intercambian filas por columnas**. Formalmente:

$$
(A^T)_{ij} = A_{ji}
$$

Ejemplo:

$$
A = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix} 
\Rightarrow 
A^T = \begin{bmatrix} 1 & 3 \\ 2 & 4 \end{bmatrix}
$$

---

### 🧱 Matrices de cambio de base con vectores ortonormales

Una matriz de cambio de base $A$ cuyos vectores columna son **ortonormales** (perpendiculares y de longitud 1) cumple:

- $a_i \cdot a_j = 0$ si $i \ne j$ (ortogonalidad)
- $a_i \cdot a_i = 1$ (longitud unitaria)

Entonces:

$$
A^T A = I
$$

¡Esto significa que la transpuesta de $A$ es su inversa!:

$$
A^{-1} = A^T
$$

---

### 📐 ¿Qué es una matriz ortogonal?

Una matriz **ortogonal** es una matriz cuadrada cuyas columnas (o filas) son un conjunto ortonormal. Cumple:

$$
A^T = A^{-1}
$$

Además:

- Su determinante es $\pm 1$
- No deforma el espacio (preserva distancias y ángulos)

---

### 🎯 Aplicaciones en ciencia de datos

Usar **matrices ortogonales** para transformar datos:

- Permite revertir fácilmente la transformación
- No colapsa el espacio
- El cambio de base se convierte en una simple proyección (producto punto)
- Es ideal en algoritmos como **PCA**

---
## 🧠 Proceso de Gram-Schmidt

El proceso de Gram-Schmidt no es solo un algoritmo matemático, sino una herramienta con propósito claro: transformar un conjunto de vectores **linealmente independientes** (pero desordenados) en una base **ortonormal**, es decir:

- Vectores **perpendiculares** entre sí
- De **longitud unitaria**
- Que abarcan el mismo espacio que los vectores originales

---

### 🎯 ¿Qué problema resuelve?

Supón que tienes vectores inclinados, largos o cortos, que no están a 90° entre sí. Trabajar con ellos puede ser complejo:

| Situación                             | Problemas con vectores arbitrarios | Con base ortonormal |
|--------------------------------------|------------------------------------|---------------------|
| Proyecciones                         | Difíciles sin ajustes              | Basta con producto punto |
| Cambios de base                      | Requiere invertir matriz           | Solo se transpone |
| Rotaciones/transformaciones          | Cálculo complejo                   | Álgebra limpia |
| Análisis de datos (PCA, etc.)        | Datos mezclados, no separables     | Componentes limpios |

---

### 🧠 Intuición geométrica

Cada paso de Gram-Schmidt:

1. **Toma un vector nuevo**
2. **Le quita** lo que ya estaba contenido en los anteriores
3. **Normaliza** lo que queda

Así conseguimos vectores **mutuamente perpendiculares**, que **no se repiten entre sí** y **forman una base más "inteligible"** del espacio.

---

## 🧪 Proceso de Gram-Schmidt paso a paso

Dado un conjunto de vectores $\vec{v}_1, \vec{v}_2, ..., \vec{v}_n$ linealmente independientes:

1. **Primer vector**:
   $$
   \vec{e}_1 = \frac{\vec{v}_1}{\|\vec{v}_1\|}
   $$

2. **Segundo vector**:
   $$
   \vec{u}_2 = \vec{v}_2 - (\vec{v}_2 \cdot \vec{e}_1)\vec{e}_1
   $$
   $$
   \vec{e}_2 = \frac{\vec{u}_2}{\|\vec{u}_2\|}
   $$

3. **Tercer vector**:
   $$
   \vec{u}_3 = \vec{v}_3 - (\vec{v}_3 \cdot \vec{e}_1)\vec{e}_1 - (\vec{v}_3 \cdot \vec{e}_2)\vec{e}_2
   $$
   $$
   \vec{e}_3 = \frac{\vec{u}_3}{\|\vec{u}_3\|}
   $$

...y así sucesivamente hasta obtener una base ortonormal $\{\vec{e}_1, \vec{e}_2, ..., \vec{e}_n\}$

---

## 📌 Resultado final

- Los vectores $\vec{e}_i$ son ortonormales.
- Siguen abarcando el mismo espacio que los $\vec{v}_i$.
- Se pueden usar para simplificar transformaciones, proyecciones y análisis.

---
# 🧮 Introducción Visual a Eigenvectores y Eigenvalores 

## 🗣️ ¿Qué significa "Eigen"?

La palabra **"eigen"** proviene del alemán y significa **característico**. Así que cuando hablamos de un **problema de autovalores/autovectores**, nos referimos a encontrar las **propiedades características de una transformación lineal**.

---

## 🔄 Transformaciones Lineales

Las matrices pueden representar transformaciones como:

- Escalado
- Rotación
- Cizalladura (shear)

Imagina aplicar una de estas transformaciones a todos los vectores del plano. Un buen truco visual es dibujar un **cuadrado centrado en el origen** y ver cómo se deforma.

---
<div style="text-align: center;">
  <img src="999. IMG FOLDER/image-20.png" alt="Mi Imagen" width="500">
</div>
## 🎨 Ejemplo: Escalado vertical


Supongamos un escalado vertical por un factor de 2:

- El cuadrado original se convierte en un **rectángulo más alto**.
- Dibujamos tres vectores: uno horizontal, uno vertical, y uno diagonal.

**Resultado:**

- El vector **horizontal** no cambia.
- El vector **vertical** se duplica en longitud.
- El vector **diagonal** cambia de dirección y longitud.

👉 Los vectores **que permanecen en su misma línea de acción** se llaman **autovectores**.

- Si su longitud cambia por un factor $\lambda$, ese número se llama **autovalor**.

---

## 📌 Definiciones Clave

- **Autovector**: Vector que **mantiene su dirección** bajo una transformación.
- **Autovalor**: Escala por la que el autovector es **alargado o contraído**.

---

# 📐 Formulación Formal de eigenvectors y eigenvalues

## 🧠 Recordatorio Conceptual

- Un **autovector** de una matriz $A$ es un vector que **no cambia de dirección** al aplicar la transformación lineal $A$.
- Un **autovalor** es la **escala** por la cual se multiplica ese autovector.

---

## 🧾 Ecuación Característica

La expresión algebraica clave es:

$$
A \vec{x} = \lambda \vec{x}
$$

- $A$ es una matriz cuadrada $n \times n$.
- $\vec{x}$ es el autovector.
- $\lambda$ es el autovalor.

Esta ecuación dice: aplicar $A$ a $\vec{x}$ es lo mismo que escalar $\vec{x}$ por $\lambda$.

---

## 🔄 Reescritura para resolver

Llevamos todo a un lado:

$$
A\vec{x} - \lambda I \vec{x} = 0
$$

Factorizamos:

$$
(A - \lambda I) \vec{x} = 0
$$

- $I$ es la matriz identidad del mismo tamaño que $A$.

---

## ❗ Condición para soluciones no triviales

Queremos **evitar** $\vec{x} = 0$ (solución trivial).

Entonces:

$$
\det(A - \lambda I) = 0
$$

Esto es la **ecuación característica** que nos da los **autovalores**.

---

## 🧪 Ejemplo: Escalado vertical

Sea:

$$
A = \begin{bmatrix} 1 & 0 \\ 0 & 2 \end{bmatrix}
$$

Entonces:

$$
\det(A - \lambda I) = 
\det \begin{bmatrix}
1 - \lambda & 0 \\ 
0 & 2 - \lambda
\end{bmatrix}
= (1 - \lambda)(2 - \lambda)
$$

Entonces los autovalores son:

$$
\lambda = 1, \quad \lambda = 2
$$


### ✅ Para $\lambda = 1$:

**1. Calculamos:**

$$
A - \lambda I =
\begin{bmatrix}
1 & 0 \\
0 & 2
\end{bmatrix}
-
\begin{bmatrix}
1 & 0 \\
0 & 1
\end{bmatrix}
=
\begin{bmatrix}
0 & 0 \\
0 & 1
\end{bmatrix}
$$

**2. Sistema resultante:**

$$
\begin{bmatrix}
0 & 0 \\
0 & 1
\end{bmatrix}
\begin{bmatrix}
x_1 \\
x_2
\end{bmatrix}
=
\begin{bmatrix}
0 \\
0
\end{bmatrix}
$$

Ecuaciones:

- $0x_1 + 0x_2 = 0$ (trivial)
- $0x_1 + 1x_2 = 0 \Rightarrow x_2 = 0$

**3. Autovector asociado:**

$$
\vec{x}_1 =
\begin{bmatrix}
t \\
0
\end{bmatrix}
\quad t \in \mathbb{R}
$$


### ✅ Para $\lambda = 2$:

**1. Calculamos:**

$$
A - \lambda I =
\begin{bmatrix}
1 & 0 \\
0 & 2
\end{bmatrix}
-
\begin{bmatrix}
2 & 0 \\
0 & 2
\end{bmatrix}
=
\begin{bmatrix}
-1 & 0 \\
0 & 0
\end{bmatrix}
$$

**2. Sistema resultante:**

$$
\begin{bmatrix}
-1 & 0 \\
0 & 0
\end{bmatrix}
\begin{bmatrix}
x_1 \\
x_2
\end{bmatrix}
=
\begin{bmatrix}
0 \\
0
\end{bmatrix}
$$

Ecuaciones:

- $-x_1 = 0 \Rightarrow x_1 = 0$
- $0x_2 = 0$ (trivial)

**3. Autovector asociado:**

$$
\vec{x}_2 =
\begin{bmatrix}
0 \\
t
\end{bmatrix}
\quad t \in \mathbb{R}
$$

---

### 🧠 Nota sobre el parámetro $t$:

Usamos $t$ como **parámetro libre** porque cualquier múltiplo escalar de un autovector también es un autovector. Lo que importa es su **dirección**, no su magnitud.

---

## 🚫 Caso sin autovectores reales: Rotación 90°

Sea:

$$
A = \begin{bmatrix} 0 & -1 \\ 1 & 0 \end{bmatrix}
$$

Entonces:

$$
\det(A - \lambda I) = 
\det \begin{bmatrix}
-\lambda & -1 \\ 
1 & -\lambda
\end{bmatrix}
= \lambda^2 + 1 = 0
$$

Esto no tiene soluciones reales → ❌ no hay autovectores reales.

---

## 📌 Conclusión

- La ecuación $A\vec{x} = \lambda\vec{x}$ describe qué vectores mantienen su dirección tras una transformación.
- La ecuación característica $\det(A - \lambda I) = 0$ nos da los autovalores.
- Luego se encuentran los autovectores resolviendo $(A - \lambda I)\vec{x} = 0$.

---

## ✨ Diagonalización y Potencias de Matrices

### 🔁 Motivación: Múltiples Transformaciones
Cuando una transformación lineal $T$ se aplica muchas veces, calcular $T^n$ directamente puede ser computacionalmente costoso.

Ejemplo:
- $\vec{v}_1 = T\vec{v}_0$
- $\vec{v}_2 = T\vec{v}_1 = T^2 \vec{v}_0$
- $\vec{v}_n = T^n \vec{v}_0$

Si $n = 1{,}209{,}600$ (dos semanas en segundos), entonces $T^n$ es muy costoso de calcular directamente.

---

### 📐 Diagonalización: El Truco del Cambio de Base
La solución está en cambiar de base a la **base de autovectores** (eigen-basis), donde $T$ se convierte en una **matriz diagonal** $D$:

$$
T = C D C^{-1}
$$

Entonces:

$$
T^n = C D^n C^{-1}
$$

Esto permite calcular $T^n$ fácilmente, porque:

- $D$ es diagonal ⇒ $D^n$ se obtiene elevando cada término en la diagonal.
- $C$ contiene los autovectores como columnas.
- $D$ contiene los autovalores correspondientes en su diagonal.

---

### 🧪 Ejemplo con $T = \begin{bmatrix} 1 & 1 \\ 0 & 2 \end{bmatrix}$

Dada la matriz de transformación:

$$
T = \begin{bmatrix}
1 & 1 \\
0 & 2
\end{bmatrix}
$$
### Paso 1: Cálculo de autovalores

Resolvemos el polinomio característico:

$$
\det(T - \lambda I) = 
\begin{vmatrix}
1 - \lambda & 1 \\
0 & 2 - \lambda
\end{vmatrix}
= (1 - \lambda)(2 - \lambda) = 0
$$

Autovalores:
- $$\lambda_1 = 1$$
- $$\lambda_2 = 2$$


### Paso 2: Cálculo de eigenvectores

### Para $$\lambda = 1$$

Sustituimos en $( (T - \lambda I) \vec{x} = 0 )$:

$$
\begin{bmatrix}
0 & 1 \\
0 & 1
\end{bmatrix}
\begin{bmatrix}
x_1 \\
x_2
\end{bmatrix}
= 0 \Rightarrow x_2 = 0
$$

Autovector asociado:

$$
\vec{x}_1 = \begin{bmatrix}
t \\
0
\end{bmatrix}
\Rightarrow \text{Elegimos } t = 1 \Rightarrow \vec{x}_1 = \begin{bmatrix}
1 \\
0
\end{bmatrix}
$$

### Para $$\lambda = 2$$

$$
\begin{bmatrix}
-1 & 1 \\
0 & 0
\end{bmatrix}
\begin{bmatrix}
x_1 \\
x_2
\end{bmatrix}
= 0 \Rightarrow -x_1 + x_2 = 0 \Rightarrow x_1 = x_2
$$

Autovector asociado:

$$
\vec{x}_2 = \begin{bmatrix}
t \\
t
\end{bmatrix}
\Rightarrow \text{Elegimos } t = 1 \Rightarrow \vec{x}_2 = \begin{bmatrix}
1 \\
1
\end{bmatrix}
$$


### Paso 3: Matriz de cambio de base

Construimos la matriz \( C \) con los eigenvectores  como columnas:

$$
C = \begin{bmatrix}
1 & 1 \\
0 & 1
\end{bmatrix}
$$

### Paso 4: Inversa de \( C \)

Como  $\det(C) = 1$, usamos la fórmula para $( 2 \times 2$):

$$
C^{-1} = \begin{bmatrix}
1 & -1 \\
0 & 1
\end{bmatrix}
$$


Matriz diagonal $D$:

$$
D = \begin{bmatrix} 1 & 0 \\ 0 & 2 \end{bmatrix}
$$

Entonces:

$$
T^2 = C D^2 C^{-1} =
\begin{bmatrix} 1 & 1 \\ 0 & 1 \end{bmatrix}
\begin{bmatrix} 1 & 0 \\ 0 & 4 \end{bmatrix}
\begin{bmatrix} 1 & -1 \\ 0 & 1 \end{bmatrix} =
\begin{bmatrix} 1 & 3 \\ 0 & 4 \end{bmatrix}
$$

Aplicando $T^2$ a $\vec{v} = \begin{bmatrix}-1 \\ 1\end{bmatrix}$:

$$
T^2\vec{v} = \begin{bmatrix}1 & 3 \\ 0 & 4\end{bmatrix} \begin{bmatrix}-1 \\ 1\end{bmatrix} = \begin{bmatrix}2 \\ 4\end{bmatrix}
$$

¡Mismo resultado que con multiplicaciones directas!

---

### 💡 Conclusión
Diagonalizar $T$ simplifica el cálculo de $T^n$. Si entiendes:

- Autovalores
- Autovectores
- Cambio de base

...entonces puedes explotar esta poderosa técnica en álgebra lineal y machine learning.

> Esta técnica es especialmente útil en dinámica de sistemas, Markov Chains, simulaciones físicas, etc.
