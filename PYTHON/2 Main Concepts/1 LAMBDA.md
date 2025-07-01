#Python 

## 🌟 ¿Qué son las funciones lambda?

Las funciones lambda en Python son funciones anónimas y de una sola línea que se utilizan para definir funciones rápidas y ligeras sin la necesidad de usar `def`.
- Se utilizan principalmente cuando se necesita una función pequeña y temporal.

- Son útiles en operaciones de mapeo, filtrado y reducción.
### 📌 Sintaxis

```python 
lambda argumentos: expresión
```

- **lambda**: Palabra clave que indica el inicio de una función lambda.
   
- **argumentos**: Lista de argumentos separados por comas.

- **expresión**: Una única expresión cuyo resultado es devuelto.

#### Ejemplo básico


```python 
doble = lambda x: x * 2 print(doble(5))  # Salida: 10`
```
---

## ✨ Características de las funciones lambda

1. **Anónimas**: No tienen un nombre explícito.
    
2. **Concisas**: Se escriben en una sola línea.
    
3. **Funciones de una sola expresión**: No pueden contener múltiples sentencias.
    
4. **Retorno implícito**: Siempre devuelven el valor de la expresión evaluada.
    

---

## 🛠️ Uso común: Funciones Lambda como argumentos

Las funciones lambda son comunes en funciones como `map()`, `filter()` y `reduce()`.

### 🔄 Ejemplo con `map()`

Aplica una función a cada elemento de una lista.

```python 
numeros = [1, 2, 3, 4] 
cuadrados = list(map(lambda x: x**2, numeros)) 
print(cuadrados)  
# Salida: [1, 4, 9, 16]
```

### 🔍 Ejemplo con `filter()`

Filtra elementos que cumplen una condición.

```python 
numeros = [1, 2, 3, 4] 
pares = list(filter(lambda x: x % 2 == 0, numeros)) 
print(pares)  
# Salida: [2, 4]
```

### ➕ Ejemplo con `reduce()`

Reduce una lista a un solo valor acumulando los resultados.

```python
from functools import reduce 
numeros = [1, 2, 3, 4] 
suma = reduce(lambda x, y: x + y, numeros) 
print(suma)  
# Salida: 10
```

---

## 🚩 Limitaciones de las funciones lambda

- **Sólo una expresión**: No pueden contener múltiples declaraciones o expresiones complejas.
    
- **Difíciles de depurar**: El uso excesivo puede hacer el código ilegible.
    
- **Limitación de funcionalidades**: No pueden contener bucles ni múltiples instrucciones.


---
## 💡 Buenas prácticas

1. Úsalas para operaciones cortas y sencillas.
    
2. Prefiere funciones normales (`def`) cuando el cuerpo es complejo o extenso.
    
3. Mantén el código limpio y legible, evitando lambdas anidadas.