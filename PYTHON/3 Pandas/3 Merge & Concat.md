#Python #Pandas
# Pandas: Merge and Concatenate

## 🌟 Introducción

Pandas proporciona dos formas principales de combinar datos de múltiples DataFrames: **merge** y **concatenate**. Estas funciones son esenciales para manipular conjuntos de datos grandes y estructurados.

---

## 🔗 Merge

El método `merge()` en Pandas se utiliza para combinar dos DataFrames en función de una o más columnas comunes. Similar a una **join** en SQL.

### 📥 Sintaxis

```python
pd.merge(df1, df2, on='columna_común', how='tipo_de_join')
```

### 🌟 Tipos de Join

- **inner** (por defecto): Devuelve solo las filas que tienen coincidencia en ambos DataFrames.
    
- **outer**: Devuelve todas las filas, combinando los valores donde coincidan.
    
- **left**: Devuelve todas las filas del DataFrame izquierdo con los valores coincidentes del derecho.
    
- **right**: Devuelve todas las filas del DataFrame derecho con los valores coincidentes del izquierdo.
    

### 💡 Ejemplo básico

```python
import pandas as pd

df1 = pd.DataFrame({'ID': [1, 2, 3], 'Nombre': ['Ana', 'Luis', 'Juan']})
df2 = pd.DataFrame({'ID': [2, 3, 4], 'Salario': [5000, 6000, 7000]})

resultado = pd.merge(df1, df2, on='ID', how='inner')
print(resultado)
```

**Salida:**

```python
   ID Nombre  Salario
0   2  Luis    5000
1   3  Juan    6000
```

### 🚩 Merge con múltiples columnas

```python
resultado = pd.merge(df1, df2, on=['ID', 'Nombre'], how='outer')
print(resultado)
```

### 📝 Usando Rename en Merge

A veces, las columnas a combinar tienen nombres diferentes en cada DataFrame. En estos casos, se usa el parámetro `left_on` y `right_on` para especificar las columnas correspondientes.

#### Ejemplo con rename:

```python
df1 = pd.DataFrame({'EmpleadoID': [1, 2], 'Nombre': ['Ana', 'Luis']})
df2 = pd.DataFrame({'ID': [1, 2], 'Salario': [5000, 6000]})

resultado = pd.merge(df1, df2, left_on='EmpleadoID', right_on='ID')
print(resultado)
```

**Salida:**

```python
   EmpleadoID Nombre  ID  Salario
0          1    Ana   1    5000
1          2   Luis   2    6000
```

#### Renombrar columnas después del merge:

```python
resultado = resultado.rename(columns={'EmpleadoID': 'ID_Empleado'})
print(resultado)
```

---

## 🔄 Concatenate

El método `concat()` se utiliza para combinar DataFrames ya sea en filas (por defecto) o en columnas.

### 📥 Sintaxis

```python
pd.concat([df1, df2], axis=0)
```

- **axis=0**: Combina filas (apilar hacia abajo).
    
- **axis=1**: Combina columnas (lado a lado).
    

### 💡 Ejemplo básico

```python
df1 = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
df2 = pd.DataFrame({'A': [5, 6], 'B': [7, 8]})

# Concatenar en filas
df_concat = pd.concat([df1, df2], axis=0)
print(df_concat)
```

**Salida:**

```python
   A  B
0  1  3
1  2  4
0  5  7
1  6  8
```

### 📂 Concatenar en columnas

```python
df_concat_col = pd.concat([df1, df2], axis=1)
print(df_concat_col)
```

**Salida:**

```
   A  B  A  B
0  1  3  5  7
1  2  4  6  8
```

### ⚠️ Manejo de índices

Para evitar conflictos de índice, usa el parámetro `ignore_index=True`:

```python
concat_reset = pd.concat([df1, df2], ignore_index=True)
print(concat_reset)
```

---

## 🚩 Buenas prácticas

1. Utiliza `merge()` para combinar datos relacionales (similar a SQL).
    
2. Prefiere `concat()` cuando los datos ya están alineados por filas o columnas.
    
3. Usa `rename()` para armonizar nombres después de combinar.
    
4. Usa `ignore_index=True` para evitar duplicación de índices al concatenar.
    

---

## 🔗 Referencias

- Pandas Merge Documentation
    
- Pandas Concat Documentation