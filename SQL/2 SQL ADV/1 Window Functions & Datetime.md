#SQL
[[2 Funnel Analysis]]
# 🧮 SQL: Window Functions and Core Operations

## 🌟 Introducción

SQL permite realizar cálculos avanzados sobre conjuntos de datos mediante **funciones de ventana**, así como operaciones matemáticas, agregaciones y manejo de fechas. Estas herramientas son clave para realizar análisis potentes sin salir de la base de datos.

---

## 🔍 Window Functions

Las funciones de ventana permiten realizar cálculos acumulativos y comparativos sin agrupar los datos.

### 🧱 Sintaxis general

```SQL
SELECT column,
       FUNCTION(...) OVER (
           PARTITION BY column
           ORDER BY column
           ROWS BETWEEN ...
       )
FROM table;
```

### 🔸 OVER y PARTITION

- `OVER`: define el marco de análisis.
    
- `PARTITION BY`: reinicia el cálculo para cada grupo.
    

### 🔢 Funciones de ventana comunes

#### 1. **ROW_NUMBER()**

Asigna un número único a cada fila por grupo.

```SQL
ROW_NUMBER() OVER (PARTITION BY department ORDER BY salary DESC)
```

#### 2. **RANK() y DENSE_RANK()**

Asigna un ranking con (o sin) saltos para empates.

```SQL
RANK() OVER (PARTITION BY team ORDER BY points DESC)
```

#### 3. **NTILE(n)**

Divide el conjunto en `n` partes iguales (cuartiles, deciles, etc.).

```SQL
NTILE(4) OVER (ORDER BY sales)
```

#### 4. **LAG() y LEAD()**

Accede a valores anteriores o posteriores.

```mysql
LAG(salary, 1) OVER (PARTITION BY dept ORDER BY hire_date)
LEAD(salary, 1) OVER (PARTITION BY dept ORDER BY hire_date)
```

#### 5. **FIRST_VALUE() y LAST_VALUE()**

Devuelve el primer o último valor dentro del marco.

```SQL
FIRST_VALUE(salary) OVER (PARTITION BY dept ORDER BY hire_date)
```

#### 6. **RANGE BETWEEN**

Controla el rango de filas dentro del marco.

```SQL
SUM(sales) OVER (ORDER BY date ROWS BETWEEN 1 PRECEDING AND CURRENT ROW)
```

---

## ➗ Math & Aggregation Functions

### Aritmética básica

```SQL
SELECT salary * 0.10 AS bonus,
       price - discount AS final_price
FROM products;
```

### Funciones estadísticas

```mySQL
AVG(salary), MAX(score), MIN(score), COUNT(*), SUM(amount)
```

### CAST (conversión de tipos)

```mySQL
CAST(score AS FLOAT)
CAST(date AS TEXT)
```

- **Transformar texto numérico:**
    

```mySQL
CAST('3.14 es pi' AS REAL)  -- Resultado: 3.14
```

- **Evitar división entera:**
    

```mySQL
SELECT 3 / 2;             -- Resultado: 1
SELECT CAST(3 AS REAL) / 2;  -- Resultado: 1.5
```

---

## ⏱️ Date and Time Functions

### Tipos comunes

- `DATE` – Solo fecha
    
- `TIME` – Solo hora
    
- `DATETIME` – Fecha y hora
    

### Obtener la fecha actual

```mySQL
SELECT CURRENT_DATE, CURRENT_TIME, CURRENT_TIMESTAMP;
```

### STRFTIME en SQLite

Formateo de fechas:

```mySQL
SELECT STRFTIME('%Y-%m', order_date) AS year_month
FROM sales;
```

|Código|Significado|
|---|---|
|%Y|Año (e.g., 2023)|
|%m|Mes|
|%d|Día|
|%H|Hora|
|%M|Minuto|
|%S|Segundo|

---

## 📚 Buenas prácticas

- Usa `PARTITION BY` cuando necesites acumulados por grupo.
    
- Prefiere `CAST` explícito al comparar tipos distintos.
    
- Usa `STRFTIME` para agrupar o filtrar por períodos.