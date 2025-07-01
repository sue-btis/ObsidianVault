#SQL 
# 🧭 SQL Funnel Analysis: From Events to Conversion

## 🌟 Introducción

Un análisis de embudo (**funnel analysis**) te ayuda a entender cuántos usuarios completan cada paso de un proceso (por ejemplo: visitar sitio → agregar al carrito → compra). En SQL, se puede construir embudos a partir de **una sola tabla de eventos** o **múltiples tablas separadas**.

---

## 1️⃣ Funnel desde una sola tabla

### 📦 Supuesto

Una tabla `events` con las columnas:

- `user_id`
    
- `event_type` (p. ej., 'visit', 'add_to_cart', 'purchase')
    
- `event_time`
    

### 🛠 Paso a paso

```
WITH event_steps AS (
  SELECT user_id,
         MAX(CASE WHEN event_type = 'visit' THEN 1 ELSE 0 END) AS visited,
         MAX(CASE WHEN event_type = 'add_to_cart' THEN 1 ELSE 0 END) AS added,
         MAX(CASE WHEN event_type = 'purchase' THEN 1 ELSE 0 END) AS purchased
  FROM events
  GROUP BY user_id
)
SELECT
  COUNT(*) AS total_users,
  SUM(visited) AS visitors,
  SUM(added) AS add_to_cart,
  SUM(purchased) AS purchases
FROM event_steps;
```

### 🎯 Resultado

Una sola fila que resume cuántos usuarios completaron cada etapa.

---

## 2️⃣ Funnel desde múltiples tablas

### 📦 Supuesto

Tablas separadas:

- `page_visits(user_id, timestamp)`
    
- `add_to_cart(user_id, timestamp)`
    
- `purchases(user_id, timestamp)`
    

### 🛠 Paso a paso

```
WITH funnel AS (
  SELECT pv.user_id,
         CASE WHEN ac.user_id IS NOT NULL THEN 1 ELSE 0 END AS added,
         CASE WHEN pu.user_id IS NOT NULL THEN 1 ELSE 0 END AS purchased
  FROM page_visits pv
  LEFT JOIN add_to_cart ac ON pv.user_id = ac.user_id
  LEFT JOIN purchases pu ON pv.user_id = pu.user_id
)
SELECT
  COUNT(*) AS total_visits,
  SUM(added) AS total_add_to_cart,
  SUM(purchased) AS total_purchases
FROM funnel;
```

### 🔁 Alternativa con intersección

También puedes usar `INTERSECT` para ver usuarios comunes entre pasos:

```
SELECT COUNT(*) FROM (
  SELECT user_id FROM page_visits
  INTERSECT
  SELECT user_id FROM add_to_cart
  INTERSECT
  SELECT user_id FROM purchases
) AS funnel_completed;
```

---

## 📊 Mejores prácticas

- Asegúrate de usar `LEFT JOIN` para no excluir usuarios que no llegaron al paso final.
    
- Usa `WITH` para claridad y reutilización.
    
- Ordena por `event_time` si necesitas seguir secuencia exacta.