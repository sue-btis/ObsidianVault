#CHEATSHEET
# 🧠 SQL Quick Reference

## 1. 🧩 Finding Data Queries

### 🔹 SELECT
```sql
SELECT * FROM table_name;
SELECT column1, column2 FROM table_name;
```

### 🔹 DISTINCT
```sql
SELECT DISTINCT column_name FROM table_name;
```

### 🔹 WHERE
```sql
SELECT * FROM table_name WHERE condition;
SELECT * FROM table_name WHERE condition1 AND condition2;
SELECT * FROM table_name WHERE condition1 OR condition2;
SELECT * FROM table_name WHERE NOT condition;
SELECT * FROM table_name WHERE condition1 AND (condition2 OR condition3);
SELECT * FROM table_name WHERE EXISTS (
  SELECT column_name FROM table_name WHERE condition
);
```

### 🔹 ORDER BY
```sql
SELECT * FROM table_name ORDER BY column ASC;
SELECT * FROM table_name ORDER BY column DESC;
SELECT * FROM table_name ORDER BY column1 ASC, column2 DESC;
```

### 🔹 SELECT TOP / LIMIT
```sql
-- SQL Server / MS Access
SELECT TOP number column_names FROM table_name;
SELECT TOP percent column_names FROM table_name;

-- MySQL
SELECT column_names FROM table_name LIMIT offset, count;
```

### 🔹 LIKE
```sql
SELECT column_name FROM table_name WHERE column_name LIKE pattern;
-- Examples
LIKE 'a%'       -- starts with a
LIKE '%a'       -- ends with a
LIKE '%or%'     -- contains 'or'
LIKE '_r%'      -- 'r' is second character
LIKE 'a_%_%'    -- starts with 'a' and at least 3 characters
LIKE '[a-c]%'   -- starts with a, b, or c
```

### 🔹 IN
```sql
SELECT column_name FROM table_name WHERE column_name IN (value1, value2, ...);
SELECT column_name FROM table_name WHERE column_name IN (SELECT ...);
```

### 🔹 BETWEEN
```sql
SELECT column_name FROM table_name WHERE column_name BETWEEN value1 AND value2;
SELECT * FROM Products WHERE column_name BETWEEN #01/07/1999# AND #03/12/1999#;
```

### 🔹 NULL
```sql
SELECT * FROM table_name WHERE column_name IS NULL;
SELECT * FROM table_name WHERE column_name IS NOT NULL;
```

### 🔹 AS (Aliases)
```sql
SELECT column_name AS alias_name FROM table_name;
SELECT column1 + ', ' + column2 AS full_name FROM table_name;
```

### 🔹 UNION / UNION ALL
```sql
SELECT column_names FROM table1
UNION
SELECT column_names FROM table2;

-- Include duplicates
UNION ALL
```

### 🔹 INTERSECT
```sql
SELECT column_names FROM table1
INTERSECT
SELECT column_names FROM table2;
```

### 🔹 EXCEPT
```sql
SELECT column_names FROM table1
EXCEPT
SELECT column_names FROM table2;
```

### 🔹 ANY / ALL
```sql
SELECT column_names FROM table1
WHERE column_name operator (ANY | ALL)
  (SELECT column_name FROM table_name WHERE condition);
```

### 🔹 GROUP BY & HAVING
```sql
SELECT column1, COUNT(column2)
FROM table_name
WHERE condition
GROUP BY column1
HAVING COUNT(column2) > 5;
```

### 🔹 WITH (Common Table Expressions)
```sql
WITH RECURSIVE cte AS (
  SELECT * FROM categories WHERE id = 1
  UNION ALL
  SELECT c.* FROM categories c
  JOIN cte ON c.parent_category_id = cte.id
)
SELECT * FROM cte;
```

## 2. 🛠️ Data Modification Queries

### 🔹 INSERT INTO
```sql
INSERT INTO table_name (column1, column2) VALUES (value1, value2);
INSERT INTO table_name VALUES (value1, value2, ...);
```

### 🔹 UPDATE
```sql
UPDATE table_name SET column1 = value1 WHERE condition;
```

### 🔹 DELETE
```sql
DELETE FROM table_name WHERE condition;
```

## 3. 📊 Reporting Queries

### 🔹 COUNT, MIN, MAX
```sql
SELECT COUNT(DISTINCT column_name) FROM table_name;
SELECT MIN(column_name), MAX(column_name) FROM table_name WHERE condition;
```

### 🔹 AVG, SUM
```sql
SELECT AVG(column_name) FROM table_name;
SELECT SUM(column_name) FROM table_name;
```

## 4. 🔗 Join Queries

### 🔹 INNER JOIN
```sql
SELECT t1.column, t2.column
FROM table1 t1
INNER JOIN table2 t2 ON t1.id = t2.fk_id;
```

### 🔹 LEFT JOIN
```sql
SELECT * FROM table1
LEFT JOIN table2 ON table1.id = table2.fk_id;
```

### 🔹 RIGHT JOIN
```sql
SELECT * FROM table1
RIGHT JOIN table2 ON table1.id = table2.fk_id;
```

### 🔹 FULL OUTER JOIN
```sql
SELECT * FROM table1
FULL OUTER JOIN table2 ON table1.id = table2.fk_id;
```

### 🔹 SELF JOIN
```sql
SELECT A.column, B.column
FROM table_name A, table_name B
WHERE A.id = B.parent_id;
```

## 5. 👁️ View Queries

### 🔹 CREATE VIEW
```sql
CREATE VIEW view_name AS
SELECT column1, column2 FROM table_name WHERE condition;
```

### 🔹 SELECT FROM VIEW
```sql
SELECT * FROM view_name;
```

### 🔹 DROP VIEW
```sql
DROP VIEW view_name;
```

## 6. 🧱 Altering Table Queries

### 🔹 ADD Column
```sql
ALTER TABLE table_name ADD column_name datatype;
```

### 🔹 MODIFY Column
```sql
ALTER TABLE table_name MODIFY column_name datatype;
```

### 🔹 DROP Column
```sql
ALTER TABLE table_name DROP COLUMN column_name;
```

## 7. 🏗️ Creating Table Queries

```sql
CREATE TABLE table_name (
  column1 datatype,
  column2 datatype,
  column3 datatype
);
```
