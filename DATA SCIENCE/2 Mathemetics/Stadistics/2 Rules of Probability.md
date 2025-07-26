#Math #Stadistics 
[[3 Intro to Probability Distributions]]
[[3 EDA ADV Variance and Standard Deviation]]
## 🌟 Introducción

Las reglas de probabilidad son fundamentales en el análisis estadístico y nos permiten calcular la probabilidad de eventos bajo diferentes condiciones. Comprender estas reglas facilita la resolución de problemas en ciencias, matemáticas y situaciones cotidianas.

### 📝 Objetivos

1. Entender las reglas básicas de la probabilidad.
    
2. Aplicar conceptos como unión, intersección y complemento.
    
3. Diferenciar entre eventos independientes y dependientes.
    
4. Utilizar reglas específicas como la suma, el producto y el teorema de Bayes.
    

---

## 🔗 Union, Intersection, and Complement

- **Unión (A ∪ B):** La probabilidad de que ocurra al menos uno de los eventos A o B.
    
- **Intersección (A ∩ B):** La probabilidad de que ocurran ambos eventos simultáneamente.
    
- **Complemento (Aᶜ):** La probabilidad de que no ocurra el evento A.
    

**Ejemplo en Python:**

```Python
A = {1, 2, 3}
B = {2, 4, 6}
union = A.union(B)
intersection = A.intersection(B)
complement = set(range(1, 7)) - A
print(f"Unión: {union}, Intersección: {intersection}, Complemento: {complement}")
```

---

## 🔄 Independence and Dependence

- **Independencia:** Dos eventos son independientes si la ocurrencia de uno no afecta la probabilidad del otro.
    
- **Dependencia:** Dos eventos son dependientes si uno influye en el resultado del otro.
    

**Ejemplo:** Lanzar una moneda y tirar un dado son eventos independientes.

---

## ❌ Mutually Exclusive Events

- **Eventos mutuamente excluyentes:** No pueden ocurrir simultáneamente (A ∩ B = ∅).
    
- **Ejemplo:** Sacar cara y cruz en el mismo lanzamiento de moneda.
    

---

## ➕ Addition Rule

- **Regla de la suma:** Para eventos A y B:
    
    - Si son mutuamente excluyentes: $P(A \cup B) = P(A) + P(B)$
        
    - Si no son excluyentes: $P(A \cup B) = P(A) + P(B) - P(A \cap B)$
        

**Ejemplo en Python:**

```Python
P_A = 0.3
P_B = 0.5
P_AB = 0.1
P_union = P_A + P_B - P_AB
print(f"Probabilidad de A ∪ B: {P_union}")
```

---

## 🧩 Conditional Probability

- **Probabilidad condicional:** La probabilidad de que ocurra A dado que B ya ocurrió.
    
- **Fórmula:** $P(A|B) = \frac{P(A \cap B)}{P(B)}$
    

**Ejemplo en Python:**

```Python
P_A_given_B = P_AB / P_B
print(f"P(A|B): {P_A_given_B}")
```

### Profundización

El análisis de probabilidad condicional es fundamental en modelos de inferencia estadística y aprendizaje automático.

---

## ✖️ Multiplication Rule

- **Regla del producto:** Para eventos independientes: $P(A \cap B) = P(A) \times P(B)$
    
- Para eventos dependientes: $P(A \cap B) = P(A) \times P(B|A)$
    

---

## 🔁 Bayes' Theorem

- **Teorema de Bayes:** Relaciona la probabilidad condicional inversa.
    
- **Fórmula:** $P(A|B) = \frac{P(B|A) \times P(A)}{P(B)}$
    

**Ejemplo en Python:**

```Python
P_B_given_A = 0.8
P_A = 0.2
P_B = 0.5
P_A_given_B = (P_B_given_A * P_A) / P_B
print(f"P(A|B) usando Bayes: {P_A_given_B}")
```

---

## 📚 Buenas prácticas

1. Utilizar diagramas de Venn para visualizar uniones e intersecciones.
    
2. Aplicar fórmulas con cuidado, diferenciando eventos independientes y dependientes.
    
3. Realizar simulaciones para verificar resultados teóricos.
    
