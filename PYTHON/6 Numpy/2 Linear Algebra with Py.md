#Python 
[[2 Linear Algebra]]
# 🧠 NumPy Essentials for Matrix Operations in Linear Algebra

```python
import numpy as np
```

## 📌 1. Create Arrays and Matrices

```python
# Vector (1D array)
v = np.array([1, 2, 3])

# Matrix (2D array)
A = np.array([[1, 2], [3, 4]])

# Identity matrix
I = np.eye(3)  # 3x3 identity matrix

# Zero matrix
Z = np.zeros((2, 3))  # 2x3 matrix of zeros
```

## 🔗 2. Constructing Matrices by Columns

```python
# Column stack vectors into a matrix
col1 = np.array([1, 0])
col2 = np.array([0, 1])
M = np.column_stack((col1, col2))  # Same as identity
```

## 🔄 3. Transpose

```python
A_T = A.T  # Transpose of A
```

## 🧮 4. Elementwise Operations

```python
B = np.array([[5, 6], [7, 8]])

# Addition
C_add = A + B

# Subtraction
C_sub = B - A

# Elementwise multiplication
C_mul = A * B
```

## 🔢 5. Matrix Multiplication

```python
# Dot product
C_dot = np.dot(A, B)

# Matrix multiplication
C_matmul = np.matmul(A, B)

# Shorthand @ operator (Python 3.5+)
C_at = A @ B
```

## 📐 6. Linear Algebra (np.linalg)

```python
# Inverse
A_inv = np.linalg.inv(A)

# Determinant
det_A = np.linalg.det(A)

# Eigenvalues and eigenvectors
eig_vals, eig_vecs = np.linalg.eig(A)

# Solving Ax = b
b = np.array([1, 0])
x = np.linalg.solve(A, b)
```

---

📌 **Tip:** Always verify matrix dimensions when multiplying! For `A @ B`, the inner dimensions must match.
