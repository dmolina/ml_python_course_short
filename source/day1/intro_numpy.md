---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.12.0
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# Numpy Introduction


<img src="https://upload.wikimedia.org/wikipedia/commons/1/1a/NumPy_logo.svg" width="30%">

Numpy is a library for Python to create and operate multidimensional matrix.

It is simple to use. First thing is to convert a vector or list to a numpy array.

```python
import numpy as np
```

```python
np.cos([2.4, 4.5])
```

Create a 1-d array

```python
V = np.array([1,2,3,4,5], dtype=float)
V
```

```python
[1, 2, 3, 4, 5]
```

```python
type(V), type([1, 2, 3])
```

You can access each element similarly to a list.

```python
V
```

```python
# V.append([1,2])
V = np.concatenate((V, [1,2]))
```

Even several values at the same time.

```python
V
```

```python
V[2]
```

```python
V[[1, 3, 4]]  # You cannot do that with a list
```

```python
V[1:3] # Work with list
```

```python
V[2:5]
```

Create a 1-d array from a list:

```python
L = [1, 2, 3, 4, 5, 6]

V = np.array(L)
len(V)
```

```python
M1 = [[1,2,3],[4,5,6]]
M2 = np.array(M1)
M2
```

```python
print(len(V), len(M2))
```

```python
M2.shape, V.shape
```

```python
print("Rows: ", M2.shape[0])
```

```python
print("Columns: ", M2.shape[1])
```

```python
M2.sum()
```

```python
M2
```

# Type of array

At difference of lists, an numpy array or matrix can only store values of the same time.


```python
np.array([1, 2, "hello"])
```

Usually from numeric type.

```python
np.array([1, 2, 3.5])
```

You can indicate the type of the element, with dtype  {int, float32, float64, bool, ...}.

```python
A=np.array(L, dtype=np.float32)
A
```

```python
np.array([1,0, 1], dtype=int)
```

```python
np.array([1, 0, 6], dtype=bool)
```

```python
np.array([True, False])
```

```python
A.tolist()
```

# Advantages from numpy arrays

The main advantage is the processing time, python lists are very slow, and numpy operations are efficiently implemented in C++.

```python
def norm1(vector):
    if len(vector)==0:
        return []
    
    maxv = minv = vector[0]
    
    for val in vector:
        if val > maxv:
            maxv = val
        elif val < minv:
            minv = val
    
    norm = len(vector)*[None]
    
    for i, val in enumerate(vector):
        norm[i] = (val-minv)/(maxv-minv)
        
    return norm    
```

```python
# Creo un vector aleatorio
v = np.random.rand(5_000_000)*10-5
v_list = v.tolist()
```

```python
v
```

```python
%time sal1=norm1(v)
```

```python
np.min(sal1)
```

Implemented with numpy:

```python
def norm2(vector):
    minv = vector.min()
    maxv = vector.max()
    return (vector-minv)/(maxv-minv)
```

```python
%time sal2=norm2(v)
```

```python
assert np.all(sal1 == sal2)
```

```python
np.where([True, False], [1, 2], [3, 4])
```

# Vectorial Operations

You can sum, prod, divide, ..., an array with a scalar. In this case the operation is carried out element by element.

```python
A/5
```

```python
np.array([1.0,2,3,4,5]) / 5
```

You can also process arrays with the same length, and the operation is done element by element.

```python
L
```

```python
M = np.array([2, 3, 4, 5, 6, 7])

P = L * M
P
```

There are also operators that work with all elements of the array.

```python
P.sum()
```

```python
# Shuffl
np.random.shuffle(P)
```

```python
print(P)
```

```python
print(P.min(), P.max())
```

```python
print(P.min(), np.argmin(P), P[np.argmin(P)])
```

There are also specific operations with arrays, like the scalar product.

```python
np.dot(L, M)
```

# 2-D Array

```python
A = np.array([[1,2,3],[4,5,6],[7,8,9],[8,7,6]])
A
```

shape returns the tuple (rows, columns)

```python
A.shape
```

```python
A[0,1]
```

```python
A[2,1]
```

```python
A[[0,1], :]
```

You can access to specific row or column:

```python
A[1,:]
```

```python
A[:,1]
```

You can also return a subarray.

```python
A[1:,1:]
```

```python
A[1:3,1:]
```

```python
A[1:,0:2]
```

You can also process values that follow a certain criterion.

```python
V = np.array([1, 2, 3, 4, 5, 6])
```

```python
V > 3
```

```python
V[V > 3]
```

```python
Ind = (V % 2 == 0) & (V < 6)
print(Ind)
V[Ind]
```

# Creation

Numpy has several generation method using random values.

```python
np.random.randint(10)
```

```python
np.random.rand(10)
```

```python
np.random.randint(-10, 10, 5)
```

There are more useful methods.

```python
np.zeros(10)
```

```python
np.ones(10)
```

```python
3*np.ones(10)
```

```python
np.arange(10)
```

# Numpy Exercise


1. Create a function that calculate the Euclideana distance between two arrays.

```python
def disteuc(vector1, vector2):
    dist = None
    # Put here your code
    return dist
```

```python
def test():
    assert 5 == disteuc(np.zeros(25), np.ones(25))
    assert 0 == disteuc(np.arange(30), np.arange(30))
    
test()
```
