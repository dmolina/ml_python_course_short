---
jupyter:
  jupytext:
    encoding: '# -*- coding: utf-8 -*-'
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.13.8
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# First steps

About the philosophy about programming in Python

```python
import this
```

Comments start with #

```python
# This is a comment
```

End of line set the end of a sentence (it does not required to use ; or other symbol)

```python
x = 5
y = 10
x
```

Indentation is key.

In Python the blocks of code are marked by the indentation. That also encourage an uniform style and readable that many found attractive in Python code.

```python
x = [1,2,3,4]
print (x)
```

```python
lower = []
upper = []
x = 5
for i in range(10):
    if i < x:
        lower.append(i)
    else:
        upper.append(i)

print('lower: ', lower)
print('upper: ', upper)
```

Some print examples:

```python
print("lower: "+lower)
print("upper: "+upper)
```

```python
print ("lower: "+str(lower))
print ("upper: "+str(upper))
```

```python
# Python 2 ! print "first value:", 1
# Python 3 !
print("first value:", 1)
```


# Objects and Variables Variables y Objetos

## Variables

In Python all variables are references.

```python
x = 4
```

x is a reference to the memory cell in which 4 is stored.

we do not need to declare the type of x, x can refer values of different types. Thus, it is said that Python is a dynamic type language. We can do:

```python
x = 1         # x is an integer
x = 'hello'   # now x is a string
x = [1, 2, 3] # now x is a list
```

This is another property that allows Python to be so easy to read and write.

#### ¡Warning!
Dynamic typing implies other consequences:

```python
x = [1, 2, 3]
y = x
```

```python
print('x: ', x)
print('y: ', y)
```

```python
x.append(4)
```

```python
print(y)
```

y reference to the same value than x, so when x changes the values of y changes also (both refer is the same data).

However, when we use '=' to change x:

```python
x = 'Bye'

print (x)
print(y)
```

The operator '=' make x to refer to a different memory value.

The original information referred by x is intact.

```python
x = 10
y = x
x += 5  # add 5 to x's value, and assign it to x
print("x =", x)
print("y =", y)
```

## Objects

Python is a Oriented Programming language, in Python everything is an object.


We have just see that in Python variables are essentially pointers, and that define a variable does not implies information about its type. However, Python is not a 'typeless' language.

Python has types, they are related to the values.

```python
x = 4
type(x)
```

```python
x = 'hello'
type(x)
```

```python
x = 3.14159
type(x)
```

# Identity Operators

## Identity operators: 'is' and 'is not'

Identity is not the same than equality.

```python
a = [1, 2, 3]
b = [1, 2, 3]
```

```python
a==b
```

```python
a is b
```

```python
a is not b
```

a y b store different objects.

Before we observed that Python are actually pointers. Operator 'is' check if they refer the same object.
So, the majority of times new students want to use 'is' they actually wanted to use '=='.

```python
a=b
```

```python
a is b
```

## Set operators
These operators check if a value is contained in a composed object.

```python
1 in [1,2,3]
```

```python
2 not in [1,2,3]
```

In object-oriented (OO) languages as Python, an object is an entity that stores data with their functionality (methods).

In Python everything is an object, each value has some information (attributes) and functionality (methods), they are accessed by dot syntax.

```python
L = [1, 2, 3]
L.append(100)
print(L)
```

Maybe it is strange but every value (not only composed) has its own methods.

```python
x = 4.5
print(x.real, "+", x.imag, 'i')
```

```python
x.is_integer()
```

```python
x = 4.0
x.is_integer()
```

Even the attributes and methods have its own information.

```python
type(x.is_integer)
```

# Control flow

## Conditionals

```python
x = -15

if x == 0:
    print(x, "is zero")
elif x > 0:
    print(x, "is positive")
elif x < 0:
    print(x, "is negative")
else:
    print(x, "I has not idea...")
```

The ':' and the identation are used to indicate the code blocks.


## Loops

### For

Some for examples in Python.

To print the elements of a list we can do:

```python
for N in [2, 3, 5, 7]:
    print(N, end=' ') # Python 3 only! print all on same line
    #print N ,         # Python 2 only! print all on same line
```


Especificamos la variable que queremos usar, la secuencia sobre la que queremos iterar, y usamos el operador 'in' para unirlos de forma intuitiva y legible. El objeto a la derecha del operador 'in' puede ser cualquier iterador de Python.

One of the iterators more widely used is the object range that generates a sequence of numbers:

```python
for i in range(10):
    print(i, end=", ")
```

```python
# range from 5 to 10
list(range(5, 10))
```

```python
# range from 0 to 10 by 2
list(range(-10, 10, 2))
```

### Bucle while

```python
i = 0
while i < 10:
    print(i, end=", ")
    i += 1
```

### break and continue

They are two useful sentences to use inside a loop:
- break: stop the loop.
- continue: ignore the current value of loop, go directly to next object.

Both can be used in for and while.

```python
for n in range(20):
    # If the rem n / 2 is 0, ignore the print
    if n % 2 == 0:
        continue
    print(n)
```

```python
# Finish a list with Fibonacci values
a, b = 0, 1
amax = 100
L = []

while True:
    (a, b) = (b, a + b)   # multiple assigment.
    if a > amax:
        break
    L.append(a)

print(L)
```

# Functions

They are defined with def.

```python
def fibonacci(N):
    L = []
    a, b = 0, 1
    while len(L) < N:
        a, b = b, a + b
        L.append(a)
    return L
```

```python
fibonacci(10)
x =2
```


We have defined a function called fibonacci with an argument N that returns a value, a list with N first values of Fibonacci sequence.

```python
fibonacci(10)
x = 2
x
```

A function can returns any object in Python.


A veces cuando definimos una función, hay ciertos valores que queremos que la función use la mayor parte del tiempo (o que creemos que son los que se van a usar mayoritariamente) pero también nos gustaría dar al usuario opciones/flexibilidad. En esos casos, podemos usar valores por defecto para los argumentos de la función.

```python
def fibonacci(N, a=0, b=1):
    L = []
    while len(L) < N:
        a, b = b, a + b
        L.append(a)
    return L
```

```python
fibonacci(10)
```

```python
fibonacci(10, 0, 1)
```

```python
fibonacci(10, 0, 2)
```

```python
fibonacci(10, b=3, a=1)
```

# Modules and packages

For loading and using existing modules we use the sentence import. 

We can import direclty one module. For using its functions and/or variable, we have to include the namespace.

```python
import math
math.cos(math.pi)
```

We can import using an alias, very common.

```python
import numpy as np
np.cos(np.pi)
```

If we are only interested in some elements, we can indicate and avoid the namespace:

```python
from math import cos, pi
cos(pi)
```

We can import all content of a package (not recommended)

```python
from math import *
sin(pi) ** 2 + cos(pi) ** 2
```

You should be careful with this options. The problem is to overload existing functions not wanted. For instance:

```python
help(sum)
```

```python
print(str(range(5)))
sum(range(5), -1)
```

```python
from numpy import *
help (sum)
sum(range(5), -1)
```

In the first case, we sum range(5) starting with -1; in the second one, we sum range(5) using the last axis (indicated by -1).


# External Libraries

One of the most useful features of Python, specially in science data, is their ecosystem, there are libraries for almost anything, and they can be easily installed. These libraries can be imported as built-in modules, but first we have to install them in our system. The standard registry is the Python Package Index (PyPI), we can check the available libraries in http://pypi.python.org/.
To install one of these modules/libraries, we will use the program pip. For instance, to install the library pandas:

$ pip install pandas

#### If you use ANACONDA:
With Anaconda, the command will be:

$ conda install pandas

Or through the user interface: Environments option.
