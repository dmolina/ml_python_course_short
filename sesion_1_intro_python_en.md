---
jupyter:
  jupytext:
    encoding: "\\# -\\*- coding: utf-8 -\\*-"
    text_representation:
      extension: .md
      format_name: pandoc
      format_version: 2.17.1.1
      jupytext_version: 1.13.8
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
  nbformat: 4
  nbformat_minor: 5
---

::: {#6d11d1aa .cell .markdown}
# First steps

Sobre la filosofía de programación en Python\...
:::

::: {#f8efacfd .cell .code}
``` python
import this
```
:::

::: {#0786a85a .cell .markdown}
Los comentarios se marcan con \#
:::

::: {#39e8a875 .cell .code}
``` python
# Esto es un comentario
```
:::

::: {#a5e99069 .cell .markdown}
El final de línea indica el fin de una sentencia (no es necesario marcarlo con ; ni con ningún otro carácter)
:::

::: {#116d85ec .cell .code}
``` python
x = 5
y = 10
x
```
:::

::: {#d404e25a .cell .markdown}
La indentación (indentation) es clave.

En Python los bloques de código se indican por indentación. El uso de indentación ayuda a reforzar el estilo uniforme y legible que muchos encuentran atractivo en el código Python.
:::

::: {#fc87104a .cell .code}
``` python
x = [1,2,3,4]
print (x)
```
:::

::: {#9d384eb4 .cell .code}
``` python
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
:::

::: {#ee01ca98 .cell .markdown}
Unos apuntes sobre print:
:::

::: {#cc55e622 .cell .code}
``` python
print("lower: "+lower)
print("upper: "+upper)
```
:::

::: {#569f9d47 .cell .code}
``` python
print ("lower: "+str(lower))
print ("upper: "+str(upper))
```
:::

::: {#6182d630 .cell .code}
``` python
# Python 2 ! print "first value:", 1
# Python 3 !
print("first value:", 1)
```
:::

::: {#09a88524 .cell .markdown}
# Variables y Objetos

### Variables

En Python las variables son punteros.
:::

::: {#441186ac .cell .code}
``` python
x = 4
```
:::

::: {#8fa006ac .cell .markdown}
x es un puntero que apunta a un espacio de memoria donde se almacena el valor 4.

Una consecuencia de esto es que no tenemos que declarar el tipo de x, es un puntero que puede apuntar a información/datos de otro tipo. Por eso se dice que Python es un lenguaje de tipado dinámico y podemos hacer cosas como:
:::

::: {#cceaacd7 .cell .code}
``` python
x = 1         # x is an integer
x = 'hello'   # now x is a string
x = [1, 2, 3] # now x is a list
```
:::

::: {#35b8093f .cell .markdown}
Esta es otra de las propiedades de Python que permite que sea rápido de escribir y fácil de leer.

##### ¡Cuidado!

Este tipado dinámico tiene otras consecuencias que hay que tener en cuenta.
:::

::: {#9223f204 .cell .code}
``` python
x = [1, 2, 3]
y = x
```
:::

::: {#6fc3c272 .cell .code}
``` python
print('x: ', x)
print('y: ', y)
```
:::

::: {#eedb95f7 .cell .code}
``` python
x.append(4)
```
:::

::: {#663aa068 .cell .code}
``` python
print(y)
```
:::

::: {#ad151fbc .cell .markdown}
Hemos usado el puntero x para cambiar el contenido del espacio de memoria al que apunta, por eso el valor de y cambia (y es otro puntero que apunta al mismo espacio de memoria que x).

Sin embargo, si usamos \'=\' para modificar x:
:::

::: {#b3a537f3 .cell .code}
``` python
x = 'hasta luego'

print (x)
print(y)
```
:::

::: {#899ca4b0 .cell .markdown}
Al usar \'=\' hemos hecho que x apunte a un espacio distinto.

El contenido del espacio de memoria al que apuntaba antes x (al que sigue apuntando y) está intacto.
:::

::: {#367c76d2 .cell .code}
``` python
x = 10
y = x
x += 5  # add 5 to x's value, and assign it to x
print("x =", x)
print("y =", y)
```
:::

::: {#c2647eec .cell .markdown}
### Objetos

Python es un lenguaje de programación orientada a objetos, y en Python todo es un objeto.
:::

::: {#867d0648 .cell .markdown}
Acabamos de ver que en Python las variables son simplemente punteros y que la declaración del nombre de una variable no lleva asociada información sobre su tipo. Sin embargo, eso no significa que Python sea un lenguaje sin tipos.

Python tiene tipos; pero no van asociados al nombre de las variables sino a los objetos en sí.
:::

::: {#56a20451 .cell .code}
``` python
x = 4
type(x)
```
:::

::: {#b027569e .cell .code}
``` python
x = 'hello'
type(x)
```
:::

::: {#d2ca4024 .cell .code}
``` python
x = 3.14159
type(x)
```
:::

::: {#b794d938 .cell .markdown}
# Operadores de identidad y pertenencia

## Operadores de identidad: \'is\' e \'is not\'

Identidad o identidad de objeto es distinto a igualdad.
:::

::: {#6b9955fd .cell .code}
``` python
a = [1, 2, 3]
b = [1, 2, 3]
```
:::

::: {#84121183 .cell .code}
``` python
a==b
```
:::

::: {#7348fa51 .cell .code}
``` python
a is b
```
:::

::: {#4bb5e95f .cell .code}
``` python
a is not b
```
:::

::: {#b4bc3661 .cell .markdown}
a y b apuntan a distintos objetos.

Antes hemos visto que en Python las variables son punteros. El operador \'is\' comprueba si las dos variables están apuntando al mismo contenedor/espacio de memoria (al mismo objeto).
Así, la mayoría de las veces que un principiante tiene la tentación de usar \'is\' lo que de verdad quiere hacer es \'==\'.
:::

::: {#1f7d51bd .cell .code}
``` python
a=b
```
:::

::: {#2f5f747e .cell .code}
``` python
a is b
```
:::

::: {#fd5f089e .cell .markdown}
## Operadores de pertenencia

Estos operadores comprueban pertenencia en objetos compuestos.
:::

::: {#5991375c .cell .code}
``` python
1 in [1,2,3]
```
:::

::: {#f14a533c .cell .code}
``` python
2 not in [1,2,3]
```
:::

::: {#00ccd2d0 .cell .markdown}
En los lenguajes de programación orientados a objetos como Python, un objeto es una entidad que contiene datos junto con metadatos y/o funcionalidad asociados.

En Python todo es un objeto, lo que significa que cada entidad tiene algunos metadatos (llamados atributos) y funcionalidad asociada (llamados métodos). Se accede a estos atributos y métodos a través de la sintaxis de puntos.
:::

::: {#37b031f3 .cell .code}
``` python
L = [1, 2, 3]
L.append(100)
print(L)
```
:::

::: {#31943d04 .cell .markdown}
Quizá nos resulte más extraño que no solo objetos compuestos (como las listas) sino también tipos simples de objetos tengan atributos y métodos asociados.
:::

::: {#4c408e69 .cell .code}
``` python
x = 4.5
print(x.real, "+", x.imag, 'i')
```
:::

::: {#8b62d551 .cell .code}
``` python
x.is_integer()
```
:::

::: {#9363d32b .cell .code}
``` python
x = 4.0
x.is_integer()
```
:::

::: {#eb7a1fc0 .cell .markdown}
Incluso los atributos y métodos de objetos son a su vez objetos con su propia información de tipo.
:::

::: {#0ed7e6dd .cell .code}
``` python
type(x.is_integer)
```
:::

::: {#8e2b2bb5 .cell .markdown}
# Flujo de control

## Condicionales
:::

::: {#f62a1331 .cell .code}
``` python
x = -15

if x == 0: 
    print(x, "es cero")
elif x > 0:
    print(x, "es positivo")
elif x < 0:
    print(x, "es negativo")
else:
    print(x, "no se parece a nada que haya visto antes ...")
```
:::

::: {#51b8158b .cell .markdown}
Tanto los dos puntos (:) como los espacios en blanco (indentación) se utilizan para indicar bloques de código separados.
:::

::: {#3cb1f1c6 .cell .markdown}
## Bucles

### Bucle for

Vamos a ver algunos ejemplos de bucles for en Python.

Si queremos imprimir cada uno de los elementos de una lista podríamos usar:
:::

::: {#c444ffbf .cell .code}
``` python
for N in [2, 3, 5, 7]:
    print(N, end=' ') # Python 3 only! print all on same line
    #print N ,         # Python 2 only! print all on same line
```
:::

::: {#947e0797 .cell .markdown}
Especificamos la variable que queremos usar, la secuencia sobre la que queremos iterar, y usamos el operador \'in\' para unirlos de forma intuitiva y legible. El objeto a la derecha del operador \'in\' puede ser cualquier iterador de Python.

Uno de los iteradores más usados en Python es el objeto range que genera una secuencia de números:
:::

::: {#d5671bb8 .cell .code}
``` python
for i in range(10):
    print(i, end=", ")
```
:::

::: {#e8b523ab .cell .code}
``` python
# range from 5 to 10
list(range(5, 10))
```
:::

::: {#15f5393e .cell .code}
``` python
# range from 0 to 10 by 2
list(range(-10, 10, 2))
```
:::

::: {#e9e7d0a5 .cell .markdown}
### Bucle while
:::

::: {#3a205548 .cell .code}
``` python
i = 0
while i < 10:
    print(i, end=", ")
    i += 1
```
:::

::: {#b802886d .cell .markdown}
### break and continue

Dos sentencias útiles que podemos utilizar dentro de los bucles para ajustar con precisión cómo se ejecutan.

-   break: rompe completamente el bucle
-   continue: salta lo que quede de la iteración actual del bucle y pasa directamente a la siguiente iteración

Ambos pueden utilizarse tanto en bucles for como en while
:::

::: {#18701583 .cell .code}
``` python
for n in range(20):
    # si el resto de n / 2 es 0, sáltate el resto del bucle
    if n % 2 == 0:
        continue
    print(n)
```
:::

::: {#cd73bc24 .cell .code}
``` python
# Completa una lista con la sucesión de Fibonacci hasta un valor determinado
a, b = 0, 1
amax = 100
L = []

while True:
    (a, b) = (b, a + b)   #asignación de tuplas, se procede valor por valor
    if a > amax:
        break
    L.append(a)

print(L)
```
:::

::: {#875b176e .cell .markdown}
# Funciones

En Python las funciones se definen con la sentencia def.
:::

::: {#b615707f .cell .code}
``` python
def fibonacci(N):
    L = []
    a, b = 0, 1
    while len(L) < N:
        a, b = b, a + b
        L.append(a)
    return L
```
:::

::: {#0d207f76 .cell .code}
``` python
fibonacci(10)
x =2
```
:::

::: {#cb6d967f .cell .markdown}
Hemos definido una función llamada fibonacci que tiene un único argumento N y que devuelve un valor, una lista con los N primeros números de la sucesión de Fibonacci.
:::

::: {#5a54c411 .cell .code}
``` python
fibonacci(10)
x = 2 
x
```
:::

::: {#6e35a73f .cell .markdown}
No se especifica ninguna información de tipos sobre la entrada o salida de la función. Una función puede devolver cualquier objeto de Python simple o compuesto.
:::

::: {#300c6122 .cell .markdown}
A veces cuando definimos una función, hay ciertos valores que queremos que la función use la mayor parte del tiempo (o que creemos que son los que se van a usar mayoritariamente) pero también nos gustaría dar al usuario opciones/flexibilidad. En esos casos, podemos usar valores por defecto para los argumentos de la función.
:::

::: {#e9b249b5 .cell .code}
``` python
def fibonacci(N, a=0, b=1):
    L = []
    while len(L) < N:
        a, b = b, a + b
        L.append(a)
    return L
```
:::

::: {#180bd629 .cell .code}
``` python
fibonacci(10)
```
:::

::: {#a9efdfbd .cell .code}
``` python
fibonacci(10, 0, 1)
```
:::

::: {#0e87060d .cell .code}
``` python
fibonacci(10, 0, 2)
```
:::

::: {#a35e55ce .cell .code}
``` python
fibonacci(10, b=3, a=1)
```
:::

::: {#e2e8698b .cell .markdown}
# Módulos y paquetes

Para poder cargar y utilizar módulos ya existentes utilizamos la sentencia import.

Podemos importar un módulo directamente. Cuando hagamos referencia al contenido de dicho módulo tendremos que indicar el namespace:
:::

::: {#95a07e79 .cell .code}
``` python
import math
math.cos(math.pi)
```
:::

::: {#67ba7e94 .cell .markdown}
Podemos importar un módulo asignándole un alias:
:::

::: {#6577f3c0 .cell .code}
``` python
import numpy as np
np.cos(np.pi)
```
:::

::: {#8773863b .cell .markdown}
Si solo nos interesa importar algunos elementos concretos del módulo también podemos hacerlo:
:::

::: {#d566358d .cell .code}
``` python
from math import cos, pi
cos(pi)
```
:::

::: {#d50c7fd0 .cell .markdown}
También podemos importar todo el contenido de un paquete en el espacio de nombres (namespace) local
:::

::: {#3a216a7c .cell .code}
``` python
from math import *
sin(pi) ** 2 + cos(pi) ** 2
```
:::

::: {#cfcb82a6 .cell .markdown}
Esta opción debe usarse con cuidado y moderación, si es que se usa. El problema es que tales importaciones pueden a veces sobrescribir nombres de funciones que no se pretende sobrescribir, y puede ser difícil darse cuenta de lo que hemos cambiado. Por ejemplo:
:::

::: {#3ed08d03 .cell .code}
``` python
help(sum)
```
:::

::: {#704dd1b1 .cell .code}
``` python
print(str(range(5)))
sum(range(5), -1)
```
:::

::: {#f9614bcf .cell .code}
``` python
from numpy import *
help (sum)
sum(range(5), -1)
```
:::

::: {#6d965ba2 .cell .markdown}
En el primero, sumamos range(5) empezando en -1; en el segundo, sumamos range(5) a lo largo del último eje (indicado por -1).
:::

::: {#5be6c8f7 .cell .markdown}
##### Módulos de \"terceros\"

Una de las cosas que hace que Python sea útil, especialmente en el mundo de la ciencia de datos, es su ecosistema de módulos de terceros. Estos pueden ser importados como los módulos built-in, pero primero tenemos que instalarlos en nuestro sistema. El registro estándar para estos módulos es el Python Package Index (PyPI), que se encuentra en la Web en <http://pypi.python.org/>.
Cuando queramos instalar uno de estos módulos en nuestro sistema usaremos el programa pip. Por ejemplo, si quisiéramos instalar el módulo supersmoother:

\$ pip install pandas

El código fuente del paquete se descargará automáticamente del repositorio de PyPI, y el paquete se instalará en la ruta estándar de Python (suponiendo que tenemos los permisos).

##### Si utilizáis ANACONDA

Con Anaconda, el comando será:

\$ conda install pandas

O a través de la Interfaz de usuario: Opción Environments
:::

::: {#fd4f2123 .cell .code}
``` python
```
:::

::: {#38f2633d .cell .code}
``` python
```
:::
