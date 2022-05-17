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

# Visualizando con Python


Un aspecto muy importante a la hora de procesar datos, sobre todo de cara a hacerlos entendibles, es una adecuada visualización de los mismos.


Por tanto, R y Python posee muchas librerías y herramientas para visualizar todo tipo de datos. En particular R es muy popular por su excelente librería **ggplot**.


Python posee varias excelentes librerías, no hay una clara ganadora, veremos algunas:

- **[Matplotlib](https://matplotlib.org/):** Librería más utilizada, muchas opciones, permite detallar mucho lo que se desea, pero tiene dos inconvenientes. 
   - Es laboriosa de usar.
   - Estilo un poco feo (herencia de Matlab) aunque se puede personalizar.
- **[Seaborn](https://seaborn.pydata.org/):** Librería construida sobre Matplotlib que presenta dos ventajas:
   - Estilo algo más bonito.
   - Tipos de gráficas más sencillas (boxplot), y además otros más avanzados (FaceGrid, ...).
   - Para ciertos detalles, es necesario conocer también Matplotlib.
- **[Desde Pandas](https://pandas.pydata.org/docs/getting_started/intro_tutorials/04_plotting.html):** Permite visualizar rápidamente información de sus atributos.
   - Feo pero rápido.
- **[Altair](https://altair-viz.github.io/):** Enfoque más declarativo, para web (o Notebook).


## Visualizando desde Matplotlib

```python
%pylab inline
```

```python
from matplotlib import pyplot as plt
```

```python
import numpy as np
```

Con  subplots() se pueden crear distintas figuras

```python
fig1, ax = plt.subplots()
ax.scatter(np.random.rand(10), np.random.rand(10))
ax.set_title("Figura")
ax.plot(np.random.rand(10),  np.random.rand(10), color='green')
```

Se "pinta" realmente en el campo ax, se pueden tener varias subfiguras.

```python
fig, ax = plt.subplots()
x = np.arange(1,11)
ax.scatter(x, x+10, marker='o')
ax.scatter(x, x*2+5, color='red')
```

## Añadiendo una leyenda


Parece bien, pero lo suyo sería una leyenda, ¿no? Para distinguir cada una. Se puede hacer usando el campo label.

```python
fig, ax = plt.subplots()
ax.scatter(x, x+10, marker='o', label='Lineal')
ax.scatter(x, x*2, color='red', label='Cuadrático')
```

Falta activarlo

```python
ax.legend()
fig
```

Podemos mover la leyenda:

```python
ax.legend(loc='lower right')
fig
```

## Creamos varias figuras

```python
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.plot(x, x+10)
ax2.scatter(x, np.log(x))
ax2.scatter(x, x)
ax2.set_xlim(0, 20)
```

```python
ax1.set_title("Figura 1")
ax2.set_title("Figura 2")
```

```python
fig
```

O de forma vertical

```python
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.scatter(np.random.rand(10), np.random.rand(10))
ax2.bar(["Granada", "Jaen"], np.random.rand(2), color='green')
```

```python
fig, [[ax1,ax2],[ax3,ax4]] = plt.subplots(2, 2)
ax1.scatter(np.random.rand(10), np.arange(10))
```

# Seaborn


No es diferente de programar en Matplotlib, pero tiene cosas interesantes.

```python
import seaborn as sns
```

```python
sns.set_theme()
```

```python
def example_plots():
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.scatter(np.random.rand(10), np.random.rand(10))
    ax2.bar(["Granada", "Jaen"], np.random.rand(2), color='green')
```

```python
example_plots()
```

## Cambiando estilo


Se pueden probar otros themas.

```python
sns.set_style('whitegrid')
example_plots()
```

Se pueden [configurar más estilos](https://seaborn.pydata.org/tutorial/aesthetics.html).


## Tipos de Gráficas


Pero lo bueno de seaborn son sus [propios tipos de gráficas](http://seaborn.pydata.org/tutorial.html).

```python
tips = sns.load_dataset("tips")
tips
```

```python
sns.relplot(data=tips, x="total_bill", y="tip");
ax = plt.gca();
x = np.linspace(0, 50, 2)
y = np.linspace(0, 10, 2)
ax.plot(x,  y, color='red')
```

Se ve que se pasa como data el DataFrame, y luego se puede indicar para cada dimensión el atributo del DataFrame (sólo se tienen en cuenta los usados).


## Identificando por objetivo


¿y si se quiere ver cuáles son fumadores? Habría dos opciones.


- Usando campo hue, que indica el atributo para distinguirlo, los datos de cada tipo tendrán un color, y se marcarán en la leyenda.

```python
tips['barato'] = tips.total_bill > 30;
sns.relplot(x="total_bill", y="tip", data=tips, col="time", hue="smoker", style="sex")
```

```python
tips['ratio'] = tips.tip/tips.total_bill
```

```python
sns.histplot(data=tips, x="total_bill", y="ratio")
```

Recientemente permite un campo col.

```python
sns.relplot(x="total_bill", y="tip", data=tips, col="smoker")
```

- Usando un FacetGrid que permite discriminar en función de un tipo por columna o fila, y luego con map pintar cada uno de ellos.

```python
g = sns.FacetGrid(data=tips, col="smoker")
g.map(sns.scatterplot, "total_bill", "tip")
```

Ahora veremos algunos tipos de gráficas muy útiles.

```python
sns.boxplot(data=tips, y='ratio', x='barato')
```

```python
sns.countplot(data=tips, x='smoker')
```

Es muy potente:

```python
sns.scatterplot(data=tips, x="total_bill", y="tip", hue="day", style="time")
```

Algo más grande:

```python
fig2, ax2 = plt.subplots(figsize=(12,7))
sns.scatterplot(data=tips, x="total_bill", y="tip", hue="day", style="time", ax=ax2)
```

Para manejar varios:

```python
penguins = sns.load_dataset("penguins")
sns.pairplot(penguins)
```

```python
sns.pairplot(penguins, hue="species")
```

# Ejemplo: Scikit-learn

```python
from sklearn import datasets, metrics, model_selection, svm, tree
X, y = datasets.make_classification(random_state=0)
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, random_state=0)
```

```python
X
```

```python
y
```

```python

```

Vamos a visualizar una curva ROC

```python
svc = svm.SVC(random_state=0)
svc.fit(X_train, y_train)
metrics.plot_roc_curve(svc, X_test, y_test)  # doctest: +SKIP
plt.show()                                   # doctest: +SKIP
```

Para añadir otros algoritmos es igual, solo hay que indicar que el campo ax es el mismo en todas.

```python
model_tree = tree.DecisionTreeClassifier(max_depth=5)
```

```python
model_tree.fit(X_train, y_train)
plot2 = metrics.plot_roc_curve(model_tree, X_test, y_test)  # doctest: +SKIP
plt.show()                                   # doctest: +SKIP
```

Si queremos ponerlo en la misma, indicamos el campo ax en el segundo (para reutilizar la figura anterior y no crear una nueva)

```python
ax = plt.gca()

for model in [svc, model_tree]:
    metrics.plot_roc_curve(model, X_test, y_test, ax=ax)  
```
