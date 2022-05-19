---
jupyter:
  jupytext:
    formats: ipynb,md
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

# Visualizing with Python

An very important issue to process data, and mainly for make them understandable, is an adequate visualization of them.

Thus, R and Python have many libraries and tools to visualize all types of data. In particular R us very popular by its excellent library **gplot**.

Python have several excellent libraries also, there is not a clear winner:

- **[Matplotlib](https://matplotlib.org/):** Widely used library, many options, very configurable, but with several drawbacks:
   - Not easy to use, many details to define.
   - Not good style by default (from Matlab), but now it can be customized easier with themes.
- **[Seaborn](https://seaborn.pydata.org/):** Library upon Matplotlib that have two advantages:
   - Better style by default.
   - Many graphics (boxplot), and also other more advanced (FaceGrid, ...).
   However, for several customs, it requires to know also Matplotlib.
- **[From Pandas](https://pandas.pydata.org/docs/getting_started/intro_tutorials/04_plotting.html):** It let visualize information about the attributes/columns.
   - Feo pero rápido.
- **[Altair](https://altair-viz.github.io/):** More declarative approach, for web (or Notebook).


## Visualizing from Matplotlib

```python
%matplotlib inline
```

```python
from matplotlib import pyplot as plt
```

```python
import numpy as np
```

With subplots() you can create several figures.

```python
fig1, ax = plt.subplots()
ax.scatter(np.random.rand(10), np.random.rand(10))
ax.set_title("Figura")
ax.plot(np.random.rand(10),  np.random.rand(10), color='green')
```
Actually it is drawn in the variable ax, and you can have several subfigures.

```python
fig, ax = plt.subplots()
x = np.arange(1,11)
ax.scatter(x, x+10, marker='o')
ax.scatter(x, x*2+5, color='red')
```

## Append a legend

It seems nice, but it should have a legend to distinguish each plot. It can be done with the  label keyword.

```python
fig, ax = plt.subplots()
ax.scatter(x, x+10, marker='o', label='Linear')
ax.scatter(x, x*2, color='red', label='Quadratic')
```

It is not visible, you have to activate it.

```python
ax.legend()
fig
```
We can decide when to put the legend:

```python
ax.legend(loc='lower right')
fig
```
## Creating several figures

```python
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.plot(x, x+10)
ax2.scatter(x, np.log(x))
ax2.scatter(x, x)
ax2.set_xlim(0, 20)
```

```python
ax1.set_title("Figure 1")
ax2.set_title("Figure 2")
```

```python
fig
```

Or in a vertical way.

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

It is not different to Matplotlib, but it has several interesting advantages:

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

## We change the style

Several themes can be used:

```python
sns.set_style('whitegrid')
example_plots()
```

You can [set more styles](https://seaborn.pydata.org/tutorial/aesthetics.html).


## More pictures

The better of seaborn are its [own graphics](http://seaborn.pydata.org/tutorial.html).

We will analyse the tip in several  restaurants.

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

It can be seen that it accepts DataFrame as data, and you can indicate for each dimension the corresponding attribute from the DataFrame. 

## Identifying several groups

How is the different considering if the client is smoker or not? There are several options:

1. Using field `hue`, that indicate the attribute to separate, data of each group will have a different color, and it will be remarked in the legend.

```python
tips['cheap'] = tips.total_bill > 30;
sns.relplot(x="total_bill", y="tip", data=tips, col="time", hue="smoker", style="sex")
```

```python
tips['ratio'] = tips.tip/tips.total_bill
```

```python
sns.histplot(data=tips, x="total_bill", y="ratio")
```

2. The other option is using a field `col`, to create a new subfigure.

```python
sns.relplot(x="total_bill", y="tip", data=tips, col="smoker")
```

3. Using a FacetGrid that allow us to separate in rows or column, and then with `map` to draw each one of them.

```python
g = sns.FacetGrid(data=tips, col="smoker")
g.map(sns.scatterplot, "total_bill", "tip")
```

Now we will see another useful drawing.

```python
sns.boxplot(data=tips, y='ratio', x='cheap')
```

```python
sns.countplot(data=tips, x='smoker')
```

It is very potent:

```python
sns.scatterplot(data=tips, x="total_bill", y="tip", hue="day", style="time")
```

A little bigger:

```python
fig2, ax2 = plt.subplots(figsize=(12,7))
sns.scatterplot(data=tips, x="total_bill", y="tip", hue="day", style="time", ax=ax2)
```

To manage several of them:

```python
penguins = sns.load_dataset("penguins")
sns.pairplot(penguins)
```

```python
sns.pairplot(penguins, hue="species")
```

# Several Scikit-learn examples

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

We are going to visualize a ROC picture:

```python
svc = svm.SVC(random_state=0)
svc.fit(X_train, y_train)
metrics.RocCurveDisplay.from_estimator(svc, X_test, y_test) 
plt.show()                                   
```

To append more algorithms is the same thing, it is only required to use the same field `ax` in all of them.

```python
model_tree = tree.DecisionTreeClassifier(max_depth=5)
```

```python
model_tree.fit(X_train, y_train)
plot2 = metrics.RocCurveDisplay.from_estimator(model_tree, X_test, y_test) 
plt.show()                      
```

If we want to use the same way, we reuse the field `ax` (to reuse the previous figure and not to create a new one).

```python
ax = plt.gca()

for model in [svc, model_tree]:
    metrics.RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax)  
```
