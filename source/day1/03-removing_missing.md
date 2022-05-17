---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.2'
      jupytext_version: 1.6.0
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# How to tackle missing values


In this *notebook* we are going to apply several options to preprocess missing values.

```python
import pandas as pd
```

## Reading with missing values

The first thing is to assure a right reading of values. If we do not consider that they could be missing values, and how they are stored in CSV, it will interpret these values as normal strings.

```python
df = pd.read_csv("mammography.csv")
```

```python
df.head()
```

```python
df.dtypes
```

The solution is to indicate that '?' means a missing value, this is done as:

```python
df = pd.read_csv("mammography.csv", na_values=["?"])
```

```python
df.head()
```

```python
df.dtypes
```

# First option, to remove

The simpler option, not the best one, is to remove any instance with missing or not valide attributes. This can be easily done with dropna().

```python
df2 = df.copy()
df2.dropna().shape
```

```python
df.shape
```

The problem is that you can loss too much data/instances, usually it is not recommended.

# Another option: Replace them

Another option is to replace the missing values with a value, it can be used several strategies:
    
- More frequent value in that column (or in the same category). This is reasonable for qualitative variables.
- Average value, for numerical variables.
- Value to maintain the same average and std in the attribute.

In order to do that, we can use sklearn.impute, we will see with an example.

```python
from sklearn import impute
```

```python
df.Density.head()
```

```python
df.Density.tail(20)
```

First we create a SimpleImputer with the strategy, and then we apply it. Because SimpleImputer works with a matrix of vectors we use [], and then we 
use the first row of the result.

```python
help(impute.SimpleImputer)
```

```python
imputer = impute.SimpleImputer(strategy="most_frequent")

values = imputer.fit_transform([df.Density.values])
values
```

Now we update the attribute with the new values.

```python
df.Density.update(pd.Series(values[0]))
```

We check there is not NaN in the attribute.

```python
df.Density.head()
```

```python
df.Density.tail(20)
```

A more complete comparison.

```python
df.head()
```

```python
df3 = df.copy()
df3.dropna()
```

```python
df2 = df2.dropna()
```

```python
df2.shape
```

```python
df2.Density.describe()
```

```python
df3 = df3.dropna()
df3.shape
```

```python
df3.Density.describe()
```
