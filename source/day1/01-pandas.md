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

# Pandas Tutorial


<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/e/ed/Pandas_logo.svg/1200px-Pandas_logo.svg.png" width="30%">

```python
import pandas as pd
```

## DataFrames and Series

The two main elements in Pandas are `Series` and ``DaraFrames`.

`Series` is basically the information of a column/attribute, and `DataFrame` is a multidimensional table composed by a collection of `Series`.

<img src="https://storage.googleapis.com/lds-media/images/series-and-dataframe.width-1200.png" width=600px />


### Creating DataFrames from scratch

You can create a `DataFrame` from a simple `Dictionary`.

In the example we have a fruit shop that sells apples and oranges. We want a column for each fruit, and a row for the sell of a client.

```python
data = {
    'apples': [3.2, 2, 0, 1], 
    'oranges': [0, 3, 7, 2],
    'tip': ['yes', 'no', 'yes', 'yes']
}
```

```python
purchases = pd.DataFrame(data)
purchases
```

The **Index** of this DataFrame was automatically create, using the number 0-3, but we could assign them as we wish.

Now the client's names will be the indexes:

```python
purchases = pd.DataFrame(data, index=['June', 'Robert', 'Lily', 'David'])

purchases
```

Now we can search the data from a client using its name:

```python
purchases.iloc[0].tip
```

```python
purchases.loc['June']
```

```python
purchases.iloc[0]
```
We can also access by row:

```python
purchases['oranges']
```

```python
atribs = ['oranges', 'tip']
purchases[atribs]
```

```python
purchases.oranges
```

### Reading from a CSV


```python
df = pd.read_csv('purchases.csv')

df.head()
```

```python
?pd.read_csv
```

Reading we can choose with column is `index_col`:

```python
df = pd.read_csv('purchases.csv', index_col=0)

df
```

## Usual operations with DataFrame

We are going to load a list of IMDB films:

```python
movies_df = pd.read_csv("IMDB-Movie-Data.csv", index_col="Title")
```

### Visualizing the Data

We show a few rows with `.head()`:

```python
movies_df.head()
```

`.head()` show the first **cinco** rows by default, but you can indicate another number `movies_df.head(10)`.

To see the last **rows** we use `.tail()`. 

```python
movies_df.tail(2)
```

### Getting information from your data

`.info()` should one of your first methods after loading your data.

```python
movies_df.info()
```

```python
movies_df.shape
```

shape returns the number of instance and columns.

### Rename the column's name

We can rename the column names if we want it.

```python
movies_df.columns
```

```python
movies_df.rename(columns={
        'Runtime (Minutes)': 'Runtime', 
        'Revenue (Millions)': 'Revenue_millions'
    }, inplace=True)


movies_df.columns
```

```python
movies_df.Runtime
```

```python
movies_df.head()
```

Another option, to have all column names in lowercase. Instead of `rename()` we directly modify the `.columns` field.

```python
movies_df.columns = ['rank', 'genre', 'description', 'director', 'actors', 'year', 'runtime', 
                     'rating', 'votes', 'revenue_millions', 'metascore']


movies_df.columns
```

But that's too much work. Instead of just renaming each column manually we can do a list comprehension:

```python
movies_df.columns = [col.lower() for col in movies_df]

movies_df.columns
```

### Learning about numerical variables


`describe()` returns a summary of the distribution of all numerical columns:

```python
movies_df.describe()
```

<!-- #region -->


`.describe()` can be used also with categorical values.
<!-- #endregion -->

```python
movies_df['genre'].describe()
```

```python
movies_df['genre'].value_counts().head(10)
```

#### Correlation between attributes


With method: `.corr()`:

```python
movies_df.corr()
```

### DataFrame: slicing, filter and 

#### By columns


```python
genre_col = movies_df['genre']

type(genre_col)
```

```python
genre_col = movies_df[['genre']]

type(genre_col)
```

```python
subset = movies_df[['genre', 'rating']]

subset.head()
```

#### By rows

<!-- #region -->
 

- `.loc` - search by column name.
- `.iloc`- search by index.


<!-- #endregion -->

```python
prom = movies_df.loc["Guardians of the Galaxy"]

prom
```

```python
prom = movies_df.iloc[0]
prom
```

```python
movie_subset = movies_df.loc['Prometheus':'Sing']

movie_subset = movies_df.iloc[1:4]

movie_subset
```

<!-- #region -->


#### Conditional Selection

<!-- #endregion -->

```python
condition = (movies_df.director == "Ridley Scott")

condition.head()
```

```python
movies_df[condition]
```

```python
movies_df[movies_df.index == 'La La Land']
```

```python
movies_df[movies_df['director'] == "Ridley Scott"].head()
```

```python
movies_df[movies_df['rating'] < 4].sort_values('rating', ascending=True)
```

```python
help(movies_df.sort_values)
```

```python
movies_df.rating
```

# Exercise

Show the directors that have done a Sci-fi film with an rating greater or equals to 8.
