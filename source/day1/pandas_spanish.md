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

# Pandas Tutorial


<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/e/ed/Pandas_logo.svg/1200px-Pandas_logo.svg.png" width="30%">

```python
import pandas as pd
```

## Series y DataFrames

Los dos elementos principales de pandas son las `Series` y los `DataFrame`. 

Una `Series` es básicamente una columna y un `DataFrame` es una tabla multidimensional compuesta de una colección de `Series`.

<img src="https://storage.googleapis.com/lds-media/images/series-and-dataframe.width-1200.png" width=600px />




### Creando DataFrames desde cero

Se puede crear a partir de un simple `diccionario`.

En el ejemplo tenemos un puesto de frutas que vende manzanas y naranjas. Queremos una columna por cada fruta y una fila por cada compra de un cliente.


```python
data = {
    'apples': [3.2, 2, 0, 1], 
    'oranges': [0, 3, 7, 2],
    'propina': ['si', 'no', 'si', 'si']
}
```

```python
purchases = pd.DataFrame(data)
purchases
```

El **Indice** de este DataFrame se creó automaticamente al iniciarlo, usando los números0-3, pero podemos asignar los que queramos.

Los nombres de los clientes serán los índices 

```python
purchases = pd.DataFrame(data, index=['June', 'Robert', 'Lily', 'David'])

purchases
```

Ahora podemos buscar el pedido de un cliente usando su nombre:

```python
purchases.iloc[0].propina
```

```python
purchases.loc['June']
```

```python
purchases.iloc[0]
```

También podemos acceder por columnas

```python
purchases['oranges']
```

```python
atribs = ['oranges', 'propina']
purchases[atribs]
```

```python
purchases.oranges
```

### Leyendo datos desde un CSV


```python
df = pd.read_csv('purchases.csv')

df.head()
```

```python
?pd.read_csv
```

Al leer podemos elegir qué columna es el `index_col`:

```python
df = pd.read_csv('purchases.csv', index_col=0)

df
```

## Operaciones principales con DataFrame

Vamos a cargar la lista de películas IMDB:

```python
movies_df = pd.read_csv("IMDB-Movie-Data.csv", index_col="Title")
```

### Visualizando tus datos

Imprimimos unas pocas filas con `.head()`:

```python
movies_df.head()
```

`.head()` muesatra las primeras **cinco** filas por defecto, pero se puede especificar otro número `movies_df.head(10)`.

Para ver las últimas **filas** usamos `.tail()`. 

```python
movies_df.tail(2)
```

### Obteniendo información de tus datos

`.info()` debería ser uno de tus primeros métodos después de cargar tus datos

```python
movies_df.info()
```

```python
movies_df.shape
```

### Cambiar los nombres de las columnas



```python
movies_df.columns
```

```python

```

```python

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

Otra opción, queremos todos los nombres de las columnas en minúscula. En lugar de `.rename()`:

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

### Comprendiendo tus variables


Usando `describe()` obtenemos un resumen de la distribuación de todas las variables continuas:

```python
movies_df.describe()
```

<!-- #region -->


`.describe()` se puede usar también con variables categóricas
<!-- #endregion -->

```python
movies_df['genre'].describe()
```

```python
movies_df['genre'].value_counts().head(10)
```

#### Correlación entre variables continuas


Usando el comando `.corr()`:

```python
movies_df.corr()
```

### DataFrame: slicing, seleccionar y extraer




#### Por columna


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

#### Por filas

<!-- #region -->
 

- `.loc` - busca por nombre
- `.iloc`- busca por índice


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


#### Selección condicional 

<!-- #endregion -->

```python
condition = (movies_df.director == "Ridley Scott")

condition.head()
```

```python
movies_df[condition]
```

```python
movies_df[movies_df.index == 'Alien']
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
movies_df.ratin
```

# Ejercicio


Mostrar los directores que han dirigido una película de Sci-Fi con nota superior a un 8.
