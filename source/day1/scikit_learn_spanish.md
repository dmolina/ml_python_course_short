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

# Primeros pasos con Scikit-Learn


<img src="https://s3.amazonaws.com/com.twilio.prod.twilio-docs/original_images/scikit-learn.png" width="30%">


Importando las librerías necesarias

```python
import pandas as pd
```

## Lectura de Datos


La entrada de datos se hace por medio de Pandas, normalmente de un fichero CSV o Excel.


Almacenamos en una variables la url desde la que vamos a descragar el dataset

```python
url = "https://raw.githubusercontent.com/vkrit/data-science-class/master/WA_Fn-UseC_-Sales-Win-Loss.csv"

```

Leemos el fichero csv con los datos

```python
sales_data= pd.read_csv(url)
```

```python
sales_data.head()
```

Exploración de datos

```python
sales_data.head(n=2)
```

Podemos ver los últimos registros también

```python
sales_data.tail()
```

Tipo de cada característica

```python
sales_data.dtypes
```

```python
sales_data.describe()
```

```python
sales_data.shape
```

# Pequeña Visualización de los Datos


Vamos a mostrar un poco los datos

```python
import seaborn as sns
import matplotlib.pyplot as plt
```

Countplot

```python
sns.set(style="whitegrid", color_codes=True)
```

```python
sns.set(rc={'figure.figsize':(11.7,8.27)})
```

```python
sns.countplot('Route To Market',data=sales_data,hue = 'Opportunity Result')
```

```python
sns.despine(offset=10, trim=True)
```

```python
plt.show()
```

```python
# sns.set(rc={'figure.figsize':(16.7,13.27)})
sns.boxplot(x='Region', y="Opportunity Amount USD", data=sales_data, hue="Opportunity Result")
plt.show()
```

```python
sns.boxplot(data=sales_data, x="Region", y="Elapsed Days In Sales Stage")
```

# Preprocesamiento de Datos


Tras estudiar un poco los datos vamos a usar Scikit-Learn para predecir "Opportunity Result"


Lo primero es tener en cuenta que los algoritmos de ML no trabajan con strings, por lo que es necesario codificar dichas cadenas como valores numéricos, por ejemplo.


Para ello, se usa la clase LabelEncoder(). Ponemos un ejemplo.

```python
from sklearn import preprocessing
# create the Labelencoder object
le = preprocessing.LabelEncoder()
#convert the categorical columns into numeric
encoded_value = le.fit_transform(["paris", "paris", "tokyo", "amsterdam", "tokyo"])
print(encoded_value)
```

```python
# La operación es reversible
le.inverse_transform(encoded_value)
```

```python
le.inverse_transform([1])
```

Tenemos bastante atributos de tipo cadena que pueden ser etiquetadas (el conjunto de valores es limitado)

```python
print("Supplies Subgroup' : ",sales_data['Supplies Subgroup'].unique())
print("Region : ",sales_data['Region'].unique())
print("Route To Market : ",sales_data['Route To Market'].unique())
print("Opportunity Result : ",sales_data['Opportunity Result'].unique())
print("Competitor Type : ",sales_data['Competitor Type'].unique())
print("Supplies Group : ",sales_data['Supplies Group'].unique())
```

Ahora vamos a transformar los datos mediante etiquetas, discretizando los datos.

```python
# create the Labelencoder object
le = preprocessing.LabelEncoder()
#convert the categorical columns into numeric
for col in ['Region', 'Route To Market']:
    sales_data[col] = le.fit_transform(sales_data[col])
    
sales_data['Supplies Subgroup'] = le.fit_transform(sales_data['Supplies Subgroup'])
# sales_data['Region'] = le.fit_transform(sales_data['Region'])
# sales_data['Route To Market'] = le.fit_transform(sales_data['Route To Market'])
sales_data['Opportunity Result'] = le.fit_transform(sales_data['Opportunity Result'])
sales_data['Competitor Type'] = le.fit_transform(sales_data['Competitor Type'])
sales_data['Supplies Group'] = le.fit_transform(sales_data['Supplies Group'])
#display the initial records
sales_data.head()
```

```python
sales_data.describe()
```

# Elegir los atributos de interés


Primero vamos a elegir el atributo que queremos predecir, y los atributos que usaremos para predecirlo.


- El objetivo es predecir "Opportunity Result".
- Escogemos todos los atributos menos 'Opportunity Number'(y el objetivo, evidentemente siempre se quita).

```python
# select columns other than 'Opportunity Number','Opportunity Result'
# dropping the 'Opportunity Number'and 'Opportunity Result' columns
cols = [col for col in sales_data.columns if col not in ['Opportunity Number','Opportunity Result']]
data = sales_data[cols]
#assigning the Opportunity Result column as target
target = sales_data['Opportunity Result']
data.head(n=2)
```

```python
target
```

# División de datos en conjuntos de entrenamiento y test


Para *evaluar* un algoritmo de ML es necesario dividir el conjunto de datos en dos partes:
    
- Conjunto de entrenamiento, utilizado para que el método de ML *aprenda*.
- Conjunto de test, para evaluar cuánto se equivoca con sus predicciones tras haber aprendido.


Para hacer la división usamos train_test_split, que divide de forma aleatoria entre conjunto de entrenamiento y test.

```python
from sklearn.model_selection import train_test_split
```

```python
help(train_test_split)
```

```python
data_train, data_test, target_train, target_test = train_test_split(data ,target, train_size = 0.70, random_state = 15)
```

```python
data_train.head(1)
```

```python
data_train.shape
```

```python
data_test.shape
```

# Construcción del modelo


Hay muchos modelos posibles disponibles desde Scikit-Learn.


<img src=https://scikit-learn.org/stable/_static/ml_map.png>



Usaremos dos modelos sencillos:
    
- Naive-Bayes: Modelo de predicción Bayesiano, basado en estadística.
- Linear SVC: Linear Support Vector Classification, muy popular.


## Primero aplicamos el Bayesiano (Naive-Bayes)

```python
# import the necessary module
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
```

### Creamos el objeto del  modelo

```python
#create an object of the type GaussianNB
gnb = GaussianNB()
```

```python
gnb
```

```python
help(GaussianNB)
```

### Ahora aprendemos el modelo pasándole los datos de entrenamiento

```python
model1 = gnb.fit(data_train, target_train)
```

### Medimos el % de acierto con el conjunto de test


Hacemos las predicciones del conjunto de test

```python
pred1 = gnb.predict(data_test)
```

```python
pred1
```

comparando el vector de predicciones con los valores reales, en % de aciertos.

```python
print("Naive-Bayes accuracy : ",accuracy_score(target_test, pred1, normalize = True))
```

Hemos obtenido un acierto del 75.9%, nada mal


### Ejemplo con Cross Validation

```python
from sklearn.model_selection import cross_val_score
```

```python
scores = cross_val_score(model1, data, target, cv=5, scoring='accuracy')
```

```python
scores
```

```python
np.mean(scores)
```

## Ahora aplicamos el Linear SVC


El comportamiento del Linear SVC se visualiza con el siguiente dibujo. Se verá en teoría.


<img src=https://www.dataquest.io/wp-content/uploads/2018/06/SVM-1.png>

```python
#import the necessary modules
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
```

Como se ve, todos los algoritmos se igual de la misma forma, el Interfaz es muy sencillo.


### Creamos el modelo (con un número de iteraciones)

```python
#create an object of type LinearSVC
# Requiere parámetros, como el número de iteraciones
svc_model = LinearSVC(random_state=10,max_iter=3000)
```

### Ahora aprendemos el modelo pasándole los datos de entrenamiento


Este modelo lleva mucho más tiempo

```python
svc_model.fit(data_train, target_train)
```

### Medimos el % de acierto con el conjunto de test

```python
pred2 = svc_model.predict(data_test)
print("LinearSVC accuracy : ",accuracy_score(target_test, pred2, normalize = True))
```

# Matriz de confusión


Hasta ahora hemos visto sólo el % de aciertos, pero nos puede interesar identificar los falsos positivos, y los falsos negativos.

```python
from sklearn.metrics import confusion_matrix
```

```python
m = confusion_matrix(target_test, pred1)
m
```

```python
total = m.sum()
total 
```

```python
data_test.shape[0]
```

```python
print("Verdadero Positivos:", m[0,0]/total)
print("Falsos Positivos:", m[0,1]/total)
print("Verdadero Negativos:", m[1,1]/total)
print("Falsos Negativos:", m[1,0]/total)
ratio = (m[0,0]+m[1,1])/total
print("Acccuracy:", ratio)
```
