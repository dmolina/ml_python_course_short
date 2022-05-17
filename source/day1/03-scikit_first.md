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

# First steps with Scikit-Learn


<img src="https://s3.amazonaws.com/com.twilio.prod.twilio-docs/original_images/scikit-learn.png" width="30%">

First, we import pandas.

```python
import pandas as pd
```

## Reading Data

The loading of data is going to be through Pandas, usually from a CSV or Excel file.

We store in a variable the url from which we are going to download it.

```python
url = "https://raw.githubusercontent.com/vkrit/data-science-class/master/WA_Fn-UseC_-Sales-Win-Loss.csv"

```

Read the CSV file.

```python
sales_data= pd.read_csv(url)
```

```python
sales_data.head()
```
## Exploration of data

```python
sales_data.head(n=2)
```

We can see the last rows:

```python
sales_data.tail()
```

Check the type of each attribute:

```python
sales_data.dtypes
```

```python
sales_data.describe()
```

```python
sales_data.shape
```

# Small visualization of data

We are going to show a little the data.

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

# Data Preprocessing

Now we are going to use Scikit-Learn to predict "Opportunity Result".

The first first is to take in account that scikit-learn does not work with strings, so it is needed to codify the string as numeric values, labels.

In order to do that, we use the class `LabelEncoder()`:

```python
from sklearn import preprocessing
# create the Labelencoder object
le = preprocessing.LabelEncoder()
#convert the categorical columns into numeric
encoded_value = le.fit_transform(["paris", "paris", "tokyo", "amsterdam", "tokyo"])
print(encoded_value)
```

```python
# The operation is reversible
le.inverse_transform(encoded_value)
```

```python
le.inverse_transform([1])
```

We have many attributes that should be labeled.

```python
print("Supplies Subgroup' : ",sales_data['Supplies Subgroup'].unique())
print("Region : ",sales_data['Region'].unique())
print("Route To Market : ",sales_data['Route To Market'].unique())
print("Opportunity Result : ",sales_data['Opportunity Result'].unique())
print("Competitor Type : ",sales_data['Competitor Type'].unique())
print("Supplies Group : ",sales_data['Supplies Group'].unique())
```

In order to do that, we create a `LabelEncoder` for each column, to avoid any conflcit.

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

# Select several interesting attributes

We are going to select one attribute to predict, and the attributes used to do that.


- The goal is to predict "Opportunity Result".
- We choose all attributes except "Opportunity Number" (the goal, obviously, is always removed).

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

# Divide the data in training and test sets

To *evaluate* how good is an ML algorithm is need to divide the data in two parts:
    
- **Training** set, used to *train* the ML algorithm.
- **Test* set, to *evaluate* the performance of the ML algorithm. Obviously, test instances cannot be in training one.

To divide the division we will use train_test_split, that divide randomly in training and test.

```python
from sklearn.model_selection import train_test_split
```

```python
help(train_test_split)
```

```python
data_train, data_test, target_train, target_test = train_test_split(data, target, train_size = 0.70, random_state = 15)
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

# Training and predicting with a model

There are many models available in Scikit-learn.


<img src=https://scikit-learn.org/stable/_static/ml_map.png>

The API for all models is the same, in order to be able to replace one or another without problems.

In this first step we will use two simple models.
    
- Naive-Bayes: Bayesian model, based on statistics.
- Linear SVC: Linear Support Vector Classification, popular model.

## First we apply the Bayesian (Naive-Bayes)

```python
# import the necessary module
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
```

### We create the object of the model

For the models there are several parameters to configure it, but the majority of them have rather good default values.

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

### Now we train the model with method `fit` and the training instances

```python
model1 = gnb.fit(data_train, target_train)
```

### We measure the accuracy ratio using the test set

First, we use `predict` to predict the class for each instance of test.

```python
pred1 = gnb.predict(data_test)
```

```python
pred1
```

Now we compare the accuracy using the real values. It is only one measure, there are a lot more.

```python
print("Naive-Bayes accuracy : ",accuracy_score(target_test, pred1, normalize = True))
```

We have achieved a good accuracy value (for the simple model).

### Cross Validation

The division in train, test is usually not enough, because the results depends a lot of the simple grouping.

In theory, you should have learn about `Cross Validation`. We are going to use it.

<img src="https://miro.medium.com/max/4984/1*kheTr2G_BIB6S4UnUhFp8g.png">

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

That accuracy value is more robust.

## Now we will apply Linear SVC

The Linear SVC behaviour is visualized with the following picture:


<img src=https://www.dataquest.io/wp-content/uploads/2018/06/SVM-1.png>

```python
#import the necessary modules
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
```

As you can see, all algorithms are trained and tested in the same way, the API is very simple.

### Create the model (with a number of iterations)

```python
#create an object of type LinearSVC
# It requires parameters, like the number of iterations
svc_model = LinearSVC(random_state=10,max_iter=3000)
```

### Training

This model training takes a lot of time:

```python
svc_model.fit(data_train, target_train)
```

### Test

We measure the accuracy with the test set:

```python
pred2 = svc_model.predict(data_test)
print("LinearSVC accuracy : ", accuracy_score(target_test, pred2, normalize = True))
```

# Confusion Matrix

Until now we have checked only the % of accuracy, but it could be useful to identify true/false positives and true/false negatives.

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
print("True Positives:", m[0,0]/total)
print("False Positives:", m[0,1]/total)
print("True Negatives:", m[1,1]/total)
print("False Negatives:", m[1,0]/total)
ratio = (m[0,0]+m[1,1])/total
print("Acccuracy:", ratio)
```

## Visual Confusion Matrix

We can visualize it directly with:

```python
from sklearn.metrics import ConfusionMatrixDisplay

disp=ConfusionMatrixDisplay(confusion_matrix=m)
disp.plot()
```

We can also use the model and test to visualize the Confusion Matrix. Also, it can be normalized:

```python
ConfusionMatrixDisplay.from_predictions(target_test, pred1, normalize='all')
```

It can be observed that results are similar to previous ones.

We can also normalize by rows:

```python
ConfusionMatrixDisplay.from_predictions(target_test, pred1, normalize='true')
```

This means that it is very good predicting one class, not the other one.
