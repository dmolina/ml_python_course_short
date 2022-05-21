---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.13.8
  kernelspec:
    argv:
    - python
    - -m
    - ipykernel_launcher
    - -f
    - '{connection_file}'
    display_name: Python 3
    env: null
    interrupt_mode: signal
    language: python
    metadata: null
    name: python3
---

# About this Notebook

In this notebook we are going to use several well-known models to classify using scikit-learn.


## Iris Dataset

First, we are going to solve the Iris Dataset, it is a classic one. 

The target is to identify the iris from the dimension of the flower. 

<img src="https://machinelearninghd.com/wp-content/uploads/2021/03/iris-dataset.png">

```python
from sklearn import datasets

iris = datasets.load_iris()
```

```python
y = iris.target
X = iris.data[:,:2]
```

```python
import pandas as pd
```

```python
y
```

We divide in train and test

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```

Now, we are going to use different classic algorithms.


## Lineal Classification


First, we are going to use a Linear Classifier.

```python
from sklearn.linear_model import SGDClassifier

model = SGDClassifier(max_iter=100)
model.fit(X_train, y_train)
```

```python
predict = model.predict(X_test)
```

```python
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
```

```python
accuracy_score(predict, y_test)
```

It is able to classify with great results.


We are going to visualize the decision bounds.

```python
from sklearn.inspection import DecisionBoundaryDisplay
import matplotlib.pyplot as plt
```

```python
disp = DecisionBoundaryDisplay.from_estimator(
    model,
    X[:,:2],
    ax=plt.gca(),
    response_method="predict",
    xlabel=iris.feature_names[0], ylabel=iris.feature_names[1], alpha=0.5
)
disp.ax_.scatter(X[:, 0], X[:, 1], c=iris.target, edgecolor="k")
```

We can see that the space is linearly divided in 3, to classify each instance.


We have use only two variables to visualize, with all four we achieve a better results. 


**Task: Modify the code to use the 4 attributes, and check the accuracy.**

```python
predict = model.predict(X_test)
```

```python
cross_val_score(model, X, y, cv=5).mean()
```

## K-Nearest Neighborhood

This algorithm allow us to classify considering the class of the nearest instances.

```python

from sklearn.neighbors import KNeighborsClassifier
```

```python
knn = KNeighborsClassifier(n_neighbors=3)
```

### Important: Data must be normalize

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
```

```python
model_knn = Pipeline([
            ("scaler", StandardScaler()),
            ("knn", knn)])
```

```python
model_knn.fit(X_train, y_train)
```

```python
predict = model_knn.predict(X_test)
```

```python
accuracy_score(predict, y_test)
```

```python
disp = DecisionBoundaryDisplay.from_estimator(
    model_knn,
    X[:,:2],
    ax=plt.gca(),
    response_method="predict",
    xlabel=iris.feature_names[0], ylabel=iris.feature_names[1], alpha=0.5
)
disp.ax_.scatter(X_test[:, 0], X_test[:, 1], c=y_test, edgecolor="k")
```

In this case, the region is grouping around each solution.


**Task: Apply with several neighborhood.**

```python
cross_val_score(model_knn, X, y, cv=5).mean()
```

## Support Vector Machine

SVM is a very popular classifier, that divide the space.

<img src="https://miro.medium.com/max/1200/1*06GSco3ItM3gwW2scY6Tmg.png" width="50%">

```python
from sklearn.svm import LinearSVC
```

```python
svc = LinearSVC()
```

```python
model_svc = Pipeline([("scale", StandardScaler()), ("svc", svc)])
```

```python
model_svc.fit(X_train, y_train)
```

```python
predict = model_svc.predict(X_test)
accuracy_score(predict, y_test)
```

```python
disp = DecisionBoundaryDisplay.from_estimator(
    model_svc,
    X[:,:2],
    response_method="predict",
    xlabel=iris.feature_names[0], ylabel=iris.feature_names[1], alpha=0.5
)
disp.ax_.scatter(X_test[:, 0], X_test[:, 1], c=y_test, edgecolor="k")
```

```python
cross_val_score(model_svc, X, y, cv=5).mean()
```

## Titanic Dataset


We are going to use another datasets, detection of Titanic.

```python
data = pd.read_csv("titanic.csv").dropna()
```

```python
data.shape
```

```python
data.head()
```

```python
y = data['Survived']
```

```python
X = data.drop(['Survived', 'Cabin', 'PassengerId', 'Name', 'Ticket'], axis=1)
```

```python
import seaborn as sns
disp = sns.countplot(x = 'Survived', hue = 'Sex', palette = 'Set1', data = data)
disp.set(title = 'Passenger status (Survived/Died) against Passenger Class', 
       xlabel = 'Passenger Class', ylabel = 'Total')
plt.show()
```

```python
from sklearn.preprocessing import LabelEncoder
```

```python
labels_t = {}

for col in ['Sex', 'Embarked']:
    labels_t[col] = LabelEncoder().fit(X[col])
    X[col] = labels_t[col].transform(X[col])
```

```python
X.info()
```

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=42)
```

## Decision Tree

The decision tree are one of the most intuitive models to predict a category. The idea is to automatically create a decision tree that, for each instance, in function of its attributes, a specific category is assigned.

```python
from sklearn.tree import DecisionTreeClassifier
```

```python
model_tree = DecisionTreeClassifier(max_depth=3)
```

```python
model_tree.fit(X_train, y_train)
```

We are going to visualize it.

```python

from sklearn import tree
```

```python
tree.plot_tree(model_tree)
plt.show()
```

```python
plt.figure(figsize=(50,50))
tree.plot_tree(model_tree, feature_names=X_train.columns)
plt.show()
```

```python
cross_val_score(model_tree, X, y, cv=5).mean()
```

## Ensemble models and Random Forest

In this example, we are going to use Ensemble Model, Random Forest.

<img src="https://miro.medium.com/max/1482/0*Srg7htj4TOMP5ldX.png">

```python
from sklearn.ensemble import RandomForestClassifier
```

```python
model_rf = RandomForestClassifier(n_estimators=50) # Limit the number of trees
```

```python
cross_val_score(model_rf, X, y, cv=5).mean()
```

# Regression

Although all previous examples are for classification, another important usage of ML is to do regression, in which we are not interesting in predict the category of a feature, but to predict a numerical feature, like the temperature for next day or the price of an item. In the following, we will see how to predict with popular XGBoost and with a Random Forest.

In Scikit-learn the final part of the name of the models indicate if they are classifier or for Regression, you can see the documentation to see many models for regression.

<!-- #region -->
## California Housing

This dataset implies to try to predict the price of houses in Boston. They use 8 features:


- **MedInc** median income in block group
- **HouseAge** median house age in block group
- **AveRooms** average number of rooms per household
- **AveBedrms** average number of bedrooms per household
- **Population** block group population
- **AveOccup** average number of household members
- **Latitude** block group latitude
- **Longitude** block group longitude

<!-- #endregion -->

## XGBoost

Although scikit-learn is the most used library for Machine Learning (aside for Deep Learning) it is not the only package with interesting models. There are specific libraries implementing specific models, as XGBoost.

XGBoost is an optimized distributed gradient boosting library that implements machine learning
algorithms under the Gradient Boosting framework. XGBoost provides a parallel tree boosting (also
known as GBDT, GBM) that solve many data science problems in a fast and accurate way. 

This is the first example using regression, that it is another important approach for ML.

```python
from sklearn.datasets import fetch_california_housing

data = fetch_california_housing()
```

```python
X = data.data
y = data.target
feature_names = data.feature_names
```

In order to evaluate the performance of our model, we split the data into training and test sets.

```python
X_train, X_test, y_train, y_test = train_test_split(X, y)
```

We load the package.

```python
import xgboost as xgb
```

Next, we initialize an instance of the XGBRegressor class. We can select the value of Lambda and Gamma, as well as the number of estimators and maximum tree depth.

```python
regressor = xgb.XGBRegressor(
    n_estimators=100,
    reg_lambda=1, 
    gamma=0,
    max_depth=3,
    random_state = 42,
)
```

We fit our model to the training set.

```python
regressor.fit(X_train, y_train)
```

We can examine the relative importance attributed to each feature, in determining the house price.

```python
pd.DataFrame(regressor.feature_importances_.reshape(1, -1), columns=feature_names)
```

As we can see, the median of incoming in the block is the greatest predictor of house price.


Finally, we use our model to predict the price of a house given what it has learnt.

```python
y_pred = regressor.predict(X_test)
```

We use the mean squared error to evaluate the model performance. The mean squared error is the average of the differences between the predictions and the actual values squared.

```python
from sklearn.metrics import mean_squared_error
```

```python
mean_squared_error(y_test, y_pred)
```

## RandomForestRegressor

We are going to compare against another regression model, the **RandomForestRegressor**, using the
same common parameters.

```python
from sklearn.ensemble import RandomForestRegressor
rf_regres = RandomForestRegressor(max_depth=3, n_estimators=100, random_state=42)
```

```python
y_pred2 = rf_regres.fit(X_train, y_train).predict(X_test)
```

```python
mean_squared_error(y_test, y_pred2)
```

It can be seen that the mean squared error from Random Forest is more than the double than using XGBoost.

# Task: Tackle 

We have a list of Seeds for Pumpins, we want to classify the Class from the Features. Test the different models compared, using cross_validation, and get the best models.

```python
from scipy.io import arff
import pandas as pd

data = arff.loadarff('Pumpkin_Seeds_Dataset.arff')
seeds = pd.DataFrame(data[0])
```

```python
seeds.head()
```

```python
target_labels = LabelEncoder().fit(seeds['Class'])
```

```python
y = target_labels.transform(seeds['Class'])
X = seeds.drop(['Class'], axis=1)
```
