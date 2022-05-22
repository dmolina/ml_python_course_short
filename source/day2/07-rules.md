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

# Association Rules


## Market Basket Analysis Introduction

Companion notebook to http://pbpython.com/market-basket-analysis.html

<img src="https://miro.medium.com/max/1400/1*z8powfNMseAgzCG6cF48wA.jpeg">
<figcaption>Photo by https://blogs.oracle.com/datascience/overview-of-traditional-machine-learning-techniques</figcaption>


Association Rules Analysis, also called Market Basket Analysis, has become familiar for analysis in the retail industry. 

Personal recommendations is becoming very population in applications such as Spotify, Netflix, and Youtube. 

One of the biggest examples of Association Rules Analysis is the correlation between beer and diaper. When Walmart, a chain store in the United States, studied the shopping behavior of customers, the study showed that diapers and beers are bought together. The reason is because while mothers stay in the house with the baby, usually fathers have to go shopping, and they buy beers (because they stay more in home).


## Apriori Algorithm

In this notebook we show the Apriori Algorithm, one of the most popular and classical algorithm. In this algorithm, there are product clusters that that pass frequently, and the relationships between them are remarked.

The importance of an Association Rules can be determined by 3 parameters:

- Support
- Confidence
- Lift

<img src="https://miro.medium.com/max/1352/1*dqCXKUfEE9Eau17kJuCp2A.png" src="50%"> 

In our case:

- Support : It is the probability of an event to occur (coverage of the rule).

- Confidence : It is a measure of conditional probability of the rule.

- Lift : It is the probability of all items occurring together divided by the product of antecedent and consequent occurring as if they are independent of each other.


# First Dataset, correlations between products

In this dataset we get information about several invoices, each line contains all products buy for the same client.
The data are obtained by https://www.kaggle.com/datasets/shazadudwadia/supermarket, I have downloaded to make it easier.


First, we load the packages.

```python
import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
```

```python
df = pd.read_csv("GroceryStoreDataSet.csv", names=['products'], sep=',')
df.head()
```

```python
df.shape
```

First, split the data creating a list.

```python
data = list(df['products'].apply(lambda x: x.split(",")))
data
```

We use TransactionEncoder to convert the list representing the 'True' to 1 and 'False' to '0'.
Products that customers bought or did not buy during shopping will now be represented by values 1 and 0.

```python
#Let's transform the list, with one-hot encoding
from mlxtend.preprocessing import TransactionEncoder
a = TransactionEncoder()
a_data = a.fit(data).transform(data)
df = pd.DataFrame(a_data,columns=a.columns_)
df = df.replace(False,0)
df
```

## Applying Apriori

We change all the parameters in the Apriori Model in the mlxtend package.
I will try to use minimum support parameters for this modeling, at 20%.


```python
apri = apriori(df, min_support = 0.2, use_colnames = True, verbose = 1)
apri
```

Now, we are going to create the association rules:

```python
rules = association_rules(apri, metric = "lift", min_threshold=1)
rules.sort_values('lift',ascending=False)
```

## Visualizing the relation between antecedents and consequents

We are going to visualizde the relationships:

```python
from matplotlib import pyplot as plt
```

```python

# Import seaborn under its standard alias
import seaborn as sns
# Replace frozen sets with strings
rules['antecedents_'] = rules['antecedents'].apply(lambda a: ','.join(list(a)))
rules['consequents_'] = rules['consequents'].apply(lambda a: ','.join(list(a)))
# Transform the DataFrame of rules into a matrix using the lift metric
pivot = rules.pivot(index = 'antecedents_', 
                    columns = 'consequents_', values= 'lift')
# Generate a heatmap with annotations on and the colorbar off
sns.heatmap(pivot, annot = True)
plt.yticks(rotation=0)
plt.xticks(rotation=90)
plt.show()
```

# Association Rules with more complex example

In this notebook we are going to use association rules to extract patterns from
online retails, a more realistic example.


First we read the date (first line for reading online, second is faster).

```python
# df = pd.read_excel('http://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx')
df = pd.read_excel('Online%20Retail.xlsx')
```

```python
df.head()
```

```python
# Clean up spaces in description and remove any rows that don't have a valid invoice
df['Description'] = df['Description'].str.strip()
df.dropna(axis=0, subset=['InvoiceNo'], inplace=True)
```

```python
# Convert to str
df['InvoiceNo'] = df['InvoiceNo'].astype('str')
# Ignore InvoiceNo with C
df = df[~df['InvoiceNo'].str.contains('C')]
```

```python
# Count from France
basket = df[df['Country'] =="France"].groupby(['InvoiceNo','Description'])['Quantity'].sum()
basket.head()
```

```python
# Count for each invoice the number of each product. If it is nan, clearly it was not bought
basket=basket.unstack().fillna(0)
basket.head()
```

This data count how many items are bought, we only are interested if they are
bought, not the number. Thus, we replace all values greater than 0 to 1.

```python
# Convert the units to 1 hot encoded values
def encode_units(x):
    if x <= 0:
        return 0
    if x >= 1:
        return 1
```

```python
basket_sets = basket.applymap(encode_units)
```

```python
# No need to track postage
basket_sets.drop('POSTAGE', inplace=True, axis=1)
```

We generate frequent item sets that have a support of at least 7% (this number was chosen so that I could get enough useful examples):

```python
# Build up the frequent items
frequent_itemsets = apriori(basket_sets, min_support=0.07, use_colnames=True)
```

```python
frequent_itemsets
```

```python
# Create the rules
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
rules
```

```python
rules[ (rules['lift'] >= 6) &
       (rules['confidence'] >= 0.8) ]
```

In looking at the rules, it seems that the green and red alarm clocks are purchased together and the red paper cups, napkins and plates are purchased together in a manner that is higher than the overall probability would suggest.

```python
basket['ALARM CLOCK BAKELIKE GREEN'].sum()
```

```python
basket['ALARM CLOCK BAKELIKE RED'].sum()
```

```python
basket2 = (df[df['Country'] =="Portugal"]
          .groupby(['InvoiceNo', 'Description'])['Quantity']
          .sum().unstack().reset_index().fillna(0)
          .set_index('InvoiceNo'))
```

```python
basket_sets2 = basket2.applymap(encode_units)
```

```python
basket_sets2.drop('POSTAGE', inplace=True, axis=1)
```

```python
frequent_itemsets2 = apriori(basket_sets2, min_support=0.05, use_colnames=True)
```

```python
rules2 = association_rules(frequent_itemsets2, metric="lift", min_threshold=1)
rules2
```

```python
rules2[ (rules2['lift'] >= 4) &
        (rules2['confidence'] >= 0.5) ]
```

**Task**: Analyze the results in Portugal. 

**Task2**: Run the experiments for Italy and analyze it.
