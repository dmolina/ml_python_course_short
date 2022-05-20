---
jupyter:
  jupytext:
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

# Applying clustering to a real problem

In this notebook we are going to solve a real problem, grouping people in
function on its personality characteristics. 

The data are obtained from
https://www.kaggle.com/datasets/arslanali4343/top-personality-dataset?select=2018-personality-data.csv

The personality-data contains the header which described as follows:

- **Userid:** the hashed user_id, *to ignore*.

- **Openness:** an assessment score (from 1 to 7) assessing user tendency to prefer new experience. 1 means the user has tendency NOT to prefer new experience, 7 means the user has tendency to prefer new experience.

- **Agreeableness:** an assessment score (from 1 to 7) assessing user tendency to be compassionate and cooperative rather than suspicious and antagonistic towards others. 1 means the user has tendency to NOT be compassionate and cooperative. 7 means the user has tendency to be compassionate and cooperative.

- **Emotional Stability:** an assessment score (from 1 to 7) assessing user tendency to have psychological stress. 1 means the user has tendency to have psychological stress, and 7 means the user has tendency to NOT have psychological stress.

- **Conscientiousness:** an assessment score (from 1 to 7) assessing user tendency to be organized and dependable, and show self-discipline. 1 means the user does not have such a tendency, and 7 means the user has such tendency.

- **Extraversion:** an assessment score (from 1 to 7) assessing user tendency to be outgoing. 1 means the user does not have such a tendency, and 7 means the user has such a tendency.

- **Assigned Metric:** one of the follows (serendipity, popularity, diversity, default). Each user, besides being assessed their personality, was evaluated their preferences for a list of 12 movies manipulated with serendipity, popularity, diversity value or none (default option).

- **Assigned Condition:** one of the follows (high, medium, low). Based on the
  assigned metric, and this assigned condition, the list of movies was generated
  for the users. For example: if the assigned metric is serendipity and the
  assigned condition is high, the movies in the list are highly serendipitous.
  We document how we manipulated the movie list based on the assigned condition
  and assigned metric in page 6 of our research paper mentioned above.
 
- **Movie_x (x is from 1 to 12):** Valoration of several films, to ignore.


First, we are going to do a preprocessing:

```python
import pandas as pd

```

```python
df_raw = pd.read_csv("2018-personality-data.csv")
# remove spaces
df_raw.head()
```

Remove useful attributes.

```python
ignores = [f"movie_{i}" for i in range(1,13)] + [f"predicted_rating_{i}" for i in range(1,13)]
ignores.append("userid")
ignores
```

```python
df_raw.columns
```

```python
df = df_raw.copy()
df.columns = df_raw.columns.str.replace(' ', '')
df = df_raw.drop(ignores, axis=1)
df.head()
```

```python
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
```

```python
labels = {}
# Now we use labels
for col in df.columns:
    if 'assigned' in col:
        labels[col] = LabelEncoder().fit(df[col])
        df[col] = labels[col].transform(df[col])
```

```python
df.head()
```

```python
# Now we normalize
df_norm = MinMaxScaler().fit_transform(df)
df_norm
```

Now it is the funny part, apply the clustering.


**First task:** Apply the K-Means with several k and to decide the best k for
the data.


**Second task:** Compare results against DBSCAN.


**Third task:** Apply the hierarchical model.
