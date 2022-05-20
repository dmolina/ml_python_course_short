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

# Extracting information by clustering

Clustering can allow us to extract useful information. For instance, from a
dataset we can set several groups of instance following a criterion, a compare
the different clusters obtained by each group.


This is an example used as practice in one of our courses, statistical
information about Spain obtained by the corresponding government agency. 


First, we will load all required libraries.

```python
import time

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
from sklearn.manifold import MDS
from math import floor
import seaborn as sns
```

```python
# First, we load the data
data = pd.read_csv('datos_hogar_2020.csv')
```

```python
# We select only information about families with house rent and public transport
subset = data.loc[(data['HY030N']>0) & (data['HC030_F']==1)]
```

```python
# rename variable
subset=subset.rename(columns={"HY020": "rent", "HY030N": "house_rent", "HC010": "food_in", "HC030": "transport"})
# We select the interesting attributes (numerical variables for the clustering)
used = ['rent','house_rent','food_in','transport']
```

```python
n_var = len(used)
X = subset[used].dropna()
```

```python
# remove outliers as which have the range outside of the 1.5 the intercurtil value
Q1 = X.quantile(0.25)
Q3 = X.quantile(0.75)
IQR = Q3 - Q1
X = X[~((X < (Q1 - 1.5 * IQR)) |(X > (Q3 + 1.5 * IQR))).any(axis=1)]
```

```python
from sklearn.preprocessing import MinMaxScaler

X_norm = MinMaxScaler().fit_transform(X)
```

```python
X_norm[:5,:]
```

```python
print('----- Running k-Means',end='')
k_means = KMeans(init='k-means++', n_clusters=5, n_init=5, random_state=123456)
t = time.time()
cluster_predict = k_means.fit_predict(X_norm,subset['DB090']) #se usa DB090 como peso para cada objeto (factor de elevación)
total_time = time.time() - t
print(": {:.2f} seconds, ".format(total_time), end='')
metric_CH = metrics.calinski_harabasz_score(X_norm, cluster_predict)
print("Calinski-Harabasz Index: {:.3f}, ".format(metric_CH), end='')

```

```python
sample_silhoutte = 0.2 if (len(X) > 10000) else 1.0
metric_SC = metrics.silhouette_score(X_norm, cluster_predict, metric='euclidean', sample_size=floor(sample_silhoutte*len(X)), random_state=123456)
print("Silhouette Coefficient: {:.5f}".format(metric_SC))

```

We convert the cluster to DataFrame to visualize

```python
clusters = pd.DataFrame(cluster_predict,index=X.index,columns=['cluster'])
```

```python
def print_cluster_size(clusters):
    print("Size for each cluster:")
    size=clusters['cluster'].value_counts()
    for num,i in size.iteritems():
        print('%s: %5d (%5.2f%%)' % (num,i,100*i/len(clusters)))
```

```python
print_cluster_size(clusters)
```

```python
centers = pd.DataFrame(k_means.cluster_centers_,columns=list(X))

```

```python
def denormalize(centers):
    centers_denorm = centers.copy()
    # se convierten los centros a los rangos originales antes de normalizar
    for var in list(centers):
        centers_denorm[var] = X[var].min() + centers[var] * (X[var].max() - X[var].min())

    return centers_denorm
```

```python
centers_denorm = denormalize(centers)
```

```python
def display_clusters(centers):
    centers_denorm = denormalize(centers)
    centers.index += 1
    hm = sns.heatmap(centers, cmap="YlGnBu", annot=centers_desnormal, annot_kws={"fontsize":18}, fmt='.3f')
    hm.set_ylim(len(centers),0)
    hm.figure.set_size_inches(15,15)
    hm.figure.savefig("centroides.png")
    centers.index -= 1
    return hm

```

```python
display_clusters(centers)
```

- **Tasks:** Show how the centroids change with k.

```python
def display_data(X, clusters):
    size=clusters['cluster'].value_counts()
    k = len(size)
    colors = sns.color_palette(palette='Paired', n_colors=k, desat=None)
    X_kmeans = pd.concat([X, clusters], axis=1)
    sns.set()
    variables = list(X_kmeans)
    variables.remove('cluster')
    sns_plot = sns.pairplot(X_kmeans, vars=variables, hue="cluster", palette=colors, plot_kws={"s": 25}, diag_kind="hist") # cluster set the color
    sns_plot.fig.subplots_adjust(wspace=.03, hspace=.03)
    sns_plot.fig.set_size_inches(15,15)
    return sns_plot
```

```python
display_data(X, clusters)
```

```python
def display_boxplot_clusters(X, centers, clusters):
    fig, axes = plt.subplots(k, n_var, sharey=True,figsize=(15,15))
    fig.subplots_adjust(wspace=0,hspace=0)
    # se añade la asignación de clusters como columna a X
    X_kmeans = pd.concat([X, clusters], axis=1)

    centers_sort = centers.sort_values(by=['rent']) #ordenamos por renta para el plot

    rango = []
    for j in range(n_var):
        rango.append([X_kmeans[used[j]].min(),X_kmeans[used[j]].max()])

    for i in range(k):
        c = centers_sort.index[i]
        dat_filt = X_kmeans.loc[X_kmeans['cluster']==c]

        for j in range(n_var):
            ax = sns.boxplot(x=dat_filt[used[j]], notch=True, color=colors[c], flierprops={'marker':'o','markersize':4}, ax=axes[i,j])

            if (i==k-1):
                axes[i,j].set_xlabel(used[j])
            else:
                axes[i,j].set_xlabel("")
        
            if (j==0):
                axes[i,j].set_ylabel("Cluster "+str(c+1))
            else:
                axes[i,j].set_ylabel("")
        
            axes[i,j].set_yticks([])
            axes[i,j].grid(axis='x', linestyle='-', linewidth='0.2', color='gray')
            axes[i,j].grid(axis='y', b=False)
        
            ax.set_xlim(rango[j][0]-0.05*(rango[j][1]-rango[j][0]),rango[j][1]+0.05*(rango[j][1]-rango[j][0]))

    fig.set_size_inches(15,15)
    return fig

```

```python
display_boxplot_clusters(X, centers, clusters)
```

```python
def display_centers(centers):
    mds = MDS(random_state=123456)
    centers_mds = mds.fit_transform(centers)
    fig=plt.figure(4)
    plt.scatter(centers_mds[:,0], centers_mds[:,1], s=size*10, alpha=0.75, c=colors)

    for i in range(k):
        plt.annotate(str(i+1),xy=centers_mds[i],fontsize=18,va='center',ha='center')

    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    fig.set_size_inches(15,15)
    return fig
```

```python
display_centers(centers)
```
