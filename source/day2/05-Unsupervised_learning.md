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

# Unsupervised Learning

In this notebook, we are going to tackle unsupervised learning. In this
learning, there is not a "right" category. Our target is to extract common
patterns from the data, grouping them in function of their similarities.


## Clustering
Clustering of unlabeled data can be performed with the module sklearn.cluster.

We have several clustering methods.

<img src="https://scikit-learn.org/stable/_images/sphx_glr_plot_cluster_comparison_001.png">




Clustering algorithm comes in two variants: 

- A class, that implements the fit method to learn the clusters on train data.
- A function, that, given train data, returns an array of integer labels corresponding to the different clusters. For the class, the labels over the training data can be found in the labels_ attribute.


## K-Means

K-Means is the simpler clustering algorithm, the idea is to group together
elements considering its distance.

<img
src="https://scikit-learn.org/stable/_images/sphx_glr_plot_kmeans_assumptions_001.png" width="50%">

This algorithms requires to indicate the number of clusters.

The good behavior depends on the distribution of the instances.

**Important:** As always that the distance is used, the data must be previously
normalized.




First, we are going to use a synthetic problem:

```python
 from sklearn.datasets import make_blobs
```

```python
k = 3
```

```python
features, _ = make_blobs(n_samples=200,
                         centers=k,
                         cluster_std=2.75,
                         random_state=42)

```

We are going to visualize them:

```python
%matplotlib inline
import seaborn as sns
from matplotlib import pyplot as plt
```

```python
features.shape
```

```python
sns.relplot(x=features[:,0], y=features[:,1])
```

Now, we are going to use K-Means to classify them. This algorithm define several
centroids, each for one cluster, assign the instances to the centroids nearest,
and in an iterative way, the centroids are optimized. 

<img
src="https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fmiro.medium.com%2Fmax%2F1280%2F1*rwYaxuY-jeiVXH0fyqC_oA.gif&f=1&nofb=1" width="50%"> 

```python
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
```

```python
kmeans = KMeans( init="random", # Init randomly
    n_clusters=k, # N-Cluster
    n_init=10,  # Scikit-Learn apply the algorithm several time to be more robust
    max_iter=300, # Iterations
    random_state=42 # Random state
)
# First normalize
features_norm = StandardScaler().fit_transform(features)
```

```python
# Now we apply the kmeans
kmeans.fit(features_norm)
```

```python
# we can see the clusters
clusters = kmeans.cluster_centers_
```

```python
plot = sns.relplot(x=features_norm[:,0], y=features_norm[:,1], color='gray')

for i in range(k):
    sns.scatterplot(x=[clusters[i,0]],y=[clusters[i,1]], ax=plot.ax, marker='^', s=100)
```

Now we are going to assign each point with its cluster. They are assigned in labels.

```python
labels = kmeans.labels_
labels
```

```python
import numpy as np
```

```python
features_np = np.array(features_norm)
```

```python
def plotpoints(labels):
    for i in range(k):
        posi = (labels==i)
        sns.scatterplot(x=features_np[posi,0],y=features_np[posi,1])
```

```python
plotpoints(kmeans.labels_)
```

## Measures

In this case, there is not a right solution, so it has no sense to use a measure
as accuracy or f1. In this case, we use another metrics like the **silhouette
coefficient**. Sihoette is a measure of cluster cohesion and separation. 

It quantifies how well a data point fits into its assigned cluster based on two
factors:

- Inter-cluster distance: How close the data point is to other points in the cluster
- Intra-cluster distance: How far away the data point is from points in other
  clusters
  
The objective is to maintain the inter-cluster distance low, and a large
intra-cluster distance.

Silhouette coefficient values range between -1 and 1.  Larger numbers indicate
that samples are closer to their clusters than they are to other clusters, so
the goal is to minimize that measure.

```python
from sklearn.metrics import silhouette_score
```

```python
silhouette_score(features_np, kmeans.labels_)
```

Is it a good or a bad value? We are going to show how it changes when the k changes.

```python
measures = []

for k in range(2, 11):
    kmeans = KMeans(init="random", n_clusters=k, n_init=10, max_iter=300, random_state=42)
    kmeans.fit(features_np)
    measures.append(silhouette_score(features_np, kmeans.labels_))
```

```python
with sns.axes_style("whitegrid"):
    disp = sns.relplot(x=range(2, 11), y=measures,  kind='line')
ax = disp.ax
ax.set_xlabel("k")
ax.set_ylabel("Silhouette")
ax.set_title("Silhouette Evolution")
```

It can be seen that it reduces when k increases, but it is important to find a k
value for which the reduction is important but at the same time k is lower. This
is important because a low k allow us to interpret better the results.


## DBSCAN

Another interesting clustering algorithm is DBSCAN. In many situations it can
make a more natural clustering.

<img
src="https://files.realpython.com/media/crescent_comparison.7938c8cf29d1.png"
width="50%"> 

From the scikit-learn User Guide: The DBSCAN algorithm views clusters as areas of high density separated by areas of low density. Due to this rather generic view, clusters found by DBSCAN can be any shape, as opposed to k-means which assumes that clusters are convex shaped. The central component to the DBSCAN is the concept of core samples, which are samples that are in areas of high density. A cluster is therefore a set of core samples, each close to each other (measured by some distance measure) and a set of non-core samples that are close to a core sample (but are not themselves core samples). There are two parameters to the algorithm, min_samples and eps, which define formally what we mean when we say dense. Higher min_samples or lower eps indicate higher density necessary to form a cluster.


To use it is very simple.

```python
from sklearn.cluster import DBSCAN
```

```python
dbscan = DBSCAN(eps=0.3) #  eps is the maximum distance between two samples for one to be considered as in the neighborhood of the other. 

dbscan.fit(features_np)
```

Now we are going to show them.

```python
plotpoints(dbscan.labels_)
```

In this case, it make only two clusters. We will see silhouette score:

```python
silhouette_score(features_np, dbscan.labels_)
```

With only two clusters, it has improve the measure.

```python
dbscan2 = DBSCAN(eps=0.25, min_samples=3)
dbscan2.fit(features_np)
plotpoints(dbscan2.labels_)
```

There are two parameters to the algorithm, *min_samples* and *eps*, which define
formally what we mean when we say dense. Higher *min_samples* or lower *eps*
indicate higher density necessary to form a cluster. *min_samples* allow
algorithm to be more robust against noise.


**Task**: Change eps and min_sample and observe the differences.


## Another algorithm: MeanShift, OPTICS

OPTICS is very similar to DBSCAN, but it presents a better behavior with large n_samples and large n_clusters. 

MeanShift clustering aims to discover blobs in a smooth density of samples. It
is a centroid based algorithm, which works by updating candidates for centroids
to be the mean of the points within a given region. These candidates are then
filtered in a post-processing stage to eliminate near-duplicates to form the
final set of centroids.

**MeanShift** requires a parameter *bandwidth* but it can be estimated through
function **estimate_bandwidth**.

```python
from sklearn.cluster import MeanShift, estimate_bandwidth
```

```python
# The following bandwidth can be automatically estimated
bandwidth = estimate_bandwidth(features_np, quantile=0.2, n_samples=500)

ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
ms.fit(features_np)
```

```python
plotpoints(ms.labels_)
```

```python
silhouette_score(features_np, ms.labels_)
```

```python
**Task**: Change quantile to observe the difference.
```

## Hierarchical clustering

In this type of clustering, instances closer are grouped in the same clusters,
creating many small clusters. Then, they are hierarchical grouping between them
in different levels.

In particular, **AgglomerativeClustering** performs a hierarchical clustering
using a bottom up approach: each observation starts in its own cluster, and
clusters are successively merged together.

The linkage criteria determines the metric used for the merge strategy:

- **ward** minimizes the sum of squared differences within all clusters. It is a variance-minimizing approach and in this sense is similar to the k-means objective function but tackled with an agglomerative hierarchical approach.

- **complete** linkage minimizes the maximum distance between observations of pairs of clusters.

- **average** linkage minimizes the average of the distances between all observations of pairs of clusters.

- **Single** linkage minimizes the distance between the closest observations of pairs of clusters.



```python
from sklearn.cluster import AgglomerativeClustering
```

```python
clustering = AgglomerativeClustering(linkage='ward', n_clusters=3).fit(features_np)

```

```python
plotpoints(clustering.labels_)
```

```python
silhouette_score(features_np, clustering.labels_)
```

We are going to show the hierarchy:

```python
from scipy.cluster.hierarchy import dendrogram
```

```python

def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)

```

```python
# distance_threshold implies to calculate all distance
clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=0).fit(features_np)
plt.figure(figsize=(100,100))
plot_dendrogram(clustering, p=3)
```
