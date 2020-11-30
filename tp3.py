"""
IFT799 - TP 3-4
2020-12-7
Gabriel Gibeau Sanchez - gibg2501
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, mean_squared_error


# Load movies ratings data
data = pd.read_csv('data/u.data', sep='\t',
                   names=['user id', 'item id', 'rating', 'timestamp'])

items = pd.read_csv('data/u.item', sep='|', encoding='latin_1',
                    names=['item id',
                           'movie title',
                           'release date',
                           'video release date',
                           'IMDb URL',
                           'unknown',
                           'Action',
                           'Adventure',
                           'Animation',
                           'Children\'s',
                           'Comedy',
                           'Crime',
                           'Documentary',
                           'Drama',
                           'Fantasy',
                           'Film - Noir',
                           'Horror',
                           'Musical',
                           'Mystery',
                           'Romance',
                           'Sci - Fi',
                           'Thriller',
                           'War',
                           'Western'])

test_sets = list()
base_sets = list()
SVDs = list()
trfrms = list()
KMns = list()
# Ks = list()
predictions = list()
max_centroids = 40
centroids = np.linspace(2, max_centroids, 1)

for i in range(0, 5):
    test_sets.append(pd.read_csv(f'data/u{i+1}.test', sep='\t',
                                 names=['user id', 'item id', 'rating', 'timestamp']))
    base_sets.append(pd.read_csv(f'data/u{i+1}.base', sep='\t',
                                 names=['user id', 'item id', 'rating', 'timestamp']))

    SVDs.append(TruncatedSVD(n_components=2))
    trfrms.append(SVDs[i].fit_transform(base_sets[i].loc[:, ['user id', 'item id', 'rating']]))


scores = list()
scores_means = list()
rmses = list()
rmses_means = list()
for j in centroids:
    KMns.append(KMeans(n_clusters=j))
    scores.clear()
    predictions.clear()
    for i in range(0, 5):
        KMns[j - 2].fit(trfrms[i])
        predictions.append(KMns[j-2].predict(SVDs[i].transform(test_sets[i].loc[:, ['user id', 'item id', 'rating']])))
        scores.append(silhouette_score(test_sets[i].loc[:, ['user id', 'item id', 'rating']], predictions[i],
                                       metric='euclidean'))
        # rmses.append()

    scores_means.append(np.array(scores).mean())
    # rmses_means.append(np.array(rmses).mean())

plt.plot(scores_means, centroids)
plt.savefig('fig.png')

# Re-arrange know ratings in a m users X n items matrix
pivot_data = data.loc[:, ['user id', 'item id', 'rating']].pivot(index = 'user id', columns=['item id'])

input("press any key to exit...")
