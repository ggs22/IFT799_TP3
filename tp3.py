"""
IFT799 - TP 3-4
2020-12-7
Gabriel Gibeau Sanchez - gibg2501
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm
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
base_trfrms = list()
test_trfrms = list()
KMns = list()
# Ks = list()
predictions = list()
max_centroids = 3
centroids = range(2, max_centroids, 1)

for i in range(0, 5):
    test_sets.append(pd.read_csv(f'data/u{i+1}.test', sep='\t',
                                 names=['user id', 'item id', 'rating', 'timestamp']))
    base_sets.append(pd.read_csv(f'data/u{i+1}.base', sep='\t',
                                 names=['user id', 'item id', 'rating', 'timestamp']))

    SVDs.append(TruncatedSVD(n_components=2))
    base_trfrms.append(SVDs[i].fit_transform(base_sets[i].loc[:, ['user id', 'item id', 'rating']]))
    test_trfrms.append(SVDs[i].fit_transform(test_sets[i].loc[:, ['user id', 'item id', 'rating']]))

scores = list()
scores_means = list()
rmses = list()
rmses_means = list()
k_index = 0

for j in tqdm(centroids):
    KMns.append(KMeans(n_clusters=int(j)))
    scores.clear()
    predictions.clear()

    KMns[k_index].fit(base_trfrms[0])
    predictions.append(KMns[k_index].predict(test_trfrms[0]))
    scores.append(silhouette_score(test_sets[0].loc[:, ['user id', 'item id', 'rating']], predictions[0],
                                   metric='euclidean'))

    # for i in tqdm(range(5)):
    #     KMns[k_index].fit(base_trfrms[i])
    #     predictions.append(KMns[k_index].predict(test_trfrms[i]))
    #     # scores.append(silhouette_score(test_sets[i].loc[:, ['user id', 'item id', 'rating']], predictions[i],
    #     #                                metric='euclidean'))
    #     rmses.append(mean_squared_error(test_trfrms[i]))

    scores_means.append(np.array(scores).mean())
    rmses_means
    k_index += 1

plt.plot(centroids, scores_means)
plt.savefig('fig.png')

# Re-arrange know ratings in a m users X n items matrix
pivot_data = data.loc[:, ['user id', 'item id', 'rating']].pivot(index = 'user id', columns=['item id'])

print("All done!")
