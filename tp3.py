"""
IFT799 - TP 3-4
2020-12-7
Gabriel Gibeau Sanchez - gibg2501
"""

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.preprocessing as prp

from tqdm import tqdm
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, mean_squared_error

genres = ['Action',
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
          'Western']

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
                           *genres])


def get_normalized_genres_means(genre_means:pd.DataFrame, normalization='naive'):
    if normalization == 'naive':
        genre_means[:] = genre_means[:] / 5
    return genre_means


def get_genres_means(data, items):
    df = pd.merge(left=data, right=items)
    df = df.drop(axis=1, labels=['timestamp',
                                 'release date',
                                 'IMDb URL',
                                 'unknown'])

    dt = dict()
    for genre in genres:
        dt[genre]=[df.loc[df[genre] == 1, 'rating'].mean()]
    return pd.DataFrame(dt)


def plot_genres_means_distribution(genre_means:pd.DataFrame):
    x = np.arange(len(genre_means.keys()))
    y_max = genre_means.iloc[:].max(axis=1).to_numpy()
    y_min = genre_means.iloc[:].min(axis=1).to_numpy()
    plt.bar(x, genre_means[:].to_numpy()[0], width=0.35)
    plt.xticks(ticks=x, labels=genre_means.keys(), rotation='vertical')
    plt.ylim([y_min - np.abs(np.log10(y_min)/10), y_max + np.abs(np.log10(y_max)/10)])
    # plt.savefig('images/{len(genre_means.keys())}_genres')
    plt.show()

df = get_genres_means(data, items)
plot_genres_means_distribution(df.loc[:, ['Action', 'Thriller', 'Western']])
plot_genres_means_distribution(df)
plot_genres_means_distribution(df.iloc[:, :10])
nmdf = get_normalized_genres_means(df)
plot_genres_means_distribution(nmdf.loc[:, ['Action', 'Thriller', 'Western']])

test_sets = list()
base_sets = list()
SVDs = list()
base_trfrms = list()
test_trfrms = list()
KMns = list()
# Ks = list()
predictions = list()
max_centroids = 22
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
    print(f'Using {j} centroids')
    KMns.append(KMeans(n_clusters=int(j),random_state=1))
    scores.clear()
    predictions.clear()

    for i in tqdm(range(5)):
        KMns[k_index].fit(base_trfrms[i])
        pr = KMns[k_index].predict(test_trfrms[i])
        pr2 = KMns[k_index].fit_predict(base_trfrms[i])
        predictions.append(pr)
        # plt.scatter(base_trfrms[i][:, 0], base_trfrms[i][:, 1], c=pr2)
        plt.scatter(test_sets[i].iloc[:, 0], test_trfrms[i][:, 1], c=pr)
        plt.savefig(f'images/{j}-centroids_set-{i}.png')
        plt.show()
        # scores.append(silhouette_score(base_sets[i].loc[:, ['user id', 'item id', 'rating']], predictions[i],
        #                                metric='euclidean'))
        scores.append(silhouette_score(test_trfrms[i], predictions[i],
                                       metric='euclidean'))

    scores_means.append(np.array(scores).mean())
    rmses_means
    k_index += 1

plt.plot(centroids, scores_means)
plt.savefig('images/fig.png')
plt.show()

# Re-arrange know ratings in a m users X n items matrix
pivot_data = data.loc[:, ['user id', 'item id', 'rating']].pivot(index = 'user id', columns=['item id'])

# def plot_pivot_data(data, elevation=0, azimut=0):
#     mpl.use('Qt5Agg')
#     fig = plt.figure()
#     ax = fig.gca(projection='3d', elev=elevation, azim=azimut)
#     # xx, yy = np.meshgrid(np.arange(0, data.shape[1], 1), np.arange(0, data.shape[0], 1))
#     xx, yy = np.meshgrid(np.arange(1, 11, 1), np.arange(1, 11, 1))
#     ax.scatter(xx, yy, data.iloc[:10, :10], marker='.')
#     # ax.axes.set_zlim3d(bottom=-10, top=10)
#     # ax.axes.set_ylim3d(bottom=-min(data.iloc[:, 0]), top=max(data.iloc[:, 0]))
#     # ax.axes.set(left=-10, right=10)
#     plt.show()
#
#
# elvs = [0, 90, 90, 0, 45]
# azms = [90, 0, 90, 0, 45]
# for elv, azm in zip(elvs, azms):
#     plot_pivot_data(pivot_data, elv, azm)


print("All done!")
