"""
IFT799 - TP 3-4
2020-12-7
Gabriel Gibeau Sanchez - gibg2501
"""

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import path

from tqdm import tqdm
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples, mean_squared_error

# Load genres names
genres = pd.read_csv('data/u.genre', sep='|').iloc[:, 0]

# Load movies ratings
data = pd.read_csv('data/u.data', sep='\t',
                   names=['user id', 'item id', 'rating', 'timestamp'])

# Load movie titles with genres
items = pd.read_csv('data/u.item', sep='|', encoding='latin_1',
                    names=['item id',
                           'movie title',
                           'release date',
                           'video release date',
                           'IMDb URL',
                           'unknown',
                           *genres])

data = pd.merge(left=data, right=items)
data = pd.merge(left=data, right=items).drop(labels=['unknown'], axis=1)


def get_genres_means(_data, normaliztion=None):
    df = _data.drop(axis=1, labels=['timestamp',
                                 'release date',
                                 'IMDb URL'])

    if normaliztion == "naive":
        df.loc[:, 'rating'] = df.loc[:, 'rating']/5
    if normaliztion == "weighted":
        counts = get_genres_count(_data)
        w = counts / (counts.sum())
        for index, row in df.iterrows():
            row['rating'] = row['rating'] * (w[row[genres] == 1]).sum()

    df_res = pd.DataFrame()
    for genre in genres:
        df_res[genre] = [df.loc[df[genre] == 1, 'rating'].mean()]
    return df_res.T


def get_genres_count(_data):
    df = _data.drop(axis=1, labels=['timestamp',
                                 'release date',
                                 'IMDb URL'])

    df_res = pd.DataFrame(df.loc[:, genres].sum(axis=0))
    return df_res


def plot_genres_means_distribution(genre_means: pd.DataFrame):
    x = np.arange(len(genre_means.index))
    y_max = genre_means.iloc[:].max(axis=0).to_numpy()
    y_min = genre_means.iloc[:].min(axis=0).to_numpy()
    plt.bar(x=x, height=genre_means[:][0])
    plt.xticks(ticks=x, labels=genre_means.index, rotation='vertical')
    plt.ylim([y_min - np.abs(np.log10(y_min)/10), y_max + np.abs(np.log10(y_max)/10)])
    plt.show()


def plot_test_transformation_scatter(set_index, predict):
    plt.scatter(test_sets[set_index].iloc[:, 0], test_trfrms[set_index][:, 1], c=predict)
    plt.savefig(f'images/{np.unique(predict[0])}-centroids_set-{set_index}.png')
    plt.show()


def get_genres_mean_per_user(_data):
    res = pd.DataFrame()
    for genre in genres:
        res = pd.concat([res, _data.loc[_data[genre] == 1, ['user id', 'rating']].groupby(['user id']).mean()],
                        axis=1, ignore_index=False)
    res.columns = genres

    # if there is no averge for a given genre (nan), then we suppose there is no interest in that genre
    res = res.replace(np.nan, 0)
    return res


# genres_means = get_genres_means(data)
# plot_genres_means_distribution(genres_means)
#
# nmdf = genres_means = get_genres_means(data, normaliztion='naive')
# plot_genres_means_distribution(nmdf)
#
# genres_counts = get_genres_count(data)
# plot_genres_means_distribution(genres_counts)
#
# # Re-arrange know ratings in a m users X n items matrix
# pivot_data = data.loc[:, ['user id', 'item id', 'rating']].pivot(index='user id', columns=['item id'])
#
# # Plot some heat maps of the known rating
# sns.heatmap(pivot_data, cmap='autumn')
# plt.show()

# # Get the average rating for each film genre, for each user
# genre_mean_per_user = get_genres_mean_per_user(data)
#
# # Plot genre mean rating per user
# sns.heatmap(genre_mean_per_user, cmap='Greens')
# plt.show()

# for i in range(5):
#     sns.heatmap(pivot_data[pivot_data['rating'] == i+1], cmap='twilight')
#     plt.show()


# for i in range(5):
#     for idx, genre in enumerate(genres):
#         plt.scatter(data[(data['rating'] == i+1) & (data[genre] == 1)].loc[:, 'item id'],
#                     data[(data['rating'] == i+1) & (data[genre] == 1)].loc[:, 'user id'],
#                     marker='.',
#                     s=0.5,
#                     color=cm.nipy_spectral(float(idx)/len(genres)))
#     plt.title(f'rating:{i}')
#     plt.legend(genres)
#     plt.savefig(f'images/rating:{i}')


# Load split data and make SVD transformations
test_sets = list()
base_sets = list()
SVDs = list()
base_trfrms = list()
test_trfrms = list()

# Create base and tests sets, mergin with movie information (info from u.item)
for i in range(0, 5):
    _df = pd.read_csv(f'data/u{i+1}.base', sep='\t', names=['user id', 'item id', 'rating', 'timestamp'])
    base_sets.append(pd.merge(_df, items, on=['item id']))
    _df = pd.read_csv(f'data/u{i+1}.test', sep='\t', names=['user id', 'item id', 'rating', 'timestamp'])
    test_sets.append(pd.merge(_df, items, on=['item id']))

    _base_gmean_u = get_genres_mean_per_user(base_sets[i])
    _test_gmean_u = get_genres_mean_per_user(test_sets[i])

    SVDs.append(TruncatedSVD(n_components=8))
    base_trfrms.append(SVDs[i].fit_transform(_base_gmean_u))
    test_trfrms.append(SVDs[i].fit_transform(_test_gmean_u))

if not path.Path('images/elbow_method.png').exists():
    centroids = range(4, 9, 1) # lets use 10 cluster number values
    scores = list()
    scores_means = list()
    k_index = 0


    for j in tqdm(centroids):
        print(f'Using {j} centroids')
        kmn = KMeans(n_clusters=int(j), random_state=1)
        scores.clear()

        for i in tqdm(range(5)):
            pr = kmn.fit_predict(base_trfrms[i])
            samples = silhouette_samples(base_trfrms[i], pr)
            scores.append(silhouette_score(base_trfrms[i], pr, metric='euclidean'))
            y_low = 10
            for k in range(j):
                cluster_k_silhouette_values = samples[pr == k]
                cluster_k_silhouette_values.sort()
                cluser_k_size = cluster_k_silhouette_values.shape[0]
                y_high = y_low + cluser_k_size

                color = cm.nipy_spectral(float(k)/ j)
                plt.fill_betweenx(np.arange(y_low, y_high), 0, cluster_k_silhouette_values,
                                  facecolor=color, edgecolor=color, alpha=0.8)
                plt.text(-.2, y_low + .5 * cluser_k_size, str(k))
                plt.vlines(scores[i], ymin=0, ymax=samples.shape[0], color="blue", linestyles="--")
                y_low = y_high + 10

            # Plot silhouette scores
            plt.xlim([-.15, .45])
            plt.ylim([0, samples.shape[0]])
            plt.xlabel('silhouette score')
            plt.yticks(ticks=[])
            plt.savefig(f'images/silhoutte-samples_centroids-{j}_dataset-{i}.png')
            plt.show()

        k_index += 1
        scores_means.append(np.array(scores).mean())

    # Plot silhouette scores
    plt.plot(centroids, scores_means)
    plt.xticks(centroids)
    plt.savefig(f'images/elbow_method.png')
    plt.show()


# Dtermined with the elbow method
nb_clusters = 5
nb_datasets = 5

cols = list()
rows = list()

rmse_per_cluster = pd.DataFrame()
rmse_rdn_per_cluster = pd.DataFrame()
rmse = list()
rmse_rdn = list()
for i in tqdm(range(nb_datasets), desc='Datasets'):

    cols += [f'dataset-{i}']
    kmn = KMeans(n_clusters=nb_clusters)

    # get centroids
    base_prediction = kmn.fit_predict(base_trfrms[i])
    # get prediction on transformed data
    test_prediction = kmn.predict(test_trfrms[i])

    # join prediction on transformed data with original data
    joint_base_predictions = pd.DataFrame(base_prediction, columns=['cluster'])
    joint_base_predictions['user id'] = range(1, len(base_prediction) + 1)
    joint_base_predictions = base_sets[i].join(joint_base_predictions.set_index('user id'), on='user id')

    joint_test_prediction = pd.DataFrame(test_prediction, columns=['cluster'])
    joint_test_prediction['item id'] = range(1, len(test_prediction) + 1)
    joint_test_prediction = test_sets[i].join(joint_test_prediction.set_index('item id'), on='item id')

    rmse.clear()
    rmse_rdn.clear()
    # use centroids to make predictions on test set
    for j in tqdm(range(nb_clusters), desc='Clusters'):
        rows += [f'cluster-{j}']
        movies_mean = joint_base_predictions.loc[joint_base_predictions['cluster'] == j,
                                             ['item id', 'rating']].groupby(['item id']).mean()
        movies_mean['item id'] = range(1, movies_mean.shape[0]+1)
        movies_mean['predicted rating'] = movies_mean['rating'].round()
        movies_mean.drop(labels=['rating'], axis=1, inplace=True)

        # Create a data frame containing rating average for current cluster
        movie_predictions = \
            joint_test_prediction[joint_test_prediction['cluster'] == j].join(movies_mean.set_index('item id'),
                                                                              on='item id')

        # Get RMSE between prediction an actual ratings in current cluster
        rmse.append(np.sqrt(((movie_predictions['rating'] - movie_predictions['predicted rating'])**2).mean()))
        # Get random vector of 5 values to compare with clustering algorithme
        rmse_rdn.append(np.sqrt(((movie_predictions['rating'] - np.random.randint(1, 6, movie_predictions.shape[0]))**2). mean()))
    rmse_per_cluster = pd.concat([rmse_per_cluster, pd.Series(rmse)], axis=1)
    rmse_rdn_per_cluster = pd.concat([rmse_rdn_per_cluster, pd.Series(rmse_rdn)], axis=1)

rmse_per_cluster.columns = cols
rmse_rdn_per_cluster.columns = cols
rmse_per_cluster.index = keys=rows[:nb_clusters]
rmse_rdn_per_cluster.index = keys=rows[:nb_clusters]

sns.heatmap(rmse_per_cluster, vmin=1, vmax=2)
plt.show()

sns.heatmap(rmse_rdn_per_cluster, vmin=1, vmax=2)
plt.show()

# sns.heatmap(rmse_rdn_per_cluster)
# plt.show()


print("All done!")

