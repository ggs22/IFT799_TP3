"""
IFT799 - TP 3-4
2020-12-7
Gabriel Gibeau Sanchez - gibg2501
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import path
import pickle

from tqdm import tqdm
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples, mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize, MinMaxScaler

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

nb_datasets = 5

'''
Helper functions
'''


def get_distance_table(point: pd.Series, points_set: pd.DataFrame):
    return points_set.sub(point, axis=1).pow(2).sum(axis=1).pow(.5)


def get_genres_means(_data, normaliztion=None):
    df = _data.drop(axis=1, labels=['timestamp',
                                    'release date',
                                    'IMDb URL'])

    if normaliztion == "naive":
        df.loc[:, 'rating'] = df.loc[:, 'rating'] / 5
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
    plt.ylim([y_min - np.abs(np.log10(y_min) / 10), y_max + np.abs(np.log10(y_max) / 10)])
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


'''
K-Mean-based method (model 1)
'''

# Get the average rating for each film genre, for each user
genre_mean_per_user = get_genres_mean_per_user(data)
print(genre_mean_per_user.head().to_string())

test_sets = list()
base_sets = list()
SVDs = list()
base_trfrms = list()
test_trfrms = list()

# Create base and tests sets, mergin with movie information (info from u.item)
for i in range(0, nb_datasets):
    _df = pd.read_csv(f'data/u{i + 1}.base', sep='\t', names=['user id', 'item id', 'rating', 'timestamp'])
    base_sets.append(pd.merge(_df, items, on=['item id']))
    _df = pd.read_csv(f'data/u{i + 1}.test', sep='\t', names=['user id', 'item id', 'rating', 'timestamp'])
    test_sets.append(pd.merge(_df, items, on=['item id']))

    _base_gmean_u = get_genres_mean_per_user(base_sets[i])
    _test_gmean_u = get_genres_mean_per_user(test_sets[i])

    SVDs.append(TruncatedSVD(n_components=8))
    base_trfrms.append(SVDs[i].fit_transform(_base_gmean_u))
    test_trfrms.append(SVDs[i].fit_transform(_test_gmean_u))

# # If it hasn't been done, perform elbow method to determine optimal cluster number
# if not path.Path('images/elbow_method.png').exists():
#     centroids = range(4, 9, 1) # lets use 10 cluster number values
#     scores = list()
#     scores_means = list()
#     k_index = 0
#
#
#     for j in tqdm(centroids):
#         print(f'Using {j} centroids')
#         kmn = KMeans(n_clusters=int(j), random_state=1)
#         scores.clear()
#
#         for i in tqdm(range(nb_datasets)):
#             pr = kmn.fit_predict(base_trfrms[i])
#             samples = silhouette_samples(base_trfrms[i], pr)
#             scores.append(silhouette_score(base_trfrms[i], pr, metric='euclidean'))
#             y_low = 10
#             for k in range(j):
#                 cluster_k_silhouette_values = samples[pr == k]
#                 cluster_k_silhouette_values.sort()
#                 cluser_k_size = cluster_k_silhouette_values.shape[0]
#                 y_high = y_low + cluser_k_size
#
#                 color = cm.nipy_spectral(float(k)/ j)
#                 plt.fill_betweenx(np.arange(y_low, y_high), 0, cluster_k_silhouette_values,
#                                   facecolor=color, edgecolor=color, alpha=0.8)
#                 plt.text(-.2, y_low + .5 * cluser_k_size, str(k))
#                 plt.vlines(scores[i], ymin=0, ymax=samples.shape[0], color="blue", linestyles="--")
#                 y_low = y_high + 10
#
#             # Plot silhouette scores
#             plt.xlim([-.15, .45])
#             plt.ylim([0, samples.shape[0]])
#             plt.xlabel('silhouette score')
#             plt.yticks(ticks=[])
#             plt.savefig(f'images/silhoutte-samples_centroids-{j}_dataset-{i}.png')
#             plt.show()
#
#         k_index += 1
#         scores_means.append(np.array(scores).mean())
#
#     # Plot elbow curve
#     plt.plot(centroids, scores_means)
#     plt.xticks(centroids)
#     plt.savefig(f'images/elbow_method.png')
#     plt.show()
#
# # Dtermined with the elbow method (see elbow graph)
# nb_clusters = 5
#
# rmse_per_cluster = pd.DataFrame()
# rmse_discrete_per_cluster = pd.DataFrame()
# rmse_rdn_per_cluster = pd.DataFrame()
#
# rmse = list()
# rmse_discrete = list()
# rmse_rdn = list()
# cols = list()
# rows = list()
#
# # For each data set
# for i in tqdm(range(nb_datasets), desc='Datasets'):
#
#     # Make a kmean with optimal number of clusters
#     kmn = KMeans(n_clusters=nb_clusters)
#     # get centroids from transformed base data
#     base_prediction = kmn.fit_predict(base_trfrms[i])
#     # get prediction on transformed data
#     test_prediction = kmn.predict(test_trfrms[i])
#
#     # join prediction on transformed data with original data (base & test data)
#     joint_base_predictions = pd.DataFrame(base_prediction, columns=['cluster'])
#     joint_base_predictions['user id'] = range(1, len(base_prediction) + 1)
#     joint_base_predictions = base_sets[i].join(joint_base_predictions.set_index('user id'), on='user id')
#
#     joint_test_prediction = pd.DataFrame(test_prediction, columns=['cluster'])
#     joint_test_prediction['item id'] = range(1, len(test_prediction) + 1)
#     joint_test_prediction = test_sets[i].join(joint_test_prediction.set_index('item id'), on='item id')
#
#     rmse.clear()
#     rmse_discrete.clear()
#     rmse_rdn.clear()
#
#     cols += [f'dataset-{i}']
#
#     # For each cluster
#     for j in tqdm(range(nb_clusters), desc='Clusters'):
#         rows += [f'cluster-{j}']
#
#         # Get the mean of each movie for current cluster
#         movies_mean = joint_base_predictions.loc[joint_base_predictions['cluster'] == j,
#                                              ['item id', 'rating']].groupby(['item id']).mean()
#         movies_mean['item id'] = range(1, movies_mean.shape[0]+1)
#         movies_mean['predicted rating'] = movies_mean['rating']
#         movies_mean['discrete predicted rating'] = movies_mean['rating'].round()
#         movies_mean.drop(labels=['rating'], axis=1, inplace=True)
#
#         # Join a data frame containing rating prediction & discrete rating prediction & actual rating
#         movie_predictions = \
#             joint_test_prediction[joint_test_prediction['cluster'] == j].join(movies_mean.set_index('item id'),
#                                                                               on='item id')
#
#         # Get RMSE between prediction an actual ratings in current cluster
#         rmse.append(np.sqrt(((movie_predictions['rating'] - movie_predictions['predicted rating'])**2).mean()))
#         # Get RMSE between discrete prediction an actual ratings in current cluster
#         rmse_discrete.append(np.sqrt(((movie_predictions['rating'] - movie_predictions['discrete predicted rating']) ** 2).mean()))
#         # Get random vector of 5 values to compare with clustering algorithme
#         rmse_rdn.append(np.sqrt(((movie_predictions['rating'] - np.random.randint(1, 6, movie_predictions.shape[0]))**2). mean()))
#
#     rmse_per_cluster = pd.concat([rmse_per_cluster, pd.Series(rmse)], axis=1)
#     rmse_discrete_per_cluster = pd.concat([rmse_discrete_per_cluster, pd.Series(rmse_discrete)], axis=1)
#     rmse_rdn_per_cluster = pd.concat([rmse_rdn_per_cluster, pd.Series(rmse_rdn)], axis=1)
#
# rmse_per_cluster.columns = cols
# rmse_discrete_per_cluster.columns = cols
# rmse_rdn_per_cluster.columns = cols
#
# rmse_per_cluster.index = keys=rows[:nb_clusters]
# rmse_discrete_per_cluster.index = keys=rows[:nb_clusters]
# rmse_rdn_per_cluster.index = keys=rows[:nb_clusters]
#
# # Plot repectives RMSEs
# sns.heatmap(rmse_per_cluster, vmin=0.5, vmax=2.5, cmap='Reds', annot=True)
# plt.title('Predictions RMSE')
# plt.savefig('images/Predictions_RMSE')
# plt.show()
#
# sns.heatmap(rmse_discrete_per_cluster, vmin=0.5, vmax=2.5, cmap='Reds', annot=True)
# plt.title('Discrete predictions RMSE')
# plt.savefig('images/Discrete_predictions_RMSE')
# plt.show()
#
# sns.heatmap(rmse_rdn_per_cluster, vmin=0.5, vmax=2.5, cmap='Reds', annot=True)
# plt.title('random vector RMSE')
# plt.savefig('images/random_vector_RMSE')
# plt.show()
#
# # Plot random vs discrete prediction RMSE
# sns.barplot(data=rmse_rdn_per_cluster, color='r')
# sns.barplot(data=rmse_discrete_per_cluster, color='g')
# plt.title('Random vector RMSE vs Prediction RMSE')
# plt.savefig('images/Random_vector_RMSE_vs_Prediction_RMSE.png')
# plt.show()


'''
Collaborative filtering method (model 4)
User-based classification
'''

# Defin numbers of nearest neighbors
k_nn = 25

# Re-arrange know ratings in a m users X n items matrix
pivot_data = data.loc[:, ['user id', 'item id', 'rating']].pivot(index='user id', columns=['item id'])
pivot_data = pivot_data.replace(np.nan, 0)

pivot_data_base = list()
pivot_data_test = list()

for i in range(0, nb_datasets):
    pivot_data_base.append(base_sets[i].loc[:, ['user id', 'item id', 'rating']].pivot(index='user id',
                                                                                       columns=['item id']))
    pivot_data_test.append(test_sets[i].loc[:, ['user id', 'item id', 'rating']].pivot(index='user id',
                                                                                       columns=['item id']))

    pivot_data_base[i] = pivot_data_base[i].replace(np.nan, 0)
    pivot_data_test[i] = pivot_data_test[i].replace(np.nan, 0)

distance_tables = list()
similarity_user_vs_nns = list()
nearest_neighbors = list()

pivot_data_original = pivot_data.copy()

scaler = MinMaxScaler()
tf = scaler.fit_transform(pivot_data.transpose()[pivot_data.transpose().columns])
pivot_data = pd.DataFrame(tf.transpose(),
                          index=pivot_data.index, columns=pivot_data.columns)

# Build distances table for each bast data set
for i in tqdm(range(0, nb_datasets), desc='Data Sets'):

    # Serialize df object fo re-use
    fname_dist_table = f'norm_dist_table_{i}.pikcle'
    fname_knn_table  = f'knn_table_{i}.pikcle'

    # Normalize ratings user-wise
    scaler = MinMaxScaler()
    tf = scaler.fit_transform(pivot_data_base[i].transpose()[pivot_data_base[i].transpose().columns])
    pivot_data_base[i] = pd.DataFrame(tf.transpose(),
                                      index=pivot_data_base[i].index, columns=pivot_data_base[i].columns)

    scaler = MinMaxScaler()
    tf = scaler.fit_transform(pivot_data_test[i].transpose()[pivot_data_test[i].transpose().columns])
    pivot_data_test[i] = pd.DataFrame(tf.transpose(),
                                      index=pivot_data_test[i].index, columns=pivot_data_test[i].columns)

    # If no serialized table has been found then...
    if not path.Path(fname_dist_table).exists():

        cols = list()
        distances_between_users = pd.DataFrame()
        nearest_neighbors_per_user = pd.DataFrame()

        # For each test user, compute normalized distance with base users
        for index, row in tqdm(pivot_data_test[i].iterrows()):
            dist = get_distance_table(row, pivot_data_base[i])
            cols += [f'test user {index}']
            distances_between_users = pd.concat([distances_between_users, dist], axis=1)

            # Get only userid (as index) of base nearest neibghbors in a dataframe, except the point itself
            nearest_neighbors_per_user = pd.concat([nearest_neighbors_per_user,
                                                    pd.Series(dist.sort_values().iloc[:k_nn].index)], axis=1)

        # Save distance table for re-use
        f_dist_table = open(fname_dist_table, 'wb')
        pickle.dump(distances_between_users, f_dist_table, pickle.HIGHEST_PROTOCOL)
        distance_tables.append(distances_between_users)

        # Save k nearest neihbots table for re-use
        f_knn_table = open(fname_knn_table, 'wb')
        pickle.dump(nearest_neighbors_per_user, f_knn_table, pickle.HIGHEST_PROTOCOL)
        nearest_neighbors.append(nearest_neighbors_per_user)

    else:
        # Load distance table for re-use if available
        print(f'Loading distance table {i}...')
        f_dist_table = open(fname_dist_table, 'rb')
        distances_between_users = pickle.load(f_dist_table)
        distance_tables.append(distances_between_users)

        print(f'Loading knn table {i}...')
        f_dist_table = open(fname_knn_table, 'rb')
        nearest_neighbors_per_user = pickle.load(f_dist_table)
        nearest_neighbors.append(nearest_neighbors_per_user)

    # Show a sample of a distance table
    print(distances_between_users.head().to_string)

    # Test users as rows, nearest neighbors as columns
    nearest_neighbors[i] = nearest_neighbors[i].transpose()
    nearest_neighbors[i].index = pivot_data_test[i].index

    _df = pivot_data_base[i].transpose()
    s = _df.index.to_numpy()
    _df.insert(0, 'rating',s , allow_duplicates=False)

    mrgr = pd.merge(pivot_data_test[i].transpose(), pivot_data_base[i].transpose(),
                    left_index=True, right_index=True, how='inner')

    pivot_data_base[i].columns = pivot_data_base[i].columns.droplevel()
    pivot_data_test[i].columns = pivot_data_test[i].columns.droplevel()

    # Merge normalized base and test data to be able to compute distance & similarity
    merged_data = pd.merge(pivot_data_base[i], pivot_data_test[i], left_index=True, right_index=True, how='outer',
                           suffixes=('_base', '_test')).dropna(axis=1)

    # for each of the nearest neighbors, append test user and get similarity
    for index, row in nearest_neighbors[i].iterrows():
        m = pd.concat([merged_data.iloc[row, :].T, merged_data.iloc[index, :]], axis=1)
        s = pd.DataFrame(cosine_similarity(m.T), index=[*row, index], columns=[*row, index])
        # get weights from similiraty, drop the last element as it is test user
        w = s.iloc[:-1:, -1:]

        weighted_votes = pd.DataFrame()
        for i in range(1, 6):
            weighted_votes = pd.concat([weighted_votes,
                                        (m.iloc[:, :-1:] == (i/5)).multiply(w.T.to_numpy()).sum(axis=1)],
                                       axis=1)
        weighted_votes.columns = range(1,6)
        # Here we make the predication based on the weighted votes of knns.
        # The prediction is "de-normalized" by the face that column index correspond to
        # original rating
        predictions = weighted_votes.idxmax(axis=1)

        rmse = np.sqrt(mean_squared_error(predictions, merged_data.iloc[index, :]))

print("All done!")
