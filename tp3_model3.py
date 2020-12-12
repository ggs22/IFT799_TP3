"""
IFT799 - TP 3-4
2020-12-11
Gabriel Gibeau Sanchez - gibg2501
Modele 3
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm
from sklearn.cluster import KMeans

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

    base_trfrms.append(_base_gmean_u)
    test_trfrms.append(_test_gmean_u)

# Dtermined with the elbow method (see elbow graph)
nb_clusters = 5
iterations = 100

rmse_per_cluster = pd.DataFrame()
rmse_discrete_per_cluster = pd.DataFrame()
rmse_rdn_per_cluster = pd.DataFrame()

rmse = list()
rmse_discrete = list()
rmse_rdn = list()
cols = list()
rows = list()

# For each data set
for i in tqdm(range(nb_datasets), desc='Datasets'):

    # create weights matrix, (C=1)
    w = np.ones((nb_clusters, len(genres)))
    simga = np.ones((nb_clusters, len(genres)))

    # Make a kmean with optimal number of clusters
    kmn = KMeans(n_clusters=nb_clusters)
    # get centroids from transformed base data
    base_pr = kmn.fit_predict(base_trfrms[i])

    # get dimensions variances for each cluster
    for k in range(0, kmn.n_clusters):
        simga[k] = base_trfrms[i].iloc[base_pr == k, :].var(axis=0)

    for _ in range(0, iterations):
        # Calculate weighted euclidian distance
        dist = pd.DataFrame()
        for k in range(0, kmn.n_clusters):
            _res = (test_trfrms[i].sub(kmn.cluster_centers_[k, :]))**2
            _res = _res.multiply(w[k]).sum(axis=1)
            _res = np.sqrt(_res)
            dist = pd.concat([dist, _res], axis=1)
        dist.index = test_trfrms[i].index
        dist.columns = range(1, nb_clusters + 1)

        # Predict
        pr = dist.idxmax(axis=1) - 1

        # Update weights
        for k in range(0, kmn.n_clusters):
            w[k] = np.divide(w[k], (simga[k] + 1))
            w[k] = np.divide(w[k], np.sqrt(np.sum(np.power(w[k], 2))))

    # join prediction on transformed data with original data (base & test data)
    joint_base_predictions = pd.DataFrame(base_pr, columns=['cluster'])
    joint_base_predictions['user id'] = range(1, len(base_pr) + 1)
    joint_base_predictions = base_sets[i].join(joint_base_predictions.set_index('user id'), on='user id')

    joint_test_prediction = pd.DataFrame(pr, columns=['cluster'])
    joint_test_prediction.index = test_trfrms[i].index
    joint_test_prediction = pd.merge(test_sets[i], joint_test_prediction, left_on='user id', right_index=True)

    rmse.clear()
    rmse_discrete.clear()
    rmse_rdn.clear()

    cols += [f'dataset-{i}']

    # For each cluster
    for j in tqdm(range(nb_clusters), desc='Clusters'):
        rows += [f'cluster-{j}']

        # Get the mean of each movie for current cluster
        movies_mean = joint_base_predictions.loc[joint_base_predictions['cluster'] == j,
                                             ['item id', 'rating']].groupby(['item id']).mean()
        # movies_mean['item id'] = range(1, movies_mean.shape[0]+1)
        movies_mean['predicted rating'] = movies_mean['rating']
        movies_mean['discrete predicted rating'] = movies_mean['rating'].round()
        movies_mean.drop(labels=['rating'], axis=1, inplace=True)

        # Join a data frame containing rating prediction & discrete rating prediction & actual rating
        movie_predictions = pd.merge(joint_test_prediction[joint_test_prediction['cluster'] == j], movies_mean,
                                     left_on='item id', right_index=True)

        # Get RMSE between prediction an actual ratings in current cluster
        rmse.append(np.sqrt(((movie_predictions['rating'] - movie_predictions['predicted rating'])**2).mean()))
        # Get RMSE between discrete prediction an actual ratings in current cluster
        rmse_discrete.append(np.sqrt(((movie_predictions['rating'] - movie_predictions['discrete predicted rating']) ** 2).mean()))
        # Get random vector of 5 values to compare with clustering algorithme
        rmse_rdn.append(np.sqrt(((movie_predictions['rating'] - np.random.randint(1, 6, movie_predictions.shape[0]))**2). mean()))

    rmse_per_cluster = pd.concat([rmse_per_cluster, pd.Series(rmse)], axis=1)
    rmse_discrete_per_cluster = pd.concat([rmse_discrete_per_cluster, pd.Series(rmse_discrete)], axis=1)
    rmse_rdn_per_cluster = pd.concat([rmse_rdn_per_cluster, pd.Series(rmse_rdn)], axis=1)

rmse_per_cluster.columns = cols
rmse_discrete_per_cluster.columns = cols
rmse_rdn_per_cluster.columns = cols

rmse_per_cluster.index = rows[:nb_clusters]
rmse_discrete_per_cluster.index = rows[:nb_clusters]
rmse_rdn_per_cluster.index = rows[:nb_clusters]

# Plot repectives RMSEs
sns.heatmap(rmse_per_cluster, vmin=0.5, vmax=2.5, cmap='Reds', annot=True)
plt.title('Predictions RMSE')
plt.savefig('images/model3/Predictions_RMSE')
plt.show()

sns.heatmap(rmse_discrete_per_cluster, vmin=0.5, vmax=2.5, cmap='Reds', annot=True)
plt.title('Discrete predictions RMSE')
plt.savefig('images/model3/Discrete_predictions_RMSE')
plt.show()

sns.heatmap(rmse_rdn_per_cluster, vmin=0.5, vmax=2.5, cmap='Reds', annot=True)
plt.title('random vector RMSE')
plt.savefig('images/model3/random_vector_RMSE')
plt.show()

# Plot random vs discrete prediction RMSE
sns.barplot(data=rmse_rdn_per_cluster, color='r')
sns.barplot(data=rmse_discrete_per_cluster, color='g')
plt.title('Random vector RMSE vs KMean prediction RMSE')
plt.ylim([0, 2.4])
plt.ylabel("RMSE")
plt.savefig('images/model3/Random_vector_RMSE_vs_Prediction_RMSE.png')
plt.show()