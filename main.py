# Data Cleaning
# 1 Feature Engineering to eliminate negative values
# 2 Handling missing data
# 3 Handling categorical data
# Data Transformation
# 1 Handling skwed data with Log transform
# 2 Centering data with standarditzation
# Data Deorrelation and Dimensionality reduction
# 1 PCA
# Data ReStandardtization
# Clustering
# 1 Kmeans
# Evaluation
# 1 Within cluster sum of squared errors
# Visualization
# 1 Clusters in 2D plane

import pandas as pd
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.decomposition import PCA
import sys
import pickle

font = {'family': 'serif',
        'weight': 'bold',
        'size': 15}

matplotlib.rc('font', **font)

#plt.style.use('fivethirtyeight')
plt.style.use('fast')

# Change this
continues_cols = []
categorical_cols = []

def data_cleaning(file_name):
    data = pd.read_csv(file_name)
    # Note change this function
    return data, data


def data_transformation(data):
    # Limit the number of columns for transformation
    # data = data.iloc[:, 0:36]

    data_log = data.copy(deep=True)

    for col in continues_cols:
        data_log.loc[:, col] = np.log(data_log.loc[:, col] + 1)

    scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
    scaler.fit(data_log)
    data_normalized = scaler.transform(data_log)
    print('Data is normalized')
    print('mean: ', data_normalized.mean(axis=0).round(2))
    print('std: ', data_normalized.std(axis=0).round(2))
    data_normalized.dump('processed_data/data_transformed.csv')

    return data_normalized


def decorrelate_reduce(data, col_names):
    pca = PCA()
    decorr_data = pca.fit_transform(data)
    print(decorr_data.shape)
    print('Data is normalized')
    print('mean: ', decorr_data.mean(axis=0).round(2))
    print('std: ', decorr_data.std(axis=0).round(2))
    decorr_data.dump('processed_data/data_decorrelated.csv')
    out = pd.DataFrame(pca.components_[0:2, ], columns=col_names[0:75], index=['PC-1', 'PC-2'])
    out.dump('pca_features.csv')

    return decorr_data

def data_re_transformation(data):
    scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
    scaler.fit(data)
    data_normalized = scaler.transform(data)
    print('Data is normalized')
    print('mean: ', data_normalized.mean(axis=0).round(2))
    print('std: ', data_normalized.std(axis=0).round(2))
    data_normalized.dump('processed_data/data_re_transformed.csv')

    return data_normalized


def visualize_distributions_for_skewness(data):
    for col in data.columns:
        print(col)
        sns_plot = sns.distplot(data.loc[:, col], bins=20)
        fig = sns_plot.get_figure()
        fig.savefig('batch_figures/' + col + ".png")
        fig.clear()

        frequency_log = np.log(data.loc[:, col] + 1)
        log_plot = sns.distplot(frequency_log)
        log_fig = log_plot.get_figure()
        log_fig.savefig('batch_figures/log' + col + ".png")
        plt.show()


def visualize_pca_variance(features, data):
    # Check for pca variance
    pca = PCA()
    pca.fit_transform(data)

    plt.bar(range(1,len(pca.explained_variance_ratio_)+1), pca.explained_variance_ratio_)
    plt.xticks()
    plt.ylabel('variance ratio')
    plt.xlabel('PCA feature')
    plt.tight_layout()
    plt.savefig('figures/pca_variance_ratio.png')
    plt.show()

    plt.bar(range(1, len(pca.explained_variance_ratio_)+1), pca.explained_variance_ratio_.cumsum())
    plt.xticks()
    plt.ylabel('cumulative sum of variances')
    plt.xlabel('PCA feature')
    plt.tight_layout()
    plt.savefig('figures/pca_variance_ratio_cumsum.png')
    plt.show()

    return pca


def visualize_feature_corr(data):
    corr_metrics = data.corr()
    corr_metrics.style.background_gradient()


def visualize_clusters_in_2D(data, num_components=2, k=3):
    # Visualize the results on PCA-reduced data

    reduced_data_high_dim = PCA().fit_transform(data)
    reduced_data = PCA(n_components=num_components).fit_transform(data)
    reduced_data = data_re_transformation(reduced_data)
    kmeans = KMeans(init='k-means++', n_clusters=k, n_init=10)
    kmeans.fit(reduced_data)

    kmeans_real_clusters = KMeans(init='k-means++', n_clusters=k, n_init=10)
    kmeans_real_clusters.fit(reduced_data_high_dim)
    clusters = kmeans_real_clusters.predict(reduced_data_high_dim)

    # Step size of the mesh. Decrease to increase the quality of the VQ.
    h = .02  # point in the mesh [x_min, x_max]x[y_min, y_max].

    # Plot the decision boundary. For that, we will assign a color to each
    x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
    y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Obtain labels for each point in mesh. Use last trained model.
    Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure(1)
    plt.clf()
    plt.imshow(Z, interpolation='nearest',
               extent=(xx.min(), xx.max(), yy.min(), yy.max()),
               cmap=plt.cm.Paired,
               aspect='auto', origin='lower')

    plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=clusters)
    plt.legend(loc='upper right')
    # Plot the centroids as a white X
    centroids = kmeans.cluster_centers_
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='x', s=169, linewidths=3,
                color='w', zorder=10)
    plt.title('K-means clustering on Propser Applicants\n'
              'Centroids are marked with white cross')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())
    plt.show()


def visualize_clusters_in_2D_1(data, num_components=2, k=3):
    # Visualize the results on PCA-reduced data

    reduced_data = PCA(n_components=num_components).fit_transform(data)
    reduced_data = data_re_transformation(reduced_data)
    kmeans = KMeans(init='k-means++', n_clusters=k, n_init=10)
    kmeans.fit(reduced_data)

    clusters = kmeans.predict(reduced_data)

    # Step size of the mesh. Decrease to increase the quality of the VQ.
    h = .02  # point in the mesh [x_min, x_max]x[y_min, y_max].

    # Plot the decision boundary. For that, we will assign a color to each
    x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
    y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Obtain labels for each point in mesh. Use last trained model.
    Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure(1)
    plt.clf()
    plt.imshow(Z, interpolation='nearest',
               extent=(xx.min(), xx.max(), yy.min(), yy.max()),
               cmap=plt.cm.Paired,
               aspect='auto', origin='lower')

    plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=clusters)
    plt.legend(loc='upper right')
    # Plot the centroids as a white X
    centroids = kmeans.cluster_centers_
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='x', s=169, linewidths=3,
                color='w', zorder=10)
    plt.title('K-means clustering on Propser Applicants\n'
              'Centroids are marked with white cross')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())
    plt.show()


def visualize_elbow(data):
    # Fit KMeans and calculate SSE for each *k*
    sse = {}
    for k in range(1, 11):
        print('Running kmeans with k %s' % k)
        kmeans = KMeans(init='k-means++', n_clusters=k, random_state=10)
        kmeans.fit(data)
        sse[k] = kmeans.inertia_  # sum of squared distances to closest cluster center

    # Plot SSE for each *k*
    plt.title('The Elbow Method')
    plt.xlabel('k')
    plt.ylabel('SSE')
    plt.tight_layout()
    sns.pointplot(x=list(sse.keys()), y=list(sse.values()))
    plt.show()
    pickle.dump(sse, 'processed_data/sse.csv')


def compare_clusters(data):
   return 0


def cluster(data, original_data, k, r):
    # Run clustering 10 times, pick the one with min sse

    if r != -1:
        selected_kmeans = KMeans(n_clusters=k, random_state=r)
        selected_kmeans.fit(data)
    else:
        min_sse = 1000000000000
        for r in range(1, 20):
            print(r)
            kmeans = KMeans(n_clusters=k, random_state=r)
            kmeans.fit(data)
            sse = kmeans.inertia_
            if sse < min_sse:
                min_sse = sse
                selected_kmeans = kmeans
                selected_r = r
        print('selected r is %s' % selected_r)

    clusters = selected_kmeans.predict(data)
    original_data['clusters'] = clusters

    grouped_apps = original_data.groupby('clusters')
    centers = grouped_apps.mean()
    centers_median = grouped_apps.median()
    stds = grouped_apps.std()

    centers.to_csv('results/cluster_centers' + str(k) + str(r) +  '.csv')
    stds.to_csv('results/cluster_stds' + str(k) + str(r) + '.csv')
    centers_median.to_csv('results/cluster_medians' + str(k) + str(r) + '.csv')

    pos_centers = grouped_apps.agg(lambda x: x[x > 0].mean())
    pos_centers_median = grouped_apps.agg(lambda x: x[x > 0].median())

    most_frequent_values = grouped_apps.agg(lambda x: x.value_counts().index[0])
    most_frequent_values.to_csv('results/frequent_values' + str(k) + '.csv')

    n_members_per_cluster = grouped_apps.count()
    print(n_members_per_cluster)

    print(grouped_apps['clusters'].count())

    return centers, clusters


def main():
    # Main code
    filename = 'data/your_file.csv'
    upload = False
    k = 3

    # Clean data
    (clean_data, data_post_clustering) = data_cleaning(filename)

    # Visualize
    visualize_distributions_for_skewness

    # Transform data to prepare for PCA/clustering
    transformed_data = \
        data_transformation(clean_data[clean_data.columns.difference(categorical_cols)])

    # Decorrelate and reduce dimension with PCA
    decorrelated_data = decorrelate_reduce(transformed_data, clean_data.columns)

    re_standardized_pca_data = data_re_transformation(decorrelated_data)

    # Cluster data with k-means
    (cluster_centers, clusters) = cluster(re_standardized_pca_data, clean_data, k, 1)
    print(cluster_centers)
