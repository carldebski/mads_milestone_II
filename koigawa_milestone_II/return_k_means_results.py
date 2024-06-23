"""
This module gets imported and called by return_clustering_results.py
"""


import pandas as pd
from sklearn.cluster import KMeans

def return_k_means_results(df_train,df_test,n_clusters=6,n_init=2,d=4,random_state=0,tol=0.000003):
    """
        This function takes in processed data from function called return_processed_data_clustering and
        returns cluster labels
        
        Parameters
        > df_train (pd.DataFrame): dataframe that contains train data
        > df_train (pd.DataFrame): dataframe that contains test data
        > n_clusters (Integer): Number of clusters to be found by K-Means
        > n_init (Integer): Hyperparameter for K-Means
        > d (Integer): Number of dimensions for PCA
        > random_state (Integer): Random State to get the same output
        > tol (Float): Hyperparameter for K-Means

        Returns
        > Labels for clustering outcome (Tuple) for train and test
    """

    print('calculating k-means result to return cluster labels')
    kmeans_orig = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=n_init,tol=tol)


    kmeans_orig.fit(df_train.values[:, :d])
    cluster_labels_orig_train = kmeans_orig.predict(df_train.values[:, :d])
    cluster_labels_orig_test = kmeans_orig.predict(df_test.values[:, :d])

    print('finished calculating k-means')

    return (cluster_labels_orig_train,cluster_labels_orig_test)