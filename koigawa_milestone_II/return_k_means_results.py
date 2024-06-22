import pandas as pd
from sklearn.cluster import KMeans

def return_k_means_results(df_train,df_test,n_clusters=6,n_init=2,d=4,random_state=0,tol=0.000003):
    """
        This function takes in processed data from function called return_processed_data_clustering and
        returns cluster labels
    """

    print('calculating k-means result to return cluster labels')
    kmeans_orig = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=n_init,tol=tol)


    kmeans_orig.fit(df_train.values[:, :d])
    cluster_labels_orig_train = kmeans_orig.predict(df_train.values[:, :d])
    cluster_labels_orig_test = kmeans_orig.predict(df_test.values[:, :d])

    print('finished calculating k-means')

    return (cluster_labels_orig_train,cluster_labels_orig_test)