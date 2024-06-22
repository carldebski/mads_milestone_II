import datetime
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import os

def return_processed_data_clustering(forums=['gme'],min_posts=3,start_date='2021-01-01',end_date='2021-09-30'):
    """
        This function takes in list of forums, minimum number of posts (to be classified into a community), and
        start date and end date that you can use for test/train split or cross validation (timesplit).
        This function outputs a dataframe with two columns, both reduced by T-SNE. It will be used to 
        produce cluster labels using K-means
    """

    print('start preparing data')

    if len(start_date) - len(end_date) != 0:
        return 'Please make sure that start date or end date are in the format YYYY-MM-DD'

    start_date = datetime.datetime(int(start_date[0:4]), int(start_date[5:7]), int(start_date[8:10]))
    end_date = datetime.datetime(int(end_date[0:4]), int(end_date[5:7]), int(end_date[8:10]))

    drop_columns=['selftext', 'thumbnail','shortlink','link_flair_text','title','created','retrieved','edited','date','num_post','id']

    #min_posts = min_posts

    df = pd.DataFrame()

    # This is assuming you saved all the reddit forums in a folder called reddit
    # For example, you would have ./reddit/gme/submissions_reddit.csv
    for main_d in os.listdir('../reddit'):
        if main_d in forums and '.' not in main_d:
            df_new = pd.read_csv('../reddit/'+ main_d + '/submissions_reddit.csv')
            df_new = df_new[df_new.author != '[deleted]']
            df = pd.concat([df, df_new])


    # Below preserves the original text-based data and produces a date field that is latest of created and edited
    df_orig = df.copy()
    df_orig['created'] = pd.to_datetime(df_orig['created'])
    df_orig['edited'] = pd.to_datetime(df_orig['edited'])
    df_orig['date'] = df_orig[['created', 'edited']].max(axis=1)

    df_test = df_orig[df_orig.date > end_date].copy()
    df_train = df_orig[(df_orig.date <= end_date) & (df_orig.date >= start_date)].copy()

    df_orig_test = df_test.copy()
    df_orig_train = df_train.copy()

    df_test = df_test.copy().reset_index(drop=True)
    df_train = df_train.copy().reset_index(drop=True)

    df_count_test = df_test.groupby('author',as_index=False)['id'].nunique().rename(columns={"id": "num_post"})
    df_count_train = df_train.groupby('author',as_index=False)['id'].nunique().rename(columns={"id": "num_post"})

    df_test = pd.merge(df_test,df_count_test,on='author',how='left')
    df_train = pd.merge(df_train,df_count_train,on='author',how='left')


    df_test = df_test[df_test.num_post > min_posts]
    df_train = df_train[df_train.num_post > min_posts]


    df_test.drop(columns=drop_columns,inplace=True)
    df_train.drop(columns=drop_columns,inplace=True)

    df_test = df_test.groupby('author',as_index=False).mean()
    df_train = df_train.groupby('author',as_index=False).mean()

    # We want to move author to index since you can't transform text
    df_test.set_index('author',inplace=True)
    df_train.set_index('author',inplace=True)

    df_test.drop(columns=['pinned','archived','deleted'],inplace=True)
    df_train.drop(columns=['pinned','archived','deleted'],inplace=True)

    scaler = StandardScaler()

    scaler.fit(df_train)
    transformed_df_train = scaler.transform(df_train)
    transformed_df_test = scaler.transform(df_test)

    pca = PCA(n_components=7)

    pca_embedding_train = pca.fit_transform(transformed_df_train)

    pca_df_train = pd.DataFrame(
                          {
                           'pca1':pca_embedding_train[:, 0],
                           'pca2':pca_embedding_train[:, 1],
                           'pca3':pca_embedding_train[:, 2],
                           'pca4':pca_embedding_train[:, 3],
                           'pca5':pca_embedding_train[:, 4],
                           'pca6':pca_embedding_train[:, 5],
                           'pca7':pca_embedding_train[:, 6]

                          }
                         )

    pca_embedding_test = pca.transform(transformed_df_test)

    pca_df_test = pd.DataFrame(
                      {
                       'pca1':pca_embedding_test[:, 0],
                       'pca2':pca_embedding_test[:, 1],
                       'pca3':pca_embedding_test[:, 2],
                       'pca4':pca_embedding_test[:, 3],
                       'pca5':pca_embedding_test[:, 4],
                       'pca6':pca_embedding_test[:, 5],
                       'pca7':pca_embedding_test[:, 6]

                      }
                     )

    return (pca_df_train,pca_df_test,df_train,df_test,df_orig_train,df_orig_test)