from return_k_means_results import return_k_means_results
from return_processed_data_clustering import return_processed_data_clustering
import pandas as pd

min_posts=3
start_date='2021-01-01'
end_date='2021-09-30'
is_test='N'


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--min_posts",
        default=min_posts,
        help="Minimum number of posts")
    parser.add_argument(
        "--start_date",
        default=start_date,
        help="Start Date of the Clustering Results")
    parser.add_argument(
        "--end_date",
        default=end_date,
        help="End Date of the Clustering Results")
    parser.add_argument(
        "--is_test",
        default=is_test,
        help="Whether the result is for test or train")
    args = parser.parse_args()

pca_df_train,pca_df_test,df_train,df_test,df_orig_train,df_orig_test = return_processed_data_clustering(min_posts=args.min_posts,start_date=args.start_date,end_date=args.end_date)

train_label,test_label = return_k_means_results(df_train=pca_df_train,df_test=pca_df_test)

if args.is_test == 'Y': #if test
    labeled_df = df_test.copy()
    labeled_df['community_label'] = test_label
    df_orig = df_orig_test
else:
    labeled_df = df_train.copy()
    labeled_df['community_label'] = train_label
    df_orig = df_orig_train


# Getting rid of clusters which had very few users
labeled_df['community_label'] = labeled_df.apply(lambda x: 3 if x['community_label']==2 else x['community_label'], axis=1)
labeled_df['community_label'] = labeled_df.apply(lambda x: 3 if x['community_label']==4 else x['community_label'], axis=1)


labeled_df.reset_index(inplace=True)

final_output = df_orig.merge(labeled_df[['author','community_label']],on='author',how='inner')

final_output = final_output[['id','author','date','selftext','community_label']]

cluster_def_dict = {
    0: 'Highly controversial',
    1: 'Unpopular',
    3: 'Core, influential Redditors',
    5: 'Scriptophobic'
}



final_output['community_label_str'] = final_output.apply(lambda x: cluster_def_dict[x['community_label']],axis=1)

file_name = 'community_output_gme_train.csv'
if args.is_test == 'Y':
    file_name = 'community_output_gme_test.csv'

final_output.to_csv(file_name)

