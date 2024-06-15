import pandas as pd
import numpy as np
import yfinance as yf


def process_reddit_data(df_cluster, df_topic):
    
    df = df_cluster.merge(df_topic[['id', 'topic']], on='id')

    # convert column formats
    df['date'] = pd.to_datetime(df['date'])
    df['date'] = df['date'].dt.date

    # topic and cluster activity by date
    df['community_discussion'] = df['community_label_str'] + ':' + df['topic']
    df_activity = df.groupby(by=['date', 'community_discussion'], as_index=False).count()
    df_activity = df_activity[['date', 'community_discussion', 'id']]
    
    df_activity.columns = ['date', 'community_discussion', 'count']
    
    df_activity = df_activity.pivot(index='date', columns='community_discussion', values='count')
    

    df_activity.reset_index(inplace=True)
    df_activity.fillna(0, inplace=True)
    
    df_activity['all'] = df_activity[[c for c in df_activity.columns if c!='date']].sum(axis=1)
    #df_activity.to_csv('community_discussion_counts_clean.csv', index=False) 
    return df_activity


def process_financial_data(df, min_date, max_date):

    df = df.set_index('date')
    idx = pd.date_range(min_date, max_date)
    df.index = pd.DatetimeIndex(df.index)
    df = df.reindex(idx, method='ffill')
    df.reset_index(inplace=True, names='date')
    df['date'] = df['date'].dt.date

    #df.to_csv('stock_prices_clean.csv', index=False)
    return df


def combine_data(df_market, df_reddit):

    df_market['date'] = pd.to_datetime(df_market['date'])
    df_reddit['date'] = pd.to_datetime(df_reddit['date'])

    df = df_market.merge(df_reddit, on='date')
    df = df.dropna(how='any')

    #df.to_csv('combined_data.csv', index=False)
    return df

def transform_data(df_combined_clean, ticker, shift=0, rolling_avg=0, stock_price='diff'):
    feature_cols = [c for c in df_combined_clean.columns if (c!='date') & (c!=ticker)]

    if stock_price=='diff':
        df_combined_clean[ticker] = df_combined_clean[ticker].diff()
    
    if shift!=0:
        df_combined_clean[feature_cols] = df_combined_clean[feature_cols].shift(shift)
        
    if rolling_avg!=0:
        df_combined_clean[feature_cols] = df_combined_clean[feature_cols].rolling(window=rolling_avg).mean()

    df_combined_clean = df_combined_clean.dropna(how='any')

    #df_combined_clean.to_csv('transformed_clean.csv', index=False)
    return df_combined_clean


def pre_process_data(df_prices, df_communites, df_topics, min_date, max_date, ticker, **kwargs):
    
    # will process stock data and save a csv 'stock_prices_clean.csv
    df_prices_clean = process_financial_data(df_prices, min_date, max_date)

    # pre-process reddit data (including dummy topics / clusters) and save as csv 'community_discussion_counts_clean.csv'
    df_reddit_clean = process_reddit_data(df_communites, df_topics)

    # combine market and reddit data
    df_combined_clean = combine_data(df_prices_clean, df_reddit_clean)

    #
    df_processed = transform_data(df_combined_clean, ticker, **kwargs)

    return df_processed

#combine_data('community_discussion_counts_clean.csv', 'stock_prices_clean.csv')
#process_financial_data('stock_prices.csv', '2021-01-05', '2021-12-10')
#process_reddit_data('koigawa_milestone_II/community_output_gme.csv', 'smoilanen_milestone_II/df_test_train_split.csv')