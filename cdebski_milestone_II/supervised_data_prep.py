import pandas as pd
import numpy as np
import yfinance as yf


def process_reddit_data(csv):
    
    df = pd.read_csv(csv)

    # convert column formats
    df['date'] = pd.to_datetime(df['date'])
    df['date'] = df['date'].dt.date

    # topic and cluster activity by date
    # add back topics once available
    df['community_discussion'] = df['community_label'] #+ ' ' + df['topics']
    df_activity = df.groupby(by=['date', 'community_discussion'], as_index=False).count()
    df_activity = df_activity[['date', 'community_discussion', 'id']]
    df_activity.columns = ['date', 'community_discussion', 'count']

    df_activity = df_activity.pivot(index='date', columns='community_discussion', values='count')
    df_activity.reset_index(inplace=True)
    df_activity.fillna(0, inplace=True)
    df_activity['all'] = df_activity[[c for c in df_activity.columns if c!='date']].sum(axis=1)

    df_activity.to_csv('community_discussion_counts_clean.csv', index=False) 


def process_financial_data(csv, min_date, max_date):
    df = pd.read_csv(csv)
    df = df.set_index('date')
    idx = pd.date_range(min_date, max_date)
    df.index = pd.DatetimeIndex(df.index)
    df = df.reindex(idx, method='ffill')
    df.reset_index(inplace=True, names='date')
    df['date'] = df['date'].dt.date


    df.to_csv('stock_prices_clean.csv', index=False)


def combine_data(reddit_csv, market_csv):
    df_market = pd.read_csv(market_csv)
    df_reddit = pd.read_csv(reddit_csv)

    df_market['date'] = pd.to_datetime(df_market['date'])
    df_reddit['date'] = pd.to_datetime(df_reddit['date'])

    df = df_market.merge(df_reddit, on='date')
    df = df.dropna(how='any')

    df.to_csv('combined_data.csv', index=False)

#combine_data('community_discussion_counts_clean.csv', 'stock_prices_clean.csv')
#process_financial_data('stock_prices.csv', '2021-01-05', '2021-12-10')
#process_reddit_data('koigawa_milestone_II/milestone_II/community_output_gme.csv')