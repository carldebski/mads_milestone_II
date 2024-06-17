import pandas as pd
import numpy as np


def process_reddit_data(df_cluster, df_topic):
    """
    Prepare reddit data by converting labeled data into counts of activity for those labels 

    Parameters
        > df_clusters (pd.DataFrame): reddit community labeled data
        > df_topic (pd.DataFrame): reddit topic labeled data

    Returns
        > df_activity (pd.DataFrame): counts of community and topic activity by date
    """
    
    # merge community and topic dataframes
    df = df_cluster.merge(df_topic[['id', 'topic']], on='id')

    # convert column formats
    df['date'] = pd.to_datetime(df['date'])
    df['date'] = df['date'].dt.date

    # calculate topic and cluster activity by date
    df['community_discussion'] = df['community_label_str'] + ' community talking about ' + df['topic']
    df_activity = df.groupby(by=['date', 'community_discussion'], as_index=False).count()
    df_activity = df_activity[['date', 'community_discussion', 'id']]
    
    # clean up dataframe
    df_activity.columns = ['date', 'community_discussion', 'count']
    df_activity = df_activity.pivot(index='date', columns='community_discussion', values='count')
    df_activity.reset_index(inplace=True)
    df_activity.fillna(0, inplace=True)
    
    # add column that includes counts of all activity
    df_activity['all'] = df_activity[[c for c in df_activity.columns if c!='date']].sum(axis=1)
    
    #df_activity.to_csv('community_discussion_counts_clean.csv', index=False) 
    return df_activity


def process_financial_data(df, min_date, max_date):
    """
    Prepare financial data by imputing in all missing dates (e.g. weekends and holidays) 
    using the last close price. 

    Parameters
        > df (pd.DataFrame): daily stock prices
        > min_date (str): start of period
        > max_date (str): end of period

    Returns
        > df (pd.DataFrame): stock prices for all days in range
    """

    # fill in missing dates (weekends and holidays)
    df = df.set_index('date')
    idx = pd.date_range(min_date, max_date)
    df.index = pd.DatetimeIndex(df.index)

    # fill in missing values (forward fill)
    df = df.reindex(idx, method='ffill')
    
    # clean up df
    df.reset_index(inplace=True, names='date')
    df['date'] = df['date'].dt.date

    #df.to_csv('stock_prices_clean.csv', index=False)
    return df


def combine_data(df_market, df_reddit):
    """
    Combine market and reddit data by date 

    Parameters
        > df_market (pd.DataFrame): daily market prices
        > df_reddit (pd.DataFrame): daily reddit community / topic counts

    Returns
        > df (pd.DataFrame): combined dataframe
    """


    # update column formats for blending
    df_market['date'] = pd.to_datetime(df_market['date'])
    df_reddit['date'] = pd.to_datetime(df_reddit['date'])

    # combine dataframes
    df = df_market.merge(df_reddit, on='date')
    df = df.dropna(how='any')

    #df.to_csv('combined_data.csv', index=False)
    return df


def transform_data(df_combined_clean, ticker, shift=0, rolling_avg=0, stock_price='diff'):
    """
    Combine market and reddit data by date 

    Parameters
        > df_combined_clean (pd.DataFrame): pre-processed data of target (stock price) and features (counts of communities / topics)
        > ticker (str): stock ticker used as target variable
        > shift (int): number of rows (days) to shift the features prior to fitting the model. 0 does not shift. 
        > rolling_avg (int): number of rows (days) to average the feature variables. 0 does not apply rolling average.
        > stock_price (str): treatment of stock price
            - 'diff': calculates the daily difference of price
            - 'log': calculates the log of the price 

    Returns
        > df (pd.DataFrame): transformed dataframe 
    """

    # identify columns with features
    feature_cols = [c for c in df_combined_clean.columns if (c!='date') & (c!=ticker)]

    # apply transformations
    # transform stock price
    if stock_price=='diff':
        df_combined_clean[ticker] = df_combined_clean[ticker].diff()
    elif stock_price=='log':
        df_combined_clean[ticker] = np.log(df_combined_clean[ticker])
    elif stock_price=='orig':
        pass

    # apply feature shift
    if shift!=0:
        df_combined_clean[feature_cols] = df_combined_clean[feature_cols].shift(shift)
    
    # apply feature rolling average
    if rolling_avg!=0:
        df_combined_clean[feature_cols] = df_combined_clean[feature_cols].rolling(window=rolling_avg).mean()

    # drop rows with missing values
    df_combined_clean = df_combined_clean.dropna(how='any')

    #df_combined_clean.to_csv('transformed_clean.csv', index=False)
    return df_combined_clean


def pre_process_data(df_prices, df_communites, df_topics, min_date, max_date, ticker, **kwargs):
    """
    Connect pre-processing functions  

    Parameters
        > df_prices (pd.DataFrame): daily market prices from yfinance
        > df_communites (pd.DataFrame): reddit community labeled data from PCA model
        > df_topics (pd.DataFrame): reddit topic labeled data from LDA model
        > min_date (str): start of period 
        > max_date (str): end of period
        > ticker (str): stock / index ticker used
        > **kwargs : additional arguments passed to override defaults of called functions if desired

    Returns
        > df (pd.DataFrame): pre-processed dataframe
    """

    # pre-process stock data
    df_prices_clean = process_financial_data(df_prices, min_date, max_date)

    # pre-process reddit data 
    df_reddit_clean = process_reddit_data(df_communites, df_topics)

    # combine market and reddit data
    df_combined_clean = combine_data(df_prices_clean, df_reddit_clean)

    # apply additional data transformations
    df_processed = transform_data(df_combined_clean, ticker, **kwargs)

    return df_processed
