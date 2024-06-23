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
    
    # count posts by community, topic, and community and topic
    df_comm = df.groupby(by=['date', 'community_label_str'], as_index=False).count()
    df_topic = df.groupby(by=['date', 'topic'], as_index=False).count()
    df_both = df.groupby(by=['date', 'community_discussion'], as_index=False).count()
    
    # convert from tall to wide 
    df_comm = df_comm.pivot(index='date', columns='community_label_str', values='id').reset_index()
    df_topic = df_topic.pivot(index='date', columns='topic', values='id').reset_index()
    df_both = df_both.pivot(index='date', columns='community_discussion', values='id').reset_index()
    
    # combine three feature sources
    df_activity = df_both.merge(df_comm, on='date')
    df_activity = df_activity.merge(df_topic, on='date')
    df_activity = df_activity.fillna(0)
    
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


def transform_data(df_combined_clean, ticker, shift=1, rolling_avg=0, stock_price='diff'):
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
        df_combined_clean[ticker] = np.log(df_combined_clean[ticker] / df_combined_clean[ticker].shift(1))
    elif stock_price=='orig':
        pass

    # create features that are shifts of existing features 
    if shift!=0:
        df_combined_clean[[c+'_shift{}'.format(shift) for c in feature_cols]] = df_combined_clean[feature_cols].shift(shift)
    
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


def feature_selection(df, ticker):
    """
    Rank correlation of features to target variable (stock price). Return r2 values. 

    Parameters
        > df (pd.DataFrame): dataframe of features and target variable
        > ticker (str): column name of target variable 

    Returns
        > df (pd.DataFrame): dataframe of r2 restuls sorted highest to lowest. 
    """
    
    # drop unused columns
    df = df.drop('date', axis=1)
  
    # dataframe to capture r2 results for each feature
    results = pd.DataFrame(columns=['feature', 'r2'])
    results['feature'] = df.columns

    # calucate r2 for each feature with the stock price
    for col in df.columns:
        correlation_matrix = df.corr()
        correlation = correlation_matrix.loc[ticker, col]
        r_squared = correlation**2
        results.loc[results['feature']==col, ['r2']] = r_squared

    # highest correlations listed first
    results.sort_values(by='r2', ascending =False, inplace=True)

    # remove correlation with self
    results = results[results['feature'] != ticker]

    return results['feature'].to_list()
