import pandas as pd
import yfinance as yf


def get_stock_prices(ticker, start, end):
    """
    Retrieve daily stock price

    Parameters
        > ticker (str): name of the target variable and its respective column in the df input
        > start (str): 'YYYY-MM-DD' for the start of the selection period
        > end (str): 'YYYY-MM-DD' for the end of the selection period
    
    Returns
        > history (pd.DataFrame): table of daily stock prices
            columns
            - date
            - GME: close of day price 
    """


    # get historical daily market price on close
    stock = yf.Ticker(ticker)
    history = stock.history(start=start, end=end, interval="1d")['Close'].reset_index()
    
    # clean up columns 
    history['Date'] = history['Date'].dt.date
    history.columns = ['date', ticker]

    history.to_csv('stock_prices.csv', index=False)
    return history

get_stock_prices('GME', '2021-01-01', '2021-09-30')