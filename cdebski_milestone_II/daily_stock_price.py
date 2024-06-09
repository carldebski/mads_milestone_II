import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

def get_stock_prices(ticker, start='2020-01-01', end='2022-01-13'):

    # get historical market price on close
    stock = yf.Ticker(ticker)
    hist = stock.history(start=start, end=end, interval="1d")['Close'].reset_index()
    
    # clean up columns 
    hist['Date'] = hist['Date'].dt.date
    hist.columns = ['date', ticker]

    # visualize stock price change over time
    plt.plot(hist.iloc[:,0], hist.iloc[:,1])
    plt.savefig('stock_price_history.png')

    # output to csv
    hist.to_csv('stock_prices.csv', index=False)
