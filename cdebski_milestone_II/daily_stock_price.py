
def get_stock_prices(ticker, start, end):
    """
    """
    import pandas as pd
    import yfinance as yf

    # get historical market price on close
    stock = yf.Ticker(ticker)
    history = stock.history(start=start, end=end, interval="1d")['Close'].reset_index()
    
    # clean up columns 
    history['Date'] = history['Date'].dt.date
    history.columns = ['date', ticker]

    # visualize stock price change over time
    #plt.plot(hist.iloc[:,0], hist.iloc[:,1])
    #plt.savefig('stock_price_history.png')

    # output to csv
    #history.to_csv('stock_prices.csv', index=False)
    return history
