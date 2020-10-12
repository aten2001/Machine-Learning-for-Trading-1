import pandas as pd
import util
import numpy as np 
import datetime as dt
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

def author():
    return 'shollister7'

def rolling_avg(df_prices, ticker, results_df):
    adj_closes = df_prices[ticker]
    adj_closes = adj_closes/adj_closes[0]
    results_df["sma"] = adj_closes.rolling(20).mean()
    results_df["price/sma"] = adj_closes / adj_closes.rolling(20).mean()

def bollinger_bands(df_prices, ticker, results_df):
    adj_closes = df_prices[ticker]
    adj_closes = adj_closes/adj_closes[0]
    ma = adj_closes.rolling(20).mean()
    sd = adj_closes.rolling(20).std()
    higher_b = ma + (2* sd)
    lower_b = ma - (2 * sd)
    results_df["upper_b"] = higher_b
    results_df["lower_b"] = lower_b
    results_df["bb_num"] = (adj_closes - ma) / (2 * sd)

def momentum(df_prices, ticker, results_df):
    adj_closes = df_prices[ticker]
    adj_closes = adj_closes/adj_closes[0]
    m = adj_closes.div(adj_closes.shift(1)) - 1
    results_df["momentum"] = m

def max_index(x):
    return 100 * (int(np.argmax(x)) + 1) / 25

def min_index(x):
    return 100 * (int(np.argmin(x)) + 1) / 25

def aroon(df_prices, ticker, results_df):
    adj_closes = df_prices[ticker]
    adj_closes = adj_closes/adj_closes[0]
    df_prices["aroon_up"] = adj_closes.rolling(25).apply(max_index, raw=True)
    df_prices["aroon_down"] = adj_closes.rolling(25).apply(min_index, raw=True)


def prepare_pricedf(symbol, startDate, endDate):
    time_period = pd.date_range(startDate, endDate)
    df_prices = util.get_data([symbol], time_period,False)
    df_prices = df_prices.dropna()
    df_prices = df_prices.fillna(method='ffill')
    df_prices = df_prices.fillna(method='bfill')
    df_prices = df_prices / df_prices.iloc[0,]
    return df_prices

def plot_sma(df_p):
    curr_plt = plt.figure(0)
    plt.title("JPM Normalized Price and 20 day SMA")
    plt.plot(df_p['JPM'], label = "JPM Normalized Price")
    plt.plot(df_p['sma'], label = "20 day sma")
    plt.legend(loc="lower left")

    plt.savefig("sma_chart.png")
    
    value_plot = plt.figure(1)
    plt.title("JPM Adj Close / SMA Value Chart")
    plt.plot(df_p['price/sma'], label = "price/sma")
    plt.legend(loc="lower left")

    plt.savefig("sma_value.png")

def plot_bb(df_p):
    curr_plt = plt.figure(2)
    plt.title("JPM Bollinger Bands")
    plt.plot(df_p['JPM'], label = "JPM Normalized Price")
    plt.plot(df_p['sma'], label = "20 day JPM SMA")
    plt.plot(df_p['upper_b'], label = "Upper Band")
    plt.plot(df_p['lower_b'], label = "Lower Band")
    plt.legend(loc="lower left")
    plt.savefig("bb_chart.png")

    value_plot = plt.figure(3)
    plt.title("JPM Bollinger Bands Percent Indicator")
    plt.plot(df_p['bb_num'], label = "BB %")
    plt.legend(loc="lower left")
    plt.savefig("bb_value.png")

def plot_momentum(df_p):
    curr_plt = plt.figure(4)
    plt.title("JPM Momentum")
    plt.plot(df_p['momentum'], label = "JPM Momentum Indicator")
    plt.legend(loc="lower left")
    plt.savefig("momentum.png")

def plot_aroon(df_p):
    curr_plt = plt.figure(5)
    plt.title("JPM Aroon Indicator")
    plt.plot(df_p['aroon_up'], label = "JPM Aroon up Indicator")
    plt.plot(df_p['aroon_down'], label = "JPM Aroon down Indicator")
    plt.legend(loc="lower left")
    plt.savefig("aroon.png")

def main():
    #start = dt.datetime(2010,1,1)
    #end = dt.datetime(2011,12,31)
    start = dt.datetime(2008,1,1)
    end = dt.datetime(2009,12,31)
    df_p = prepare_pricedf("JPM", start, end)
    
    
    rolling_avg(df_p, "JPM", df_p)
    bollinger_bands(df_p, "JPM", df_p)
    momentum(df_p, "JPM", df_p )
    aroon(df_p, "JPM", df_p)

    #Call methods to plot every indicator
    plot_aroon(df_p)
    plot_sma(df_p)
    plot_bb(df_p)
    plot_momentum(df_p)
    #print(df_p)
    

    #print(df_p)



if __name__ == "__main__":
    main()
