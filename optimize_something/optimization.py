"""MC1-P2: Optimize a portfolio.  		   	  			  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
Copyright 2018, Georgia Institute of Technology (Georgia Tech)  		   	  			  	 		  		  		    	 		 		   		 		  
Atlanta, Georgia 30332  		   	  			  	 		  		  		    	 		 		   		 		  
All Rights Reserved  		   	  			  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
Template code for CS 4646/7646  		   	  			  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
Georgia Tech asserts copyright ownership of this template and all derivative  		   	  			  	 		  		  		    	 		 		   		 		  
works, including solutions to the projects assigned in this course. Students  		   	  			  	 		  		  		    	 		 		   		 		  
and other users of this template code are advised not to share it with others  		   	  			  	 		  		  		    	 		 		   		 		  
or to make it available on publicly viewable websites including repositories  		   	  			  	 		  		  		    	 		 		   		 		  
such as github and gitlab.  This copyright statement should not be removed  		   	  			  	 		  		  		    	 		 		   		 		  
or edited.  		   	  			  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
We do grant permission to share solutions privately with non-students such  		   	  			  	 		  		  		    	 		 		   		 		  
as potential employers. However, sharing with other current or future  		   	  			  	 		  		  		    	 		 		   		 		  
students of CS 7646 is prohibited and subject to being investigated as a  		   	  			  	 		  		  		    	 		 		   		 		  
GT honor code violation.  		   	  			  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
-----do not edit anything above this line---  		   	  			  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
Student Name: Sebastian Hollister (replace with your name)  		   	  			  	 		  		  		    	 		 		   		 		  
GT User ID: shollister7 (replace with your User ID)  		   	  			  	 		  		  		    	 		 		   		 		  
GT ID: 903304661 (replace with your GT ID)  		   	  			  	 		  		  		    	 		 		   		 		  
"""  		   	  			  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
import pandas as pd  		   	  			  	 		  		  		    	 		 		   		 		  
import matplotlib.pyplot as plt  		   	  			  	 		  		  		    	 		 		   		 		  
import numpy as np  		   	  			  	 		  		  		    	 		 		   		 		  
import datetime as dt  		  
import scipy.optimize as opt 	  			  	 		  		  		    	 		 		   		 		  
from util import get_data, plot_data


def normalize(prices):
    return prices / prices.iloc[0]

def get_returns(prices):
    return prices.pct_change()

def calc_sharpe_ratio(returns_df, rfr=0.0):
    trading_days = 252.0
    excessReturn = (returns_df[1:] - rfr).mean()
    excessReturnStd = (returns_df[1:] - rfr).std()
    return np.sqrt(trading_days) * (excessReturn / excessReturnStd)


def get_portfolio_stats(allocs, prices):
    #return  sharpe ratio (sr), cr (cumulative return), adr (avg daily return), sddr (st. dev daily return)
    normalized_prices = normalize(prices)
    total_value = (allocs * normalized_prices).sum(axis=1)
    returns_df = get_returns(total_value)

    #calculate cr
    start = total_value[0]
    end_gains = total_value[-1]
    cr = (end_gains - start / start)
    
    #calc adr and sddr
    adr = returns_df.mean()
    sddr = returns_df.std()

    #call helper fun to calc sharpe ratio
    sr = calc_sharpe_ratio(returns_df)

    return cr, adr, sddr, sr

def negate_sharpe(allocs, prices):
    cr, adr, sddr, sr = get_portfolio_stats(allocs, prices)
    return -1.0 * sr

def get_allocs(syms, prices):
    allocs = len(syms) * [1.0/len(syms)]
    limits = ((0,1), ) * len(syms)
    optimized = opt.minimize(negate_sharpe, allocs, args=(prices,),method='SLSQP',bounds=limits,
                                constraints=({'type':'eq', 'fun': lambda x:np.sum(x)-1}))
    allocs = optimized["x"]
    return allocs


  			  	 		  		  		    	 		 		   		 		  
# This is the function that will be tested by the autograder  		   	  			  	 		  		  		    	 		 		   		 		  
# The student must update this code to properly implement the functionality  		   	  			  	 		  		  		    	 		 		   		 		  
def optimize_portfolio(sd=dt.datetime(2008,6,1), ed=dt.datetime(2009,6,1), \
    syms= ['IBM', 'X', 'GLD', 'JPM'], gen_plot=True):  		   	  			  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
    # Read in adjusted closing prices for given symbols, date range  		   	  			  	 		  		  		    	 		 		   		 		  
    dates = pd.date_range(sd, ed)  		   	  			  	 		  		  		    	 		 		   		 		  
    prices_all = get_data(syms, dates)  # automatically adds SPY  		   	  			  	 		  		  		    	 		 		   		 		  
    prices = prices_all[syms]  # only portfolio symbols  		   	  			  	 		  		  		    	 		 		   		 		  
    prices_SPY = prices_all['SPY']  # only SPY, for comparison later  		   	  			  	 		  		  		    	 		 		   		 		  
    
    
    # find the allocations for the optimal portfolio  		   	  			  	 		  		  		    	 		 		   		 		  
    # note that the values here ARE NOT meant to be correct for a test case  		   	  			  	 		  		  		    	 		 		   		 		  
    allocs = np.asarray([0.2, 0.1, 0.3, 0.3, 0.1]) # add code here to find the allocations  		   	  			  	 		  		  		    	 		 		   		 		  
    cr, adr, sddr, sr = [0.25, 0.001, 0.0005, 2.1] # add code here to compute stats

    #GET ALLOCS & PROTFOLIO STATS
    allocs = get_allocs(syms, prices)
    cr, adr, sddr, sr = get_portfolio_stats(allocs, prices)  

    # Get daily portfolio value
    normalized_prices = normalize(prices)
    total_value = (allocs * normalized_prices).sum(axis=1)
    port_val = total_value	  			  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		     	  			  	 		  		  		    	 		 		   		 	 	  			  	 		  		  		    	 		 		   		 		  
    # Compare daily portfolio value with SPY using a normalized plot  		   	  			  	 		  		  		    	 		 		   		 		  
    if gen_plot:  		   	  			  	 		  		  		    	 		 		   		 		  
        chart = port_val.plot(title = "Daily Portfolio Value and SPY", label='Portfolio', color='blue')
        benchmark = normalize(get_data(['SPY'], dates=dates))
        benchmark.plot(label="SPY", color='green', ax=chart)
        chart.legend(loc='upper left')
        chart.set_xlabel("Date")
        chart.set_ylabel("Price")
        chart.figure.savefig("results.png")
    
    
    
  		   	  			  	 		  		  		    	 		 		   		 		  
    return allocs, cr, adr, sddr, sr  		   	  			  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
def test_code():  		   	  			  	 		  		  		    	 		 		   		 		  
    # This function WILL NOT be called by the auto grader  		   	  			  	 		  		  		    	 		 		   		 		  
    # Do not assume that any variables defined here are available to your function/code  		   	  			  	 		  		  		    	 		 		   		 		  
    # It is only here to help you set up and test your code  		   	  			  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
    # Define input parameters  		   	  			  	 		  		  		    	 		 		   		 		  
    # Note that ALL of these values will be set to different values by  		   	  			  	 		  		  		    	 		 		   		 		  
    # the autograder!  		   	  			  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
    start_date = dt.datetime(2008,6,1)  		   	  			  	 		  		  		    	 		 		   		 		  
    end_date = dt.datetime(2009,6,1)  		   	  			  	 		  		  		    	 		 		   		 		  
    symbols = ['IBM', 'X', 'GLD', 'JPM'] 		   	  			  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
    # Assess the portfolio  		   	  			  	 		  		  		    	 		 		   		 		  
    allocations, cr, adr, sddr, sr = optimize_portfolio(sd = start_date, ed = end_date,\
        syms = symbols, \
        gen_plot = True)  		   	  			  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
    # Print statistics  		   	  			  	 		  		  		    	 		 		   		 		  
    print(f"Start Date: {start_date}")  		   	  			  	 		  		  		    	 		 		   		 		  
    print(f"End Date: {end_date}")  		   	  			  	 		  		  		    	 		 		   		 		  
    print(f"Symbols: {symbols}")  		   	  			  	 		  		  		    	 		 		   		 		  
    print(f"Allocations:{allocations}")  		   	  			  	 		  		  		    	 		 		   		 		  
    print(f"Sharpe Ratio: {sr}")  		   	  			  	 		  		  		    	 		 		   		 		  
    print(f"Volatility (stdev of daily returns): {sddr}")  		   	  			  	 		  		  		    	 		 		   		 		  
    print(f"Average Daily Return: {adr}")  		   	  			  	 		  		  		    	 		 		   		 		  
    print(f"Cumulative Return: {cr}")

   	  			  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
if __name__ == "__main__":  		   	  			  	 		  		  		    	 		 		   		 		  
    # This code WILL NOT be called by the auto grader  		   	  			  	 		  		  		    	 		 		   		 		  
    # Do not assume that it will be called  		   	  			  	 		  		  		    	 		 		   		 		  
    test_code()  		   	  			  	 		  		  		    	 		 		   		 		  
