"""MC2-P1: Market simulator.  		   	  			  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
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
  		   	  			  	 		  		  		    	 		 		   		 		  
Student Name: Tucker Balch (replace with your name)  		   	  			  	 		  		  		    	 		 		   		 		  
GT User ID: shollister7 (replace with your User ID)  		   	  			  	 		  		  		    	 		 		   		 		  
GT ID: 903304661 (replace with your GT ID)  		   	  			  	 		  		  		    	 		 		   		 		  
"""  		   	  			  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
import pandas as pd  		   	  			  	 		  		  		    	 		 		   		 		  
import numpy as np  		   	  			  	 		  		  		    	 		 		   		 		  
import datetime as dt  		   	  			  	 		  		  		    	 		 		   		 		  
import os  		   	  			  	 		  		  		    	 		 		   		 		  
from util import get_data, plot_data  		   	  			  	 		  		  		    	 		 		   		 		  

def author():
    return 'shollister7'

def compute_portvals(orders_df, start_val = 1000000, commission=9.95, impact=0.005):  		   	  			  	 		  		  		    	 		 		   		 		  
    # this is the function the autograder will call to test your code  		   	  			  	 		  		  		    	 		 		   		 		  
    # NOTE: orders_file may be a string, or it may be a file object. Your  		   	  			  	 		  		  		    	 		 		   		 		  
    # code should work correctly with either input  		   	  			  	 		  		  		    	 		 		   		 		  
    # TODO: Your code here  		   	  			  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
    #Load in:
    # - orders
    # - dates
    # - list of stocks to call get_data on
    #orders = pd.read_csv(orders_file, index_col='Date', parse_dates=True, na_values=['nan'] ).sort_index()
    orders = orders_df.sort_index()
    stocks = orders['Symbol'].unique().tolist()
    start_date = orders.index[0]
    end_date = orders.index[-1]
    dates=pd.date_range(start_date, end_date)
    orders.fillna(method='ffill', inplace=True)
    orders.fillna(method='backfill', inplace=True)

    #get data and fill na
    data = get_data(stocks, dates)
    data.fillna(method='ffill', inplace=True)
    data.fillna(method='backfill', inplace=True)
    data["cash_change"] = 1.0
    
    #df that has change in number of shares by day for each asset
    # df also includes change in cash
    # make another df called port to aggregate portfolio
    #share_chg = pd.DataFrame(np.zeros((data.shape)), data.index, data.columns)
    share_chg = data.copy()
    port = data.copy()
    for col in share_chg.columns:
        share_chg[col].values[:] = 0
        port[col].values[:] = 0
    rows = orders.iterrows()
    for idx, row in rows:
        ticker = row[0]
        ord_type = row[1]
        shares = row[2]
        value = data.loc[idx, ticker] * shares
        cost = value * impact + commission
        curr_shares = share_chg.loc[idx, ticker]
        curr_cash = share_chg.loc[idx, "cash_change"]

        if ord_type == "SELL":
            share_chg.loc[idx, ticker] = curr_shares - shares
            share_chg.loc[idx, "cash_change"] = curr_cash + (value - cost)
        elif ord_type == "BUY":
            share_chg.loc[idx, ticker] = curr_shares + shares
            share_chg.loc[idx, "cash_change"] = curr_cash - (value + cost)

    port.iloc[0, :-1] = share_chg.iloc[0, :-1]
    port.iloc[0, -1] = share_chg.iloc[0, -1] + start_val

    for count in range(1, len(port.index)): port.iloc[count] = port.iloc[count-1] + share_chg.iloc[count]
    port = (data * port).sum(axis = 1)
    portvals = pd.DataFrame(port, index= port.index, columns = ["portfolio_totals"])
    rv = pd.DataFrame(index=portvals.index, data=portvals.values)

    return portvals 			  	 		  		  		    	 		 		   		 		    		   	  			  	 		  		  		    	 		 		   		 		  
    return rv 		   	  			  	 		  		  		    	 		 		   		 		  
     		   	  			  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
def test_code(orders_df):  		   	  			  	 		  		  		    	 		 		   		 		  
    # this is a helper function you can use to test your code  		   	  			  	 		  		  		    	 		 		   		 		  
    # note that during autograding his function will not be called.  		   	  			  	 		  		  		    	 		 		   		 		  
    # Define input parameters  		   	  			  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
    of = "./orders/orders-02.csv"  		   	  			  	 		  		  		    	 		 		   		 		  
    sv = 1000000  		   	  			  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
    # Process orders  		   	  			  	 		  		  		    	 		 		   		 		  
    portvals = compute_portvals(orders_df, start_val = sv)  		   	  			  	 		  		  		    	 		 		   		 		  
    if isinstance(portvals, pd.DataFrame):  		   	  			  	 		  		  		    	 		 		   		 		  
        portvals = portvals[portvals.columns[0]] # just get the first column  		   	  			  	 		  		  		    	 		 		   		 		  
    else:  		   	  			  	 		  		  		    	 		 		   		 		  
        "warning, code did not return a DataFrame"  		   	  			  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
    # Get portfolio stats  		   	  			  	 		  		  		    	 		 		   		 		  
    # Here we just fake the data. you should use your code from previous assignments.  		   	  			  	 		  		  		    	 		 		   		 		  
    start_date = dt.datetime(2008,1,1)  		   	  			  	 		  		  		    	 		 		   		 		  
    end_date = dt.datetime(2008,6,1)  		   	  			  	 		  		  		    	 		 		   		 		  
    cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = [0.2,0.01,0.02,1.5]  		   	  			  	 		  		  		    	 		 		   		 		  
    cum_ret_SPY, avg_daily_ret_SPY, std_daily_ret_SPY, sharpe_ratio_SPY = [0.2,0.01,0.02,1.5]  		   	  			  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
    # Compare portfolio against $SPX  		   	  			  	 		  		  		    	 		 		   		 		  
    print(f"Date Range: {start_date} to {end_date}")  		   	  			  	 		  		  		    	 		 		   		 		  
    print()  		   	  			  	 		  		  		    	 		 		   		 		  
    print(f"Sharpe Ratio of Fund: {sharpe_ratio}")  		   	  			  	 		  		  		    	 		 		   		 		  
    print(f"Sharpe Ratio of SPY : {sharpe_ratio_SPY}")  		   	  			  	 		  		  		    	 		 		   		 		  
    print()  		   	  			  	 		  		  		    	 		 		   		 		  
    print(f"Cumulative Return of Fund: {cum_ret}")  		   	  			  	 		  		  		    	 		 		   		 		  
    print(f"Cumulative Return of SPY : {cum_ret_SPY}")  		   	  			  	 		  		  		    	 		 		   		 		  
    print()  		   	  			  	 		  		  		    	 		 		   		 		  
    print(f"Standard Deviation of Fund: {std_daily_ret}")  		   	  			  	 		  		  		    	 		 		   		 		  
    print(f"Standard Deviation of SPY : {std_daily_ret_SPY}")  		   	  			  	 		  		  		    	 		 		   		 		  
    print()  		   	  			  	 		  		  		    	 		 		   		 		  
    print(f"Average Daily Return of Fund: {avg_daily_ret}")  		   	  			  	 		  		  		    	 		 		   		 		  
    print(f"Average Daily Return of SPY : {avg_daily_ret_SPY}")  		   	  			  	 		  		  		    	 		 		   		 		  
    print()  		   	  			  	 		  		  		    	 		 		   		 		  
    print(f"Final Portfolio Value: {portvals[-1]}")  		   	  			  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		  		   	  			  	 		  		  		    	 		 		   		 		  
