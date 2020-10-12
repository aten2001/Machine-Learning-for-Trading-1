import pandas as pd
import util
import numpy as np 
import datetime as dt
import matplotlib.pyplot as plt
from marketsimcode import *
from indicators import *
import matplotlib.pyplot as plt

class TheoreticallyOptimalStrategy(object):
    def look_ahead(self, df_prices, symbol="JPM"):
        share_orders = []
        dates = []
        count = []
        adj_closes = df_prices[symbol]
        curr_shares = 0
        #Max shares you can have is +- 1000
        #So, look ahead and if the next day closes higher, buy either 1000 or 2000 to get to + 1000
        # If next day closes lower, sell either 1000 or 2000
        for date in range(len(df_prices) - 1):
            today = adj_closes[date]
            tomorrow = adj_closes[date+1]
            if (tomorrow > today):
                if curr_shares == 0:
                    dates.append(df_prices.index[date])
                    share_orders.append(1000)
                    curr_shares = curr_shares + 1000
                    count.append(curr_shares)
                elif curr_shares == -1000:
                    dates.append(df_prices.index[date])
                    share_orders.append(2000)
                    curr_shares = curr_shares + 2000
                    count.append(curr_shares)
            elif (tomorrow < today):
                if curr_shares == 0:
                    dates.append(df_prices.index[date])
                    share_orders.append(-1000)
                    curr_shares = curr_shares - 1000
                    count.append(curr_shares)
                if curr_shares == 1000:
                    dates.append(df_prices.index[date])
                    share_orders.append(-2000)
                    curr_shares = curr_shares - 2000
                    count.append(curr_shares)
        
        if curr_shares != 0:
            share_orders.append(-curr_shares)
            dates.append(df_prices.index[len(df_prices.index)-2])
        
        buy_sell = []
        for order in share_orders:
            if order < 0:
                buy_sell.append("SELL")
            elif order > 0:
                buy_sell.append("BUY")
        abs_orders = [abs(x) for x in share_orders]
        symbols=[]
        for i in range(len(abs_orders)):
            symbols.append(symbol)
        
        df_trades = pd.DataFrame(data = symbols, index = dates, columns = ['Symbol'])
        df_trades["Order"] = buy_sell
        df_trades["Shares"] = abs_orders
        df_trades.index.name = "Date"
        return df_trades
        
    def plot_ideal_trades(self, portvals, bench_portvals):
        portvals["Theoretical Returns"] = portvals["portfolio_totals"]
        portvals = portvals.drop("portfolio_totals", axis =1)

        bench_portvals["Benchmark Returns"] = bench_portvals["portfolio_totals"]
  

        total_df = pd.DataFrame(index=portvals.index)
        total_df["Theoretical Returns"] = portvals["Theoretical Returns"]
        total_df["Benchmark Returns"] = bench_portvals["Benchmark Returns"]
        total_df["Theoretical Returns"] = total_df["Theoretical Returns"] / total_df["Theoretical Returns"][0]
        total_df["Benchmark Returns"] = total_df["Benchmark Returns"] / total_df["Benchmark Returns"][2]
   

        plt.title("Returns of Theoretically Optimal Strategy vs Benchmark")
        plt.plot(total_df["Theoretical Returns"], color="red", label = "Theoretical Returns", )
        plt.plot(total_df["Benchmark Returns"], color="green", label = "Benchmark Returns")
        plt.legend(loc="upper left")

        plt.savefig("theoreticalStratReturns2.png")

    def create_benchmark_tradesDF(self, df_prices, symbol="JPM"):
        dates = []
        dates.append(df_prices.index[3])
        dates.append(df_prices.index[len(df_prices.index)-2])
        symbols = [symbol,symbol]
        df_trades = pd.DataFrame(data = symbols, index = dates, columns = ['Symbol'])
        df_trades["Order"] = ["BUY", "SELL"]
        df_trades["Shares"] = [1000, 1000]
        df_trades.index.name = "Date"
        return df_trades
    
    def compute_port_stats(self, portvals):
        rfr = 0.0
        sr = 252.0
        portvals = portvals["portfolio_totals"]
        cumulative_return = (portvals[-1]/portvals[0]) - 1
        daily_ret = (portvals/portvals.shift(1)) - 1   
        
        avg_daily = daily_ret.mean()
        std_daily = daily_ret.std()
        diff = (daily_ret - rfr).mean()
        sharpe = np.sqrt(sr) * (diff / std_daily)
        return cumulative_return, avg_daily, std_daily, sharpe

    def author(self):
        return "shollister7"
        
    def testPolicy(self, symbol = "JPM", sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,12,31), sv=10000):
        dates = pd.date_range(sd,ed)
        df_prices = util.get_data([symbol], dates, False)
        trades_df = self.look_ahead(df_prices)
        return trades_df

if __name__ == "__main__":
    ts = TheoreticallyOptimalStrategy()
    sd=dt.datetime(2008,1,1)
    ed=dt.datetime(2009,12,31)
    df_prices = prepare_pricedf(sd, ed)
    
    #Create Benchmark trades_df
    bench_df = ts.create_benchmark_tradesDF(df_prices)
    bench_portvals = compute_portvals(bench_df, start_val = 100000, commission=0.00, impact=0.00)

    #Create Manual Strat Trades DF
    trades_df = ts.testPolicy()

    portvals = compute_portvals(trades_df, start_val = 100000, commission=0.00, impact=0.00)

    #Plot
    
    ts.plot_ideal_trades(portvals, bench_portvals)

    #Print out Metrics
    start_date = sd 		   	  			  	 		  		  		    	 		 		   		 		  
    end_date = ed

    #Port Metrics   	  			  	 		  		  		    	 		 		   		 		  
    cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = ts.compute_port_stats(portvals)			  	 		  		  		    	 		 		   		 		  
    cum_ret_bench, avg_daily_ret_bench, std_daily_ret_bench, sharpe_ratio_bench = ts.compute_port_stats(bench_portvals) 		   	  			  	 		  		  		    	 		 		   		 		  

    bench_returns = bench_portvals["portfolio_totals"]
    man_returns = portvals["portfolio_totals"]                                                                                          
    # Compare portfolio against $SPX  		   	  			  	 		  		  		    	 		 		   		 		  
    print(f"Date Range: {start_date} to {end_date}")  		   	  			  	 		  		  		    	 		 		   		 		  
    print()  		   	  			  	 		  		  		    	 		 		   		 		  
    print(f"Sharpe Ratio of Theoretical Strat: {sharpe_ratio}")  		   	  			  	 		  		  		    	 		 		   		 		  
    print(f"Sharpe Ratio of Benchmark : {sharpe_ratio_bench}")  		   	  			  	 		  		  		    	 		 		   		 		  
    print()  		   	  			  	 		  		  		    	 		 		   		 		  
    print(f"Cumulative Return of Theoretical Strat: {cum_ret}")  		   	  			  	 		  		  		    	 		 		   		 		  
    print(f"Cumulative Return of Benchmark : {cum_ret_bench}")  		   	  			  	 		  		  		    	 		 		   		 		  
    print()  		   	  			  	 		  		  		    	 		 		   		 		  
    print(f"Standard Deviation of Theoretical Strat: {std_daily_ret}")  		   	  			  	 		  		  		    	 		 		   		 		  
    print(f"Standard Deviation of Benchmark : {std_daily_ret_bench}")  		   	  			  	 		  		  		    	 		 		   		 		  
    print()  		   	  			  	 		  		  		    	 		 		   		 		  
    print(f"Average Daily Return of Theoretical Strat: {avg_daily_ret}")  		   	  			  	 		  		  		    	 		 		   		 		  
    print(f"Average Daily Return of Benchmark : {avg_daily_ret_bench}")  		   	  			  	 		  		  		    	 		 		   		 		  
    print()  		   	  			  	 		  		  		    	 		 		   		 		  
    print(f"Final Portfolio Value Theoretical Strat: {man_returns[-1]}")
    print(f"Final Portfolio Value Benchmark: {bench_returns[-1]}")  