import matplotlib.pyplot as plt   	  			  	 		  		  		    	 		 		   		 		  
import datetime as dt  		   	  			  	 		  		  		    	 		 		   		 		  
import pandas as pd  		   	  			  	 		  		  		    	 		 		   		 		  
import util as ut  		   	  			  	 		  		  		    	 		 		   		 		  
import random
import QLearner as ql
import indicators as ind
import util as ut
import numpy as np 
import marketsimcode as ms
import ManualStrategy
import StrategyLearner


def create_benchmark_tradesDF(df_prices, symbol="JPM"):
        dates = []
        dates.append(df_prices.index[3])
        dates.append(df_prices.index[len(df_prices.index)-2])
        symbols = [symbol,symbol]
        df_trades = pd.DataFrame(data = symbols, index = dates, columns = ['Symbol'])
        df_trades["Order"] = ["BUY", "SELL"]
        df_trades["Shares"] = [1000, 1000]
        df_trades.index.name = "Date"
        return df_trades

def compute_port_stats(portvals):
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

def experiment1():
    start_val = 100000
    symbol = "JPM"
    commission = 9.95
    impact = 0.005
    num_shares = 1000
    sd=dt.datetime(2008,1,1)
    ed=dt.datetime(2009,12,31)

    #In-Sample Q-Learning Strat Port Stats
    seed = 1021080103
    #seed = 1001090000
    #seed = 1111090000
    np.random.seed(seed)  		   	  			  	 		  		  		    	 		 		   		 		  
    random.seed(seed)
    stl = StrategyLearner.StrategyLearner( impact=impact)
    stl.addEvidence(symbol=symbol, sd=sd, ed=ed)
    df_trades = stl.testPolicy(symbol=symbol, sd=sd, ed=ed)
    df_trades = reformat_trades(symbol, df_trades)
    strat_portvals = ms.compute_portvals(df_trades, start_val=100000, commission=9.95, impact=0.005)
    cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = compute_port_stats(strat_portvals)
    strat_returns = strat_portvals["portfolio_totals"]
    normed_strat_returns = strat_portvals["portfolio_totals"] / strat_portvals["portfolio_totals"][0]

    #Man Strat Port Stats
    man = ManualStrategy.ManualStrategy()
    man_trades_df = man.testPolicy(symbol=symbol, sd=sd, ed=ed)
    man_portvals = ms.compute_portvals(man_trades_df, start_val=100000, commission=9.95, impact=0.005)
    cum_retM, avg_daily_retM, std_daily_retM, sharpe_ratioM = compute_port_stats(man_portvals)
    man_returns = man_portvals["portfolio_totals"]
    normed_man_returns = man_portvals["portfolio_totals"] / man_portvals["portfolio_totals"][0]

    #Benchmark Stats
    df_prices = ind.prepare_pricedf(symbol, sd, ed)
    bench_df = create_benchmark_tradesDF(df_prices)
    bench_portvals = ms.compute_portvals(bench_df, start_val = 100000, commission=9.95, impact=0.005)
    cum_retB, avg_daily_retB, std_daily_retB, sharpe_ratioB = compute_port_stats(bench_portvals)
    bench_returns = bench_portvals["portfolio_totals"]
    normed_bench_returns = bench_portvals["portfolio_totals"] / bench_portvals["portfolio_totals"][0]

    print(f"Date Range: {sd} to {ed}")  		   	  			  	 		  		  		    	 		 		   		 		  
    print()  		   	  			  	 		  		  		    	 		 		   		 		  
    print(f"Sharpe Ratio of Q Strat: {sharpe_ratio}")
    print(f"Sharpe Ratio of Man Strat: {sharpe_ratioM}")  		   	  			  	 		  		  		    	 		 		   		 		  
    print(f"Sharpe Ratio of Benchmark : {sharpe_ratioB}")   	  			  	 		  		  		    	 		 		   		 		  
    print()  		   	  			  	 		  		  		    	 		 		   		 		  
    print(f"Cumulative Return of Q Strat: {cum_ret}")
    print(f"Cumulative Return of Man Strat : {cum_retM}")		   	  			  	 		  		  		    	 		 		   		 		  
    print(f"Cumulative Return of Benchmark : {cum_retB}")  		   	  			  	 		  		  		    	 		 		   		 		  
    print()  		   	  			  	 		  		  		    	 		 		   		 		  
    print(f"Standard Deviation of Q Strat: {std_daily_ret}")
    print(f"Standard Deviation of Man Strat : {std_daily_retM}") 		   	  			  	 		  		  		    	 		 		   		 		  
    print(f"Standard Deviation of Benchmark : {std_daily_retB}")  		   	  			  	 		  		  		    	 		 		   		 		  
    print()  		   	  			  	 		  		  		    	 		 		   		 		  
    print(f"Average Daily Return of Q Strat: {avg_daily_ret}")
    print(f"Average Daily Return of Man Strat : {avg_daily_retM}") 		   	  			  	 		  		  		    	 		 		   		 		  
    print(f"Average Daily Return of Benchmark : {avg_daily_retB}")  		   	  			  	 		  		  		    	 		 		   		 		  
    print()  		   	  			  	 		  		  		    	 		 		   		 		  
    print(f"Final Portfolio Value Q Strat: {strat_returns[-1]}")
    print(f"Final Portfolio Value Man Strat: {man_returns[-1]}")
    print(f"Final Portfolio Value Benchmark: {bench_returns[-1]}")

    chart_df = pd.concat([normed_strat_returns, normed_man_returns, normed_bench_returns], axis=1)
    chart_df.columns = ['Q-Learning Strategy', 'Manual Strategy', 'Benchmark']
    chart_df.plot(grid=True, title='Strategy Learner vs Manual Strategy vs Benchmark (In-Sample)', use_index=True, color=['Red', 'Blue', 'Black'])
    #plt.show()
    plt.ylabel('Returns Normalized', fontsize=10)
    plt.xlabel('Dates', fontsize=10)
    plt.savefig("experiment1.png")

def reformat_trades(symbol, df_trades):
    orders = []
    for row in range(df_trades.index.shape[0]):
        if df_trades.iloc[row]["Shares"] < 0:
            orders.append("SELL")
        else:
            orders.append("BUY")

    share_orders = df_trades["Shares"].tolist()
    abs_orders = [abs(x) for x in share_orders]
    
    symbols = []
    for i in range(len(share_orders)):
            symbols.append(symbol)
    
    dates = df_trades.index.tolist()
    df_trades = pd.DataFrame(data = symbols, index = dates, columns = ['Symbol'])
    df_trades["Order"] = orders
    df_trades["Shares"] = abs_orders
    df_trades.index.name = "Date"
    #print(df_trades)
    return df_trades

def author():
    return 'shollister7'

if __name__ == "__main__":
    experiment1()


