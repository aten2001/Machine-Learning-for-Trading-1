"""Assess a betting strategy.  		   	  			  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
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
GT ID: 903304661(replace with your GT ID)  		   	  			  	 		  		  		    	 		 		   		 		  
"""  		   	  			  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
import numpy as np

import matplotlib.pyplot as plt


def author():  		   	  			  	 		  		  		    	 		 		   		 		  
        return 'shollister7' # replace tb34 with your Georgia Tech username.  		   	  			  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
def gtid():  		   	  			  	 		  		  		    	 		 		   		 		  
	return 903304661 # replace with your GT ID number  		   	  			  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
def get_spin_result(win_prob):  		   	  			  	 		  		  		    	 		 		   		 		  
	result = False  		   	  			  	 		  		  		    	 		 		   		 		  
	if np.random.random() <= win_prob:  		   	  			  	 		  		  		    	 		 		   		 		  
		result = True  		   	  			  	 		  		  		    	 		 		   		 		  
	return result  		   	  			  	 		  		  		    	 		 		   		 		  

def test_code():  		   	  			  	 		  		  		    	 		 		   		 		  
	win_prob = 0.4737 # set appropriately to the probability of a win  		   	  			  	 		  		  		    	 		 		   		 		  
	np.random.seed(gtid()) # do this only once  		   	  			  	 		  		  		    	 		 		   		 		     	  			  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
	# add your code here to implement the experiments
	exp1_fig1(win_prob)
	exp1_fig2(win_prob)
	exp1_fig3(win_prob)
	exp2_fig4(win_prob)
	exp2_fig5(win_prob)
	

def run_simulator(prob):
	winnings = np.full((1000,),80)
	episode_winnings = 0
	bet_amount = 1
	for i in range(1000):
		if episode_winnings >= 80:
			continue
		won = get_spin_result(prob)
		if won == True:
			episode_winnings = episode_winnings + bet_amount
			winnings[i] = episode_winnings
			bet_amount = 1		
		else:
			episode_winnings = episode_winnings - bet_amount
			bet_amount = bet_amount * 2
			winnings[i] = episode_winnings
	return winnings	

def run_realistic_simulator(prob):
	winnings = np.full((1000,),80)
	episode_winnings = 0
	bet_amount = 1
	for i in range(1000):
		if episode_winnings >= 80:
			continue
		if episode_winnings <= -256:
			winnings[i] = -256
			continue
		if (episode_winnings < 0):
			if bet_amount > 256 - episode_winnings:
				bet_amount = episode_winnings
		elif (bet_amount > 256 + episode_winnings):
			bet_amount = episode_winnings

		won = get_spin_result(prob)
		if won == True:
			episode_winnings = episode_winnings + bet_amount
			winnings[i] = episode_winnings
			bet_amount = 1		
		else:
			episode_winnings = episode_winnings - bet_amount
			bet_amount = bet_amount * 2
			winnings[i] = episode_winnings
	return winnings	

def exp1_fig1(prob):
	results = np.empty((0, 1000))
	plt.figure(0)
	for i in range(10):
		run = run_simulator(prob)
		plt.plot(run, label=i)
		results = np.append(results, [run], axis = 0)
	plt.title('Experiment 1: Figure 1')
	plt.xlabel('Bet Number')
	plt.ylabel('Return ($)')
	plt.xlim((0,300))
	plt.ylim((-256, 100))
	plt.legend(loc="lower right")
	plt.savefig('figure1.png')

def exp1_fig2(prob):
	results = np.empty((0, 1000))
	for i in range(1000):
		run = run_simulator(prob)
		results = np.append(results, [run], axis = 0)
	
	mean = np.mean(results, axis=0)
	std = np.std(results, axis=0)

	up_band = mean + std 
	low_band = mean - std

	plt.figure(1)
	plt.plot(mean, label='mean')
	plt.plot(up_band, label='mean + 1 std')
	plt.plot(low_band, label='mean - 1 std')
	plt.xlim((0,300))
	plt.ylim((-256, 100))
	plt.title('Experiment 1: Figure 2 (Mean + / - Std)')
	plt.xlabel('Bet Number')
	plt.ylabel('Return ($)')
	plt.xlim((0,300))
	plt.ylim((-256, 100))
	plt.legend(loc="lower right")
	plt.savefig('figure2.png')

def exp1_fig3(prob):
	results = np.empty((0, 1000))
	for i in range(1000):
		run = run_simulator(prob)
		results = np.append(results, [run], axis = 0)
	
	median = np.median(results, axis=0)
	std = np.std(results, axis=0)

	up_band = median + std 
	low_band = median - std

	plt.figure(2)
	plt.plot(median, label='median')
	plt.plot(up_band, label='median + 1 std')
	plt.plot(low_band, label='median - 1 std')
	plt.xlim((0,300))
	plt.ylim((-256, 100))
	plt.title('Experiment 1: Figure 3 (Median + / - Std)')
	plt.xlabel('Bet Number')
	plt.ylabel('Return ($)')
	plt.xlim((0,300))
	plt.ylim((-256, 100))
	plt.legend(loc="lower right")
	plt.savefig('figure3.png')

def exp2_fig4(prob):
	results = np.empty((0, 1000))
	for i in range(1000):
		run = run_realistic_simulator(prob)
		results = np.append(results, [run], axis = 0)

	mean = np.mean(results, axis=0)
	std = np.std(results, axis=0)

	up_band = mean + std 
	low_band = mean - std

	plt.figure(3)
	plt.plot(mean, label='mean')
	plt.plot(up_band, label='mean + 1 std')
	plt.plot(low_band, label='mean - 1 std')
	plt.xlim((0,300))
	plt.ylim((-256, 100))
	plt.title('Experiment 2: Figure 4 (Mean + / - Std)')
	plt.xlabel('Bet Number')
	plt.ylabel('Return ($)')
	plt.xlim((0,300))
	plt.ylim((-256, 100))
	plt.legend(loc="lower right")
	plt.savefig('figure4.png')

def exp2_fig5(prob):
	results = np.empty((0, 1000))
	for i in range(1000):
		run = run_realistic_simulator(prob)
		results = np.append(results, [run], axis = 0)

	median = np.median(results, axis=0)
	std = np.std(results, axis=0)

	up_band = median + std 
	low_band = median - std

	plt.figure(4)
	plt.plot(median, label='median')
	plt.plot(up_band, label='median + 1 std')
	plt.plot(low_band, label='median - 1 std')
	plt.xlim((0,300))
	plt.ylim((-256, 100))
	plt.title('Experiment 2: Figure 5 (Median + / - Std)')
	plt.xlabel('Bet Number')
	plt.ylabel('Return ($)')
	plt.xlim((0,300))
	plt.ylim((-256, 100))
	plt.legend(loc="lower right")
	plt.savefig('figure5.png')

				  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
if __name__ == "__main__":  		   	  			  	 		  		  		    	 		 		   		 		  
    test_code()  		   	  			  	 		  		  		    	 		 		   		 		  
