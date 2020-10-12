"""  		   	  			  	 		  		  		    	 		 		   		 		  
Test a learner.  (c) 2015 Tucker Balch  		   	  			  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
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
"""  		   	  			  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
import numpy as np  		   	  			  	 		  		  		    	 		 		   		 		  
import math  		   	  			  	 		  		  		    	 		 		   		 		  
import LinRegLearner as lrl  		   	  			  	 		  		  		    	 		 		   		 		  
import sys
import matplotlib.pyplot as plt	   	
import pandas as pd
import DTLearner as dt
import RTLearner as rt
import BagLearner as bl

def provided():
    if len(sys.argv) != 2:  		   	  			  	 		  		  		    	 		 		   		 		  
        print("Usage: python testlearner.py <filename>")  		   	  			  	 		  		  		    	 		 		   		 		  
        sys.exit(1)  		   	  			  	 		  		  		    	 		 		   		 		  
    inf = open(sys.argv[1])  		   	  			  	 		  		  		    	 		 		   		 		  
    data = np.array([list(map(str,s.strip().split(','))) for s in inf.readlines()])  
    data = data[1:, 1:]		   	  			  	 	
    data = data.astype("float")	  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
    # compute how much of the data is training and testing  		   	  			  	 		  		  		    	 		 		   		 		  
    train_rows = int(0.6* data.shape[0])  		   	  			  	 		  		  		    	 		 		   		 		  
    test_rows = data.shape[0] - train_rows  		   	  			  	 		  		  		    	 		 		   		 		  

    np.random.shuffle(data())	   	  			  	 		  		  		    	 		 		   		 		  
    # separate out training and testing data  		   	  			  	 		  		  		    	 		 		   		 		  
    trainX = data[:train_rows,0:-1]  		   	  			  	 		  		  		    	 		 		   		 		  
    trainY = data[:train_rows,-1]  		   	  			  	 		  		  		    	 		 		   		 		  
    testX = data[train_rows:,0:-1]  		   	  			  	 		  		  		    	 		 		   		 		  
    testY = data[train_rows:,-1]  		   	  			  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
    print(f"{testX.shape}")  		   	  			  	 		  		  		    	 		 		   		 		  
    print(f"{testY.shape}")  		   	  			  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
    # create a learner and train it  		   	  			  	 		  		  		    	 		 		   		 		  
    learner = lrl.LinRegLearner(verbose = True) # create a LinRegLearner  		   	  			  	 		  		  		    	 		 		   		 		  
    learner.addEvidence(trainX, trainY) # train it  		   	  			  	 		  		  		    	 		 		   		 		  
    print(learner.author())  		   	  			  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
    # evaluate in sample  		   	  			  	 		  		  		    	 		 		   		 		  
    predY = learner.query(trainX) # get the predictions  		   	  			  	 		  		  		    	 		 		   		 		  
    rmseIn = math.sqrt(((trainY - predY) ** 2).sum()/trainY.shape[0])  		   	  			  	 		  		  		    	 		 		   		 		   		   	  			  	 		  		  		    	 		 		   		 		  
    print("In sample results LinReg")  		   	  			  	 		  		  		    	 		 		   		 		  
    print(f"RMSE: {rmseIn}")  		   	  			  	 		  		  		    	 		 		   		 		  
    c = np.corrcoef(predY, y=trainY)  		   	  			  	 		  		  		    	 		 		   		 		  
    print(f"corr: {c[0,1]}")  		   	  			  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
    # evaluate out of sample  		   	  			  	 		  		  		    	 		 		   		 		  
    predY = learner.query(testX) # get the predictions  		   	  			  	 		  		  		    	 		 		   		 		  
    rmseOut = math.sqrt(((testY - predY) ** 2).sum()/testY.shape[0])  		   	  			  	 		  		  		    	 		 		   		 		  
    print()  		   	  			  	 		  		  		    	 		 		   		 		  
    print("Out of sample results LinReg")  		   	  			  	 		  		  		    	 		 		   		 		  
    print(f"RMSE: {rmseOut}")  		   	  			  	 		  		  		    	 		 		   		 		  
    c = np.corrcoef(predY, y=testY)  		   	  			  	 		  		  		    	 		 		   		 		  
    print(f"corr: {c[0,1]}")

    """plt.figure(0)
    plt.plot(rmseIn,label="In Sample Errors (RMSE)")
    plt.plot(rmseOut, label="Out Sample Errors (RMSE)")
    plt.xlim(0,100)
    plt.title("Figure")
    plt.legend(loc="lower right")"""


    # TEST DT LEARNER
    import DTLearner as dt
    learnerD = dt.DTLearner(leaf_size = 1, verbose = False) # constructor
    learnerD.addEvidence(trainX, trainY) # training step
    YD = learner.query(testX) # query 

    # evaluate in sample  		   	  			  	 		  		  		    	 		 		   		 		  
    predY = learnerD.query(trainX) # get the predictions  		   	  			  	 		  		  		    	 		 		   		 		  
    rmse = math.sqrt(((trainY - predY) ** 2).sum()/trainY.shape[0])  		   	  			  	 		  		  		    	 		 		   		 		  
    print()  		   	  			  	 		  		  		    	 		 		   		 		  
    print("In sample results DTLearner")  		   	  			  	 		  		  		    	 		 		   		 		  
    print(f"RMSE: {rmse}")  		   	  			  	 		  		  		    	 		 		   		 		  
    c = np.corrcoef(predY, y=trainY)  		   	  			  	 		  		  		    	 		 		   		 		  
    print(f"corr: {c[0,1]}")  		   	  			  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
    # evaluate out of sample  		   	  			  	 		  		  		    	 		 		   		 		  
    predY = learnerD.query(testX) # get the predictions  		   	  			  	 		  		  		    	 		 		   		 		  
    rmse = math.sqrt(((testY - predY) ** 2).sum()/testY.shape[0])  		   	  			  	 		  		  		    	 		 		   		 		  
    print()  		   	  			  	 		  		  		    	 		 		   		 		  
    print("Out of sample results DT Learner")  		   	  			  	 		  		  		    	 		 		   		 		  
    print(f"RMSE: {rmse}")  		   	  			  	 		  		  		    	 		 		   		 		  
    c = np.corrcoef(predY, y=testY)  		   	  			  	 		  		  		    	 		 		   		 		  
    print(f"corr: {c[0,1]}")	 
    
    #TEST RT LEARNER
    import RTLearner as rt
    learnerD = rt.RTLearner(leaf_size = 1, verbose = False) # constructor
    learnerD.addEvidence(trainX, trainY) # training step
    YD = learner.query(testX) # query 

    # evaluate in sample  		   	  			  	 		  		  		    	 		 		   		 		  
    predY = learnerD.query(trainX) # get the predictions  		   	  			  	 		  		  		    	 		 		   		 		  
    rmse = math.sqrt(((trainY - predY) ** 2).sum()/trainY.shape[0])  		   	  			  	 		  		  		    	 		 		   		 		  
    print()  		   	  			  	 		  		  		    	 		 		   		 		  
    print("In sample results RTLearner")  		   	  			  	 		  		  		    	 		 		   		 		  
    print(f"RMSE: {rmse}")  		   	  			  	 		  		  		    	 		 		   		 		  
    c = np.corrcoef(predY, y=trainY)  		   	  			  	 		  		  		    	 		 		   		 		  
    print(f"corr: {c[0,1]}")  		   	  			  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
    # evaluate out of sample  		   	  			  	 		  		  		    	 		 		   		 		  
    predY = learnerD.query(testX) # get the predictions  		   	  			  	 		  		  		    	 		 		   		 		  
    rmse = math.sqrt(((testY - predY) ** 2).sum()/testY.shape[0])  		   	  			  	 		  		  		    	 		 		   		 		  
    print()  		   	  			  	 		  		  		    	 		 		   		 		  
    print("Out of sample results RT Learner")  		   	  			  	 		  		  		    	 		 		   		 		  
    print(f"RMSE: {rmse}")  		   	  			  	 		  		  		    	 		 		   		 		  
    c = np.corrcoef(predY, y=testY)  		   	  			  	 		  		  		    	 		 		   		 		  
    print(f"corr: {c[0,1]}")	

    
def get_data():
    if len(sys.argv) != 2:  		   	  			  	 		  		  		    	 		 		   		 		  
        print("Usage: python testlearner.py <filename>")  		   	  			  	 		  		  		    	 		 		   		 		  
        sys.exit(1)  		   	  			  	 		  		  		    	 		 		   		 		  
    inf = open(sys.argv[1])  		   	  			  	 		  		  		    	 		 		   		 		  
    data = np.array([list(map(str,s.strip().split(','))) for s in inf.readlines()])  
    data = data[1:, 1:]		   	  			  	 	
    data = data.astype("float")	  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
    # compute how much of the data is training and testing  		   	  			  	 		  		  		    	 		 		   		 		  
    train_rows = int(0.6* data.shape[0])  		   	  			  	 		  		  		    	 		 		   		 		  
    test_rows = data.shape[0] - train_rows  		   	  			  	 		  		  		    	 		 		   		 		  

    np.random.shuffle(data)	   	  			  	 		  		  		    	 		 		   		 		  
    # separate out training and testing data  		   	  			  	 		  		  		    	 		 		   		 		  
    trainX = data[:train_rows,0:-1]  		   	  			  	 		  		  		    	 		 		   		 		  
    trainY = data[:train_rows,-1]  		   	  			  	 		  		  		    	 		 		   		 		  
    testX = data[train_rows:,0:-1]  		   	  			  	 		  		  		    	 		 		   		 		  
    testY = data[train_rows:,-1]

    return trainX, trainY, testX, testY 

def train_test_learner(trainX, trainY, testX, testY, learner_arg, num_iterations=1, 
    max_leaf_size=None, max_bag_size=None, **kwargs):

    if max_leaf_size is None and max_bag_size is None:
        print ("Please specify the max_leaf_size or max_bag_size and try again;")
        print ("Returning fake data filled with zeros for now")
        return np.zeros((1, 1)), np.zeros((1, 1)), np.zeros((1, 1)), np.zeros((1, 1))

    max_val = max_leaf_size or max_bag_size
    RMSEin = np.zeros((max_val, num_iterations))
    RMSEout = np.zeros((max_val, num_iterations))
    c_in = np.zeros((max_val, num_iterations))
    c_out = np.zeros((max_val, num_iterations))

    # Train the learner and record RMSEs
    for i in range(1, max_val):
        for j in range(num_iterations):
            # Create a learner and train it
            if max_leaf_size is not None:
                learner = learner_arg(leaf_size=i, **kwargs)
            elif max_bag_size is not None:
                learner = learner_arg(bags=i, **kwargs)
            learner.addEvidence(trainX, trainY)

            # Evaluate in-sample
            predY = learner.query(trainX)
            rmse = math.sqrt(((trainY - predY) ** 2).sum()/trainY.shape[0])
            RMSEin[i, j] = rmse
            c = np.corrcoef(predY, y=trainY)
            c_in[i, j] = c[0, 1]
       
    

            # Evaluate out-of-sample
            predY = learner.query(testX)
            rmse = math.sqrt(((testY - predY) ** 2).sum()/testY.shape[0])
            RMSEout[i, j] = rmse
            c = np.corrcoef(predY, y=testY)
            c_out[i, j] = c[0, 1]
          
    
    # Get the means of RMSEs from all iterations
    RMSEin_mean = np.mean(RMSEin, axis=1)
    RMSEout_mean = np.mean(RMSEout, axis=1)
    c_inMean = np.mean(c_in, axis=1)
    c_outMean = np.mean(c_out, axis=1)

    return RMSEin_mean, RMSEout_mean, c_inMean, c_outMean

def plot_results(in_sample, out_of_sample, title, xlabel, ylabel, 
    legend_loc="lower right", xaxis_length=1):
    
    xaxis = np.arange(1, xaxis_length + 1)
    plt.plot(xaxis, in_sample, label="in-sample", linewidth=2.0)
    plt.plot(xaxis, out_of_sample, label="out of sample", linewidth=2.0)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(loc=legend_loc)
    plt.title(title)
    plt.savefig("{}.png".format("RMSEs for DTLearner vs RT Learner"))
  		   	  			  	 		  		  		    	 		 		   		 		  
if __name__=="__main__":  		   	  			  	 		  		  		    	 		 		   		 		  

    trainX, trainY, testX, testY = get_data()
    RMSEin_mean, RMSEout_mean, c_in, c_out = train_test_learner(trainX, trainY, testX, testY,
                        dt.DTLearner, max_leaf_size=200, num_iterations=1)

    xaxis = np.arange(1, 200+ 1)
    plt.plot(xaxis, c_in, label="in-sample DT Learner", linewidth=2.0)
    plt.plot(xaxis, c_out, label="out-of-sample DT Learner", linewidth=2.0)
    plt.xlabel("Leaf Size")
    plt.ylabel("Correlations")
    plt.legend(loc="lower right")
    plt.title("Correlations for DTLearner vs RT Learner")
  

    RMSEin_mean, RMSEout_mean, c_in, c_out = train_test_learner(trainX, trainY, testX, testY,
                        rt.RTLearner, max_leaf_size=200, num_iterations=1)
    xaxis = np.arange(1, 200+ 1)
    plt.plot(xaxis, c_in, label="in-sample RT Learner", linewidth=2.0)
    plt.plot(xaxis, c_out, label="out-of-sample RT Learner", linewidth=2.0)
    plt.xlabel("Leaf Size")
    plt.ylabel("Correlations")
    plt.title("Correlations for DTLearner vs RT Learner")
    plt.legend(loc="lower right")
    plt.savefig("{}.png".format("dtvsRtCorr.png"))

    #max_leaf_size = 20
    #trainX, trainY, testX, testY = get_data()
    #RMSEin_mean, RMSEout_mean = train_test_learner(trainX, trainY, testX, testY,
    # bl.BagLearner, max_bag_size=20, num_iterations=1, learner=dt.DTLearner, kwargs={} )
    
    #plot_results(RMSEin_mean, RMSEout_mean, "RMSEs for BagLearner using DTLearners", 
    #"Leaf size", "Root Mean Squared Error (RMSE)", xaxis_length=20)
    



