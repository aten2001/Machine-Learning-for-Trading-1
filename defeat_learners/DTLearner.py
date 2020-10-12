"""  		   	  			  	 		  		  		    	 		 		   		 		  
A simple wrapper for linear regression.  (c) 2015 Tucker Balch  		   	  			  	 		  		  		    	 		 		   		 		  
Note, this is NOT a correct DTLearner; Replace with your own implementation.  		   	  			  	 		  		  		    	 		 		   		 		  
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
  		   	  			  	 		  		  		    	 		 		   		 		  
import numpy as np  		   	  			  	 		  		  		    	 		 		   		 		  
import warnings  		   	  			  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
class DTLearner(object):  		   	  			  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
    def __init__(self, leaf_size=1, verbose = False):  		   	  			  	 		  		  		    	 		 		   		 		  
        self.leaf_size = leaf_size
        self.verbose = verbose
        self.tree = np.array([])	   	  			  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
    def author(self):  		   	  			  	 		  		  		    	 		 		   		 		  
        return 'shollister7' # replace tb34 with your Georgia Tech username

    def buildTree(self, data, Y):
        """
        @summary Builds DT by splitting best feature, which is xi
        that has highest absolute correlation with Y.
        @param np array of features (X data)
        @param np array of values 
        @returns np array which is tree [leaf, data.Y, leftchild(None), rightChild(None)]
        """

        leaf = np.array([-1, np.mean(Y), np.nan, np.nan])
        
        if data.shape[0] == 1: return leaf
        if len(np.unique(Y)) == 1: return leaf
        if data.shape[0] <= self.leaf_size: return leaf
        
        #find best i to split on, using coeffecients
        possible_feats = range(data.shape[1])
        coeffecients = []
        for index in possible_feats:
            coMatrix = np.corrcoef(data[:,index], Y)
            coeff = coMatrix[0,1]
            coeffecients.append((np.abs(coeff),index))
        
        coeffecients.sort(reverse=True)
        i= coeffecients[0][1]
        #print("max value is {}".format(i))
        splitVal = np.median(data[:,i])
        
        left = data[:, i] <= splitVal

        if ((np.all(left)) or np.all(~left)):
            return leaf

        #lefttree = self.buildTree(data[data[:,small]<=splitVal] , Y[data[:,small]<=splitVal])
        #righttree= self.buildTree(data[data[:,~small]>splitVal], Y[data[:,~small]>splitVal])

        lefttree = self.buildTree(data[left,:], Y[left])
        righttree = self.buildTree(data[~left,:], Y[~left])

        if lefttree.ndim == 1:
            righttree_start = 2
        elif lefttree.ndim > 1:
            righttree_start = lefttree.shape[0] + 1
        root = np.array([i, splitVal, 1, righttree_start])

        #result = np.concatenate((root, lefttree, righttree))
        return np.vstack((root, lefttree, righttree))
          	 		  		  		    	 		 		   		 		  
    def addEvidence(self,dataX,dataY):  		   	  			  	 		  		  		    	 		 		   		 		  
        """  		   	  			  	 		  		  		    	 		 		   		 		  
        @summary: Add training data to learner  		   	  			  	 		  		  		    	 		 		   		 		  
        @param dataX: X values of data to add  		   	  			  	 		  		  		    	 		 		   		 		  
        @param dataY: the Y training values  		   	  			  	 		  		  		    	 		 		   		 		  
        """ 
        self.tree = self.buildTree(dataX, dataY)   	  			  	 		  		  		    	 		 		   		 		  

    def search(self, point, row):
        #print("shape is {}".format(self.tree.shape))
        feat , splitVal = self.tree[row, 0:2]
        if feat == -1:
            return splitVal
        elif point[int(feat)] <= splitVal:
            result = self.search(point, row + int(self.tree[row,2]))
        else:
            result = self.search(point, row+int(self.tree[row,3]))
        return result
    
    def query(self,points):  		   	  			  	 		  		  		    	 		 		   		 		  
        """  		   	  			  	 		  		  		    	 		 		   		 		  
        @summary: Estimate a set of test points given the model we built.  		   	  			  	 		  		  		    	 		 		   		 		  
        @param points: should be a numpy array with each row corresponding to a specific query.  		   	  			  	 		  		  		    	 		 		   		 		  
        @returns the estimated values according to the saved model.		   	  			  	 		  		  		    	 		 		   		 		  
        """
        results = []
        for p in points:
            results.append(self.search(p, 0))
        return np.asarray(results)	   	  			  	 		  		  		    	 		 		   		 		  
        	   	  			  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
if __name__=="__main__":  		   	  			  	 		  		  		    	 		 		   		 		  
    print("DT Learner")