   	  			  	 		  		  		    	 		 		   		 		  
import numpy as np  		   	  			  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
class RTLearner(object):  		   	  			  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
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
        """possible_feats = range(data.shape[1])
        coeffecients = []
        for index in possible_feats:
            rand_i =np.random.choice(data.shape[1])
            randRows =[np.random.randint(0, data.shape[0]), np.random.randint(0, data.shape[0])]
            if randRows[0] == randRows[1]:
                break"""

        for attempt in range(0,10):
            rand_i = [np.random.randint(0, data.shape[0]), np.random.randint(0, data.shape[0])]
            rand_ft = np.random.randint(0, data.shape[1])
            #if both rows diff, break 
            if data[rand_i[1], rand_ft] != data[rand_i[0], rand_ft]:
                break
        #after 10 tries just make a leaf
        if data[rand_i[1], rand_ft] == data[rand_i[0], rand_ft]:
            return leaf;

        total = (data[rand_i[0], rand_ft] + data[rand_i[1], rand_ft])
        mean = total / 2
        split_val = mean
        lefttree = self.buildTree(data[(data[:, rand_ft] <= split_val), :], Y[(data[:, rand_ft] <= split_val)]);
        righttree = self.buildTree(data[(data[:, rand_ft] > split_val), :], Y[(data[:, rand_ft] > split_val)]);

        if lefttree.ndim == 1:
            righttree_start = 2
        elif lefttree.ndim > 1:
            righttree_start = lefttree.shape[0] + 1
        root = np.array([rand_ft, split_val, 1, righttree_start], dtype=float)

        #result = np.concatenate((root, lefttree, righttree))"""
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
    print("RT Learner")  		   	  			  	 		  		  		    	 		 		   		 		  
