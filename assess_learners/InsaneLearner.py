import numpy as np  	
from scipy import stats
import BagLearner as bl
import LinRegLearner as lrl
import DTLearner as dtl	  			  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
class InsaneLearner(object):  		   	  			  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
    def __init__(self, verbose = False, count=20):
        self.verbose = verbose
        self.count = count
        learner_list = []
        for l in range(self.count):
            learner_list.append(bl.BagLearner(lrl.LinRegLearner, kwargs={}, bags = 20, verbose = self.verbose))
        self.learners = learner_list
        if self.verbose == True:
            self.get_learner_summary()
        		  	 		  		  		    	 		 		   		 		   			  	 		  		  		    	 		 		   		 		  
    def author(self):  		   	  			  	 		  		  		    	 		 		   		 		  
        return 'shollister7' # replace tb34 with your Georgia Tech username
          	 		  		  		    	 		 		   		 		  
    def addEvidence(self,data,Y):  		   	  			  	 		  		  		    	 		 		   		 		  
        """  		   	  			  	 		  		  		    	 		 		   		 		  
        @summary: Add training data to learner  		   	  			  	 		  		  		    	 		 		   		 		  
        @param data: X values of data to add  		   	  			  	 		  		  		    	 		 		   		 		  
        @param Y: the Y training values  		   	  			  	 		  		  		    	 		 		   		 		  
        """
        for l in self.learners:
            rand_data_slice = np.random.choice(data.shape[0], data.shape[0])
            bag_x = data[rand_data_slice]
            bag_y = Y[rand_data_slice]
            l.addEvidence(bag_x, bag_y)

    def query(self,points):  		   	  			  	 		  		  		    	 		 		   		 		  
        """		   	  			  	 		  		  		    	 		 		   		 		  
        @summary: Estimate a set of test points given the model we built.  		   	  			  	 		  		  		    	 		 		   		 		  
        @param points: should be a numpy array with each row corresponding to a specific query.  		   	  			  	 		  		  		    	 		 		   		 		  
        @returns the estimated values according to the saved model.		   	  			  	 		  		  		    	 		 		   		 		  
        """

        dim = points.shape[0]
        output = np.empty((dim, 20))
        
        for ith_col in range(self.count):
            output[:,ith_col] = self.learners[ith_col].query(points)
        output = output.mean(1)
        return output
    
    def get_learner_summary(self):
        return "This is the INSANE learner"
		  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
if __name__=="__main__":  		   	  			  	 		  		  		    	 		 		   		 		  
    print("Insane Learner")  		   	  			  	 		  		  		    	 		 		   		 		  
