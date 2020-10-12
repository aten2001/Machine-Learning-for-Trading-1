   	  			  	 		  		  		    	 		 		   		 		  
import numpy as np  	
from scipy import stats	   	  			  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
class BagLearner(object):  		   	  			  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
    def __init__(self, learner, kwargs, bags, boost=False,verbose = False):
        self.verbose = verbose
        self.kwargs = kwargs
        self.bags = bags
        self.boost = boost

        learner_list = []
        for l in range(bags):
            learner_list.append(learner(**kwargs))
        self.learners = learner_list
        if verbose == True:
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
        results = []
        for learner in self.learners:
            results.append(learner.query(points))
        results = np.array(results)
        results = np.mean(results, 0)
        return results
    
    def get_learner_summary(self):
        return "This is the Bag Learner"
        	  			  	 		  		  		    	 		 		   		  	   	  			  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
if __name__=="__main__":  		   	  			  	 		  		  		    	 		 		   		 		  
    print("Bag Learner")  		   	  			  	 		  		  		    	 		 		   		 		  
