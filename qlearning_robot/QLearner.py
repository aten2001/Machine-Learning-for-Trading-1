"""  		   	  			  	 		  		  		    	 		 		   		 		  
Template for implementing QLearner  (c) 2015 Tucker Balch  		   	  			  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
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
  		   	  			  	 		  		  		    	 		 		   		 		  
import numpy as np  		   	  			  	 		  		  		    	 		 		   		 		  
import random as rand  		   	  			  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
class QLearner(object):  		   	  			  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
    def __init__(self, \
        num_states=100, \
        num_actions = 4, \
        alpha = 0.2, \
        gamma = 0.9, \
        rar = 0.5, \
        radr = 0.99, \
        dyna = 0, \
        verbose = False):  		   	  			  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
        self.rar = rar
        self.radr = radr
        self.dyna = dyna
        self.num_states = num_states
        self.num_actions = num_actions
        self.gamma = gamma
        self.alpha = alpha
        self.q = np.zeros((self.num_states, self.num_actions))
        self.saved_actions = {"mem":[]}
        self.verbose = verbose  		   	  			  	 		  		  		    	 		 		   		 		  	   	  			  	 		  		  		    	 		 		   		 		  
        self.s = 0  		   	  			  	 		  		  		    	 		 		   		 		  
        self.a = 0  
        self.setup_table(self.dyna)
    	  			  	 		  		  		    	 		 		   		 		  

    
    def querysetstate(self, s):  		   	  			  	 		  		  		    	 		 		   		 		  
        """  		   	  			  	 		  		  		    	 		 		   		 		  
        @summary: Update the state without updating the Q-table  		   	  			  	 		  		  		    	 		 		   		 		  
        @param s: The new state  		   	  			  	 		  		  		    	 		 		   		 		  
        @returns: The selected action  		   	  			  	 		  		  		    	 		 		   		 		  
        """  		   	  			  	 		  		  		    	 		 		   		 		  
        curr_actions = self.num_actions - 1
        self.s = s   	  			  	 		  		  		    	 		 		   		 		  
        action = rand.randint(0, curr_actions) 
        if rand.random() > self.rar:
            action = np.argmax(self.q[s])
        #self.rar = self.rar * self.radr
        #self.a = action	   	  			  	 		  		  		    	 		 		   		 		  
        if self.verbose: print(f"s = {s}, a = {action}")  		   	  			  	 		  		  		    	 		 		   		 		  
        return action  		   	  			  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
    def query(self,s_prime,r):  		   	  			  	 		  		  		    	 		 		   		 		  
        """  		   	  			  	 		  		  		    	 		 		   		 		  
        @summary: Update the Q table and return an action  		   	  			  	 		  		  		    	 		 		   		 		  
        @param s_prime: The new state  		   	  			  	 		  		  		    	 		 		   		 		  
        @param r: The ne state  		   	  			  	 		  		  		    	 		 		   		 		  
        @returns: The selected action  		   	  			  	 		  		  		    	 		 		   		 		  
        """ 
        #self.setup_table(self.dyna)
        old_value = self.q[self.s, self.a]
        l_rate = self.alpha
        learned_value = r + self.gamma * np.max(self.q[s_prime, :])
        #self.q[self.s, self.a] = (1-self.alpha) * old_value + l_rate * learned_value
        self.update(old_value, l_rate, learned_value)
        action = np.argmax(self.q[s_prime,:])
        if self.rar > rand.uniform(0.0, 1.0): action = rand.randint(0, self.num_actions - 1)
        self.saved_actions["mem"].append((self.s, self.a, s_prime, r))
        self.rar = self.rar * self.radr
        self.dyna_hallucinate(self.dyna, s_prime, r)
        self.s = s_prime
        self.a = action
        

        if self.verbose: print(f"s = {s_prime}, a = {action}, r={r}") 
        
        return action  		   	  			  	 		  		  		    	 		 		   		 		  
    
    def setup_table(self, dyna):
        table_size = (self.num_states, self.num_actions)
        self.q = np.random.uniform(-1, 1, table_size)
        
        if dyna:
            self.t_count = np.full((self.num_states,self.num_actions,self.num_states),0.00001)
            self.T = self.t_count / self.t_count.sum(axis=2, keepdims=True)
            self.R = np.full(table_size,-1.0)
    
    def update(self, old_value, l_rate, learned_value):
       self.q[self.s, self.a] = (1-self.alpha) * old_value + l_rate * learned_value
    
    def dyna_hallucinate(self, dyna, s_prime, r):
        if self.dyna:
            self.R[self.s, self.a] = (1 - self.alpha) * self.R[self.s, self.a] + (self.alpha * r)
            self.t_count[self.s, self.a, s_prime] = self.t_count[self.s, self.a, s_prime] + 1
            self.T = self.t_count / self.t_count.sum(axis=2, keepdims=True)

            dyna_a = np.random.randint(0, self.num_actions, size=self.dyna)
            dyn_s = np.random.randint(0, self.num_states, size=self.dyna)
            exp_tuple_len = len(self.saved_actions["mem"])
            random_tuple = np.random.randint(exp_tuple_len, size=self.dyna)
           
            for i in range(dyn_s.shape[0]):
                dyna_sp = self.saved_actions["mem"][random_tuple[i]][2]
                r = self.saved_actions["mem"][random_tuple[i]][3]
                s = self.saved_actions["mem"][random_tuple[i]][0]
                a = self.saved_actions["mem"][random_tuple[i]][1]
                
                
                old_value = self.q[s, a]
                l_rate = self.alpha
                learned_value = r + self.gamma * np.max(self.q[dyna_sp, :])
                self.q[s, a] = (1-self.alpha) * old_value + l_rate * learned_value
  
        else:
            pass

    def author(self):
        return "shollister7"

if __name__=="__main__":  		   	  			  	 		  		  		    	 		 		   		 		  
    print("Remember Q from Star Trek? Well, this isn't him")  		   	  			  	 		  		  		    	 		 		   		 		  
