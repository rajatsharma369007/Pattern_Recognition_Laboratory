'''
This script helps in generating values from normal distribution. This scipt is
based on Box-Muller Method
'''

# importing Libraries
import numpy as np
import math
import random

# generating values
def gaussian_function(n_samples):
    x = []
    y = []
    for i in range(n_samples):
        '''
        U and V are independent random numbers, distributed uniformly on (0,1)
        '''
        U = random.uniform(0,1)
        V = random.uniform(0,1)
        x.append(math.sqrt(-2*math.log(U))*math.cos(2*math.pi*(V)))
        y.append(math.sqrt(-2*math.log(U))*math.sin(2*math.pi*(V)))
        
        
    # converting to numpy array
    x = np.array(x)
    y = np.array(y)
    
    return x, y
    
