'''
This script helps in generating values from normal distribution. This scipt is
based on Marsaglia Polar Method
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
        U and V are independent random numbers, distributed uniformly on (-1,1)
        S is the squared sum of U and V
        '''
        U = random.uniform(-1,1)
        V = random.uniform(-1,1)
        S = U*U + V*V
        x.append(U*math.sqrt(abs((-2*(math.log(S)))/S)))
        y.append(V*math.sqrt(abs((-2*(math.log(S)))/S)))
        
        
    # converting to numpy array
    x = np.array(x)
    y = np.array(y)
    
    return x, y
    