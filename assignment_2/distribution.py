# importing libraries
import numpy as np
import math
import matplotlib.pyplot as plt
import random

# no. of samples
n_samples = 10000

'''
Box Muller Method
'''

# calculating gaussian noise
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

# plotting the histogram with bin size = 20
plt.hist(x, bins = 20, color = "blue");
plt.title("Gaussian Distribution (random variable X)")
plt.xlabel("BOX-MULLER METHOD")
plt.show()
plt.hist(y, bins = 20, color = "orange");
plt.title("Gaussian Distribution (random variable Y)")
plt.xlabel("BOX-MULLER METHOD")
plt.show()


'''
Marsaglia Polar Method
'''

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

# plotting the histogram with bin size = 20
plt.hist(x, bins = 20, color = "blue");
plt.title("Gaussian Distribution (random variable X)")
plt.xlabel("MARSAGLIA POLAR METHOD")
plt.show()
plt.hist(y, bins = 20, color = "orange");
plt.title("Gaussian Distribution (random variable Y)")
plt.xlabel("MARSAGLIA POLAR METHOD")
plt.show()

