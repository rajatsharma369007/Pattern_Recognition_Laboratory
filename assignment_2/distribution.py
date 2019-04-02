# importing libraries
from utils import box_muller
from utils import marsaglia_polar
from utils import plot

# no. of samples
n_samples = 10000


'''
Box-Muller Method
'''

# generating values
'''
Class : gaussian_function()
Arguments : n_samples ---> no. of samples
'''
x, y = box_muller.gaussian_function(n_samples)

# plotting the gaussian distribution
'''
Class : histogram()
Arguments : x ---> random variable X
            y ---> random variable Y
     bin_size ---> width span of each bar
        label ---> xlabel
'''
plot.histogram(x, y, 20, "Box-Muller Method")


'''
Marsaglia Polar Method
'''

# generating values
'''
Class : gaussian_function()
Arguments : n_samples ---> no. of samples
'''
w, z = marsaglia_polar.gaussian_function(n_samples)

# plotting the gaussian distribution
'''
Class : histogram()
Arguments : w ---> random variable X
            z ---> random variable Y
     bin_size ---> width span of each bar
        label ---> xlabel
'''
plot.histogram(w, z, 20, "Marsaglia Polar Method")




