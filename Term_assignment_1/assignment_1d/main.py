'''
This script helps to calculate the mahalanobis distance between the mean and
a arbitrary point x, given the covariance matrix. This is part of pattern 
recognition assignment-1d
'''

# importing libraries
import numpy as np
import math
import random

# number of dimensions
dim = int(input('Enter the number of dimension: '))
# number of sample points
n = int(input('Enter the number of sample: '))

arr = []
for i in range(n*dim):
    arr.append(random.randint(-20, 20))
    
points = np.array(arr).reshape(n, dim)

# mean vector
mean = np.mean(points, axis=0)

# creating the covariance matrix
cov_matrix = np.cov(points.T)

# inverse covariance matrix
inv_cov = np.linalg.inv(cov_matrix)

# selecting an arbitrary random point
x_random = random.choice(points)


# calculating the mahalanobis distance
distance = math.sqrt(np.matmul(np.matmul((x_random - mean).T, inv_cov), (x_random - mean)))

# result
print('\nGiven covariance matrix: \n{}\n\nMean : {}\n\nArbitrary point: {}\n' \
      .format(cov_matrix, mean, x_random))
print('Mahalanobis distance : {}'.format(distance))

