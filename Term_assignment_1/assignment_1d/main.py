'''
This script helps to calculate the mahalanobis distance between the mean and
a arbitrary point x, given the covariance matrix. This is part of pattern 
recognition assignment-1d
'''

# importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import random

df = pd.read_excel("/home/rajat/Documents/8th sem/Pattern_recognition/Pattern_Recognition_Laboratory/dataset/points.xlsx", header=None)

x = np.array(df[0])
y = np.array(df[1])

# creating coordinates
points = np.stack([x, y]).T

# mean vector
mean = np.mean(points, axis=0)

# creating the covariance matrix
cov_matrix = np.cov(points.T)

# inverse covariance matrix
inv_cov = np.linalg.inv(cov_matrix)

# selecting an arbitrary random point
x_random = random.choice(points)

# plotting points
plt.scatter(x, y)
plt.xlabel('X')
plt.ylabel('Y')
plt.xlim(min(x)-2, max(x)+2)
plt.ylim(min(y)-2, max(y)+2)
plt.show()

# calculating the mahalanobis distance
distance = math.sqrt(np.matmul(np.matmul((x_random - mean).T, inv_cov), (x_random - mean)))

# result
print('Given covariance matrix: \n{}\n\nMean : {}\n\nArbitrary point: {}\n' \
      .format(cov_matrix, mean, x_random))
print('Mahalanobis distance : {}'.format(distance))

