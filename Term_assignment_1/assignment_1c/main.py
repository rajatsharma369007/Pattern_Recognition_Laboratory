'''
This script helps to calculate the euclidean distance between two arbitrary 
points. This is part of pattern recognition assignment-1c
'''

# importing libraries
import numpy as np
import random
import math

# number of dimensions
dim = int(input('Enter the number of dimension: '))
# number of sample points
n = int(input('Enter the number of sample: '))

arr = []
for i in range(n*dim):
    arr.append(random.randint(-20, 20))
    
points = np.array(arr).reshape(n, dim)

# randomly choosing an arbitrary points
point1 = random.choice(points)
print('Point 1: ', point1)
point2 = random.choice(points)
print('Point 2: ', point2)

# calculate distance
distance = math.sqrt(np.sum((point2 - point1)**2))
print("distance between point 2 and point 1: \n{}".format(distance))
