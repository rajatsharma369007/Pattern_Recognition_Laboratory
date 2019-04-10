'''
This script helps to calculate the euclidean distance between two arbitrary 
points. This is part of pattern recognition assignment-1c
'''

# importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import math

# reading dataset
df = pd.read_excel('/home/rajat/Documents/8th sem/Pattern_recognition/Pattern_Recognition_Laboratory/dataset/points.xlsx', header=None)

# numpy array for x and y values
x = np.array(df[0])
y = np.array(df[1])

# creating coordinates
points = np.stack([x, y]).T

# randomly choosing an arbitrary points
point1 = random.choice(points)
point2 = random.choice(points)

# calculate distance
distance = math.sqrt(((point2[0] - point1[0])**2) + ((point2[1] - point1[1])**2))
print("distance between point ({},{}) and ({},{}): \n{}".format(point1[0], point1[1], point2[0], point2[1], distance))

# plotting graph
plt.scatter(x, y)
plt.scatter([point1[0], point2[0]], [point1[1], point2[1]], c='black')
plt.plot([point1[0], point2[0]], [point1[1], point2[1]], color = 'red', label=distance)
plt.title("Euclidean distance")
plt.xlabel('X')
plt.ylabel('Y')
plt.xlim(min(x)-2, max(x)+2)
plt.ylim(min(y)-2, max(y)+2)
plt.legend()
plt.show()
