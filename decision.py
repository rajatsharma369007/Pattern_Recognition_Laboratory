# importing class
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt 

# reading the excel file
df = pd.read_excel("data.xlsx")
print(df)

# extracting inputs from file
x1w1 = np.array(df['x1w1'])
x2w1 = np.array(df['x2w1'])
x1w2 = np.array(df['x1w2'])
x2w2 = np.array(df['x2w2'])

# creating the column vector
xvec1 = np.stack((x1w1,x2w1), axis=1)
xvec2 = np.stack((x1w2,x2w2), axis=1)

# mean column vector
mean1 = np.mean(xvec1, axis=0)
mean2 = np.mean(xvec2, axis=0)

# creating the covariance matrix
cov1 = np.cov(xvec1.T)
cov2 = np.cov(xvec2.T)

# creating the inverse of the covariance matrix
inv_cov1 = np.linalg.inv(cov1)
inv_cov2 = np.linalg.inv(cov2)

# prior probability
w1 = w2 = 1/2

# discriminant function
def function(x):
    d1 = (-1/2)*np.matmul(np.matmul((x - mean1).T, inv_cov1), (x - mean1)) - (1/2)*(math.log(np.linalg.det(cov1))) + math.log(w1)
    d2 = (-1/2)*np.matmul(np.matmul((x - mean2).T, inv_cov2), (x - mean2)) - (1/2)*(math.log(np.linalg.det(cov2))) + math.log(w2)    
    return d1, d2

x = np.array([3, 4])
g1, g2 = function(x)

def investigate_class(g1, g2):
    if (g1 > g2):
        g = g1 - g2
        print("Point x falls in class 1")
        print("g  : ", g)
        print("g1 : ", g1)
        print("g2 : ", g2)
    else:
        g = g1 - g2
        print("Point x falls in class 2")
        print("g  : ", g)
        print("g1 : ", g1)
        print("g2 : ", g2)

investigate_class(g1, g2)

    
