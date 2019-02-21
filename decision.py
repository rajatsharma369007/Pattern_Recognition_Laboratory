import pandas as pd
import numpy as np
import math

df = pd.read_excel("data.xlsx")

print(df)

# extracting inputs from file
x1w1 = np.array(df['x1w1'])
x2w1 = np.array(df['x2w1'])
x1w2 = np.array(df['x1w2'])
x2w2 = np.array(df['x2w2'])

xvec1 = np.stack((x1w1,x2w1), axis=1)
xvec2 = np.stack((x1w2,x2w2), axis=1)

# mean vector
mean1 = np.mean(xvec1, axis=0)
mean2 = np.mean(xvec2, axis=0)

# creating the covariance matrix
cov1 = np.cov(xvec1)
cov2 = np.cov(xvec2)

# creating the inverse of the covariance matrix
inv_cov1 = np.linalg.inv(cov1)
inv_cov2 = np.linalg.inv(cov2)

# prior probability
w1 = w2 = 1/2

# calculating discriminant matrix
g1 = (-1/2)*np.matmul(np.matmul((xvec1 - mean1).T, inv_cov1), (xvec1 - mean1)) - (1/2)*(math.log(np.linalg.det(cov1))) + math.log(w1)
g1 = (-1/2)*np.matmul(np.matmul((xvec2 - mean2).T, inv_cov2), (xvec2 - mean2)) - (1/2)*(math.log(np.linalg.det(cov2))) + math.log(w2)


