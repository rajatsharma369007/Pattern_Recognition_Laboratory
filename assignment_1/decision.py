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

x = np.array([0, 3.6])
g1, g2 = function(x)

def investigate_class(g1, g2):
    if (g1 > g2):
        g = g1 - g2
        print("Point x falls in class 1")
        print("g  : ", g)
        print("g1 : ", g1)
        print("g2 : ", g2)
        return g
    else:
        g = g1 - g2
        print("Point x falls in class 2")
        print("g  : ", g)
        print("g1 : ", g1)
        print("g2 : ", g2)
        return g
        
    
# g < 0 == class 2 || g > 0 == class 1
investigate_class(g1, g2)

max_x = max(np.append(xvec1[:, 0], xvec2[:, 0]))
min_x = min(np.append(xvec1[:, 0], xvec2[:, 0]))
max_y = max(np.append(xvec1[:, 1], xvec2[:, 1]))
min_y = min(np.append(xvec1[:, 1], xvec2[:, 1]))

plot_class1 = []
plot_class2 = []

for i in range((min_x-2) * 10 , ((max_x+2)+1) * 10):
    i = i * 0.1
    for j in range((min_y-2) * 10 , ((max_y+2)+1) * 10):
        j = j * 0.1
        x = np.array([i, j])
        g1, g2 = function(x)
        if g1 - g2 > 0:
            plot_class1.append(x)
        else:
            plot_class2.append(x)

plot_class1 = np.array(plot_class1)
plot_class2 = np.array(plot_class2)

plt.scatter(plot_class1[:, 0], plot_class1[:, 1], c="yellow")
plt.scatter(plot_class2[:, 0], plot_class2[:, 1], c="orange")
plt.scatter(xvec1[:, 0], xvec1[:, 1], c="green")
plt.scatter(xvec2[:, 0], xvec2[:, 1], c="red")
plt.hlines(y=0, xmin=-1, xmax=8, linestyles='dashed')
plt.vlines(x=0, ymin=-6, ymax=11, linestyles='dashed');




