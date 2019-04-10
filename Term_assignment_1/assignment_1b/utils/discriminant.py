'''
This script help to return the value obtain from the discriminant function
defined. Here, I have used the third case: Sigma = arbitrary (mentioned in 
Duda, Hart, Stork book of pattern recognition)
'''

# importing Libraries
import numpy as np
import math

# discriminant function
def function(x, xvec1, xvec2, w1, w2):
    '''
    d1 is the discriminant function for class w1 which is defined by using xvec1
    d2 is the discriminant function for class w2 which is defined by using xvec2
    
    arguments:
        x = new point
        xvec1 = points of class1
        xvec2 = points of class2
        
        w1 = prior probability of class1
        w2 = prior probability of class2
    '''
    # mean column vector
    mean1 = np.mean(xvec1, axis=0)
    mean2 = np.mean(xvec2, axis=0)
    
    # creating the covariance matrix
    cov1 = np.cov(xvec1.T)
    cov2 = np.cov(xvec2.T)
    
    # creating the inverse of the covariance matrix
    inv_cov1 = np.linalg.inv(cov1)
    inv_cov2 = np.linalg.inv(cov2)

    # formula
    d1 = (-1/2)*np.matmul(np.matmul((x - mean1).T, inv_cov1), (x - mean1)) - (1/2)*(math.log(np.linalg.det(cov1))) + math.log(w1)
    d2 = (-1/2)*np.matmul(np.matmul((x - mean2).T, inv_cov2), (x - mean2)) - (1/2)*(math.log(np.linalg.det(cov2))) + math.log(w2)    
    
    '''
    returning the values two values of discriminant function which will help 
    classifying the points
    '''
    return d1, d2


# classification function
def classify(d1, d2):
    '''
    arguments:
    d1 is the discriminant function for class w1 which is defined by using xvec1
    d2 is the discriminant function for class w2 which is defined by using xvec2
    '''
    g = d1 - d2
    if (g > 0):
        print("discriminant value : ", g)
        print("Point x falls in class 1")
        #print("g1 : ", d1)
        #print("g2 : ", d2)
        return g
    else:
        print("discriminant value : ", g)
        print("Point x falls in class 2")
        #print("g1 : ", d1)
        #print("g2 : ", d2)
        return g
    