'''
This script is for training the model and prediction
'''

# importing libraries
import random
import numpy as np


# train method
def train(X, y, epoch, learning_rate):
    weight = np.array([random.randint(-2, 2), random.randint(-2, 2), random.randint(-2, 2)])
    
    for i in range(epoch):
        print("epoch:", i)
        correct = 0
        
        for j in range(len(y)):
            D_actual = weight[0] + weight[1]*X[j][0] + weight[2]*X[j][1]
            signD = activation(D_actual)
            
            if(signD != y[j]):
                weight[0] = weight[0] + learning_rate * signD * 1
                weight[1] = weight[1] + learning_rate * signD * X[j][0]
                weight[2] = weight[2] + learning_rate * signD * X[j][1]
            else:
                correct = correct + 1    
        
        print('accuracy : ', (correct/len(y))*100)    
    return weight

# predict method
def predict(model, X):
    output = model[0] + model[1]*X[0] + model[2]*X[1]
    signD = activation(output)
    return signD
        

# activation method
def activation(x):
    if(x >= 0):
        signD = -1
    else:
        signD = 1

    return signD

