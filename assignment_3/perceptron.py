'''
this is the third assignment of pattern recognition laboratory, perceptron 
algorithm
'''

# importing libraries
import numpy as np
from read_dataset import read_data
from model import train, predict

'''
for OR gate 
'''
# splitting the data
X, y = read_data("../dataset/OR.xlsx")


# training the model
'''
Class : train()
Arguments : X ---> input vector
            y ---> label
            number of epochs
            learning_rate
Output : weights of the model
'''
model = train(X, y, epoch=10, learning_rate=1)


# user input
x1 = int(input("Enter x1 value : "))
x2 = int(input("Enter x2 value : "))
X_new = np.array([x1, x2])


# prediction
'''
Class : predict()
Arguments : model ---> weight of the model
            X_new ---> verification input
'''
output = predict(model, X_new)
print("gate output : ",output)


'''
for AND gate
'''
# splitting the data
X, y = read_data("../dataset/AND.xlsx")


# training the model
'''
Class : train()
Arguments : X ---> input vector
            y ---> label
            number of epochs
            learning_rate
Output : weights of the model
'''
model = train(X, y, epoch=10, learning_rate=1)


# user input
x1 = int(input("Enter x1 value : "))
x2 = int(input("Enter x2 value : "))
X_new = np.array([x1, x2])


# prediction
'''
Class : predict()
Arguments : model ---> weight of the model
            X_new ---> verification input
'''
output = predict(model, X_new)
print("gate output : ",output)

