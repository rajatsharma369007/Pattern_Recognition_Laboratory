'''
This script helps to demonstrate the performance of perceptron algorithm. I have
used truth table of AND and OR gate to verify the results. To check the 
implementation, checkout the utils folder
'''

# importing libraries
import numpy as np
from utils import read_dataset
from utils import model

'''
for OR gate 
'''
# splitting the data
X, y = read_dataset.read_data("../dataset/OR.xlsx")

# training the model
'''
Class : train()
Arguments : X ---> input vector
            y ---> label
            number of epochs
            learning_rate
Output : weights of the model
'''
print('training started')
model1 = model.train(X, y, epoch=10, learning_rate=1)
print('\nOR gate model Trained\n\n')

##############################################################################

'''
for AND gate
'''
# splitting the data
X, y = read_dataset.read_data("../dataset/AND.xlsx")


# training the model
'''
Class : train()
Arguments : X ---> input vector
            y ---> label
            number of epochs
            learning_rate
Output : weights of the model
'''
print('training started')
model2 = model.train(X, y, epoch=10, learning_rate=1)
print('\nAND gate model Trained\n\n')

##############################################################################

# user input
while(1):
    x1 = int(input("Enter x1 value : "))
    x2 = int(input("Enter x2 value : "))
    
    if x1 in [-1, 1] and x2 in [-1, 1]:
        X_new = np.array([x1, x2])
        break
    else:
        print('x1 and x2 value can only be -1 or 1')
    
# prediction on OR gate model
'''
Class : predict()
Arguments : model ---> weight of the model
            X_new ---> verification input
'''
output = model.predict(model1, X_new)
print("\nOR gate output : ",output)


# prediction on AND gate model
'''
Class : predict()
Arguments : model ---> weight of the model
            X_new ---> verification input
'''
output = model.predict(model2, X_new)
print("AND gate output : ",output)
