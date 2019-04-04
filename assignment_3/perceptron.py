'''
this is the third assignment of pattern recognition laboratory, perceptron 
algorithm
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
model1 = model.train(X, y, epoch=10, learning_rate=1)

##############################################################################

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
output = model.predict(model1, X_new)
print("OR gate output : ",output)

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
model2 = model.train(X, y, epoch=10, learning_rate=1)

##############################################################################

# prediction
'''
Class : predict()
Arguments : model ---> weight of the model
            X_new ---> verification input
'''
output = model.predict(model2, X_new)
print("AND gate output : ",output)

