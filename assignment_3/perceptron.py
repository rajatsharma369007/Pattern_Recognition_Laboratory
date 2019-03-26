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
model = train(X, y, 10, 1)

# user input
x1 = int(input("Enter x1 value : "))
x2 = int(input("Enter x2 value : "))
X_new = np.array([x1, x2])

# prediction
output = predict(model, X_new)

print("gate output : ",output)

'''
for AND gate
'''
# splitting the data
X, y = read_data("../dataset/AND.xlsx")

# training the model
model = train(X, y, 10, 1)

# user input
x1 = int(input("Enter x1 value : "))
x2 = int(input("Enter x2 value : "))
X_new = np.array([x1, x2])

# prediction
output = predict(model, X_new)

print("gate output : ",output)

