'''
this script helps to demonstrate the backpropagation on iris dataset. To see the 
model implementation checkout utils folder
'''

# importing libraries
from utils import read_dataset
from utils import model

# reading data
X_train, X_test, y_train, y_test = read_dataset.read_irisdata("../dataset/Iris.csv")

# initializing the model
n_hidden = 2
epochs = 8000
learnrate = 0.01

# fitting the model
'''
function : train_irismodel
arguments : X features, y label, no of epochs, learning rate, no of hidden layer
'''
iris_model = model.train_irismodel(X_train, y_train, epochs, learnrate, n_hidden)

# testing the model
'''
function : predict_irismodel
arguments : X features, y label, iris_model
'''
predictions = model.predict_irismodel(X_test, y_test, iris_model)

