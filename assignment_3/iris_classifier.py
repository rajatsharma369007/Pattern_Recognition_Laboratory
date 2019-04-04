'''
this is the fourth assignment of pattern recognition laboratory, implementing 
backpropagation
'''

import numpy as np
from utils import read_dataset

# reading data
X_train, X_test, y_train, y_test = read_dataset.read_irisdata("../dataset/Iris.csv")


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


n_hidden = 2
epochs = 20000
learnrate = 0.01

n_records, n_features = X_train.shape
last_loss = None

weights_input_hidden = np.random.normal(scale = 1 / n_features ** 0.5, size = (n_features, n_hidden))
weights_hidden_output = np.random.normal(scale = 1 / n_features ** 0.5, size = n_hidden)

for e in range(epochs):
    # initializing all the delta weights with zeros
    del_w_input_hidden = np.zeros(weights_input_hidden.shape)
    del_w_hidden_output = np.zeros(weights_hidden_output.shape)
    # In each epoch, and in each value of column
    for x, y in zip(X_train.values, y_train):
        # forward pass
        hidden_input = np.dot(x, weights_input_hidden)
        hidden_output = sigmoid(hidden_input)
        output = sigmoid(np.dot(hidden_output, weights_hidden_output))
        
        # backward pass
        error = y - output
        # calculate the error term for the output unit,
        output_error_term = error * output * (1 - output)
        
        # propagate the errors to hidden layers
        # hidden layer's contribution to the error
        hidden_error = np.dot(output_error_term, weights_hidden_output)
        # error term for the hidden layer
        hidden_error_term = hidden_error * hidden_output * (1 - hidden_output)
        
        # update the change in weights
        del_w_hidden_output += output_error_term * hidden_output
        del_w_input_hidden += hidden_error_term * x[:, None]
        
    weights_input_hidden += learnrate * del_w_input_hidden / n_records
    weights_hidden_output += learnrate * del_w_hidden_output / n_records 
    
        # Printing out the mean square error on the training set
    if e % (epochs / 10) == 0:
        hidden_output = sigmoid(np.dot(x, weights_input_hidden))
        out = sigmoid(np.dot(hidden_output, weights_hidden_output))
        loss = np.mean((out - y_train) ** 2)

        if last_loss and last_loss < loss:
            print("Train loss: ", loss)
        else:
            print("Train loss: ", loss)
        last_loss = loss
        

# Calculate accuracy on test data
hidden = sigmoid(np.dot(X_test, weights_input_hidden))
out = sigmoid(np.dot(hidden, weights_hidden_output))
predictions = out > 0.5
accuracy = np.mean(predictions == y_test)
print("Prediction accuracy: {:.3f}".format(accuracy))





