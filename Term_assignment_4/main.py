'''
This script is regarding the implementation of kmeans clustering algorithm.
To see the model implementation checkout utils folder.
'''

# importing libraries
from sklearn.metrics import accuracy_score
from utils import read_dataset
from utils import model

# reading data
X_train, X_test, y_train, y_test = read_dataset.read_irisdata("../dataset/Iris.csv")

# number of K
k_value = 2
epoch = 5

'''
method : train()
arguments : number of clusters, features, labels, number of epochs
returns : k number of centroids
'''
centroids = model.train(k_value, X_train, y_train, epoch)

'''
method : predict()
arguments : number of clusters, k centroids, testing set
returns : predicted class label
'''
class_label = model.predict(k_value, centroids, X_test)

# accuracy score
accuracy = accuracy_score(y_test, class_label)
print('Test Accuracy: {}\n\n'.format(accuracy))
