'''
This script is about the kmeans model implementation, modularization of train
and prediction method.
'''

# importing libraries
import random
import numpy as np
import math
import pandas as pd
from sklearn.metrics import accuracy_score

# training the model
def train(k_value, X_train, y_train, epochs):
    # initializing centroids
    centroid = []
    for i in range(k_value):
        centroid.append(random.choice(np.array(X_train)))
    centroid = np.array(centroid).reshape(k_value, X_train.shape[1])
    
    for e in range(epochs):
        class_label = []
        for row in X_train.iterrows():
            min_euclidean = math.inf
            for k in range(k_value):
                distance = math.sqrt(np.sum((centroid[k] - np.array(row[1]))**2))
                if distance < min_euclidean:
                    min_euclidean = distance
                    label = k
            class_label.append(label)
        class_label = pd.DataFrame(class_label)
        accuracy = accuracy_score(y_train, np.array(class_label))
    
        if accuracy == 0.0:
            class_label = 1-class_label
            accuracy = accuracy_score(y_train, np.array(class_label))
    
        print('epoch: {}\nAccuracy: {}\n'.format(e, accuracy))
    
        for i in range(k_value):
            class_index = class_label.index[class_label[0] == i]
            centroid_update = [0] * X_train.shape[1]
            for j in class_index:
                centroid_update += X_train.iloc[j]
            try:
                centroid[i] = centroid_update/len(class_index)
            except TypeError:
                continue
    return centroid
            
# prediction
def predict(k_value, centroid, X_test):
    class_label=[]
    for row in X_test.iterrows():
        min_euclidean = math.inf
        for k in range(k_value):
            distance = math.sqrt(np.sum((centroid[k] - np.array(row[1]))**2))
            if distance < min_euclidean:
                min_euclidean = distance
                label = k
        class_label.append(label)
    class_label = pd.DataFrame(class_label)
    return class_label
