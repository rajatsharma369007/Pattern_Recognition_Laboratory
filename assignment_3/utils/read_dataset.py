''' 
This script helps in reading the dataset
''' 
# importing libraries
import numpy as np
import pandas as pd
    
# function for splitting data
def read_data(pathname):
    # reading dataset
    df = pd.read_excel(pathname)
    
    # X vector and d label
    X = np.array(df[['x1','x2']])
    D_desired = np.array(df[['D']])
    
    return X, D_desired
    

def read_iris(pathname):
    # reading dataset
    df = pd.read_csv(pathname)
    # for 2 category
    X_df = df[0:100][['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
    X = np.array(X_df)
    y_df = df[0:100]['Species']
    y = np.array(pd.get_dummies(y_df, drop_first=True))
    
    return X, y
    
