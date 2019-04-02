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
    
