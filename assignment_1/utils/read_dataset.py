'''
 This script helps to provide the x1, x2 vectors from the excel dataset
'''

# Importing Libraries
import pandas as pd
import numpy as np


def create_vector(path_name):
    # reading the excel file
    df = pd.read_excel(path_name)
    print("Printing dataframe...")
    print(df)
    
    '''
    dataset has columns as
    x1w1 ---> x1 for class w1 
    x2w1 ---> x2 for class w1
    
    x1w2 ---> x1 for class w2
    x2w2 ---> x2 for class w2
    '''
    
    # extracting inputs from file
    x1w1 = np.array(df['x1w1'])
    x2w1 = np.array(df['x2w1'])
    x1w2 = np.array(df['x1w2'])
    x2w2 = np.array(df['x2w2'])
    
    # creating the column vector
    '''
    numpy's stack method helps to combine the array's elements either rowwise
    or columnwise. axis=1 states rowwise stacking of elements.
    '''
    xvec1 = np.stack((x1w1,x2w1), axis=1)
    xvec2 = np.stack((x1w2,x2w2), axis=1)
    
    return xvec1, xvec2