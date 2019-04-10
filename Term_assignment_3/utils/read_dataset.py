''' 
This script helps in reading the dataset
''' 
# importing libraries
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
    

def read_irisdata(pathname):
    # reading csv data
    df = pd.read_csv(pathname)
    df = df.drop(['Id'], axis=1)
    
    # data analysis
    g = sns.PairGrid(df, hue="Species")
    g = g.map_diag(plt.hist)
    g = g.map_offdiag(plt.scatter)
    g = g.add_legend() 

    # Make dummy variables for rank
    df = pd.concat([df, pd.get_dummies(df['Species'], drop_first=True)], axis=1)
    df = df.drop(['Species'], axis=1)
    
    # normalizing the values
    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(df)
    normalized_df = pd.DataFrame(normalized_data)
    
    # Splitting 10% of the data for testing
    np.random.seed(21)
    sample = np.random.choice(normalized_df.index, size=int(len(normalized_df) \
                                                            *0.9), replace=False)
    
    # splitting rows
    train_data, test_data = normalized_df.ix[sample], normalized_df.drop(sample)
    
    # splitting features and targets
    X_train, y_train = train_data.drop(4, axis=1), train_data[4]
    X_test, y_test = test_data.drop(4, axis=1), test_data[4]
    
    return X_train, X_test, y_train, y_test
    
