''' 
This script helps in reading the dataset
''' 
# importing libraries
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
       

def read_irisdata(pathname):
    # reading csv data
    df = pd.read_csv(pathname)

    # Make dummy variables for rank
    df = pd.concat([df, pd.get_dummies(df['Species'], drop_first=True)], axis=1)
    df = df.drop(['Id','Species'], axis=1)
    
    # dataset analysis
    sns.pairplot(df[['SepalLengthCm','SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']])
    
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
    
