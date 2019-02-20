import pandas as pd
import numpy as np

df = pd.read_excel("data.xlsx")

print(df)

# input vector
x1 = np.array(df['w1'])
x2 = np.array(df['w2'])
x1 = x1.reshape(2,4)
x2 = x2.reshape(2,4)

print("X1 = ", x1)
print("X2 = ", x2)

# mean vector
mean1 = np.mean(x1, axis=1)
mean2 = np.mean(x2, axis=1)

print("mean 1 = ",mean1)
print("mean 2 = ",mean2)

cov1 = np.cov(x1)
cov2 = np.cov(x2)

print("covariance 1 = ",cov1)
print("covariance 2 = ",cov2)





