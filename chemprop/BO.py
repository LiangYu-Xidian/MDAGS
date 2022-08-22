from sklearn import gaussian_process
gp =gaussian_process.GaussianProcess(theta0=1e-2, thetaL=1e-4,thetaU=1e-1)
import pandas as pd
import numpy as np

row_path = 'train_regression.csv'
row = pd.read_csv(row_path, sep=',')
y= row.iloc[:, 1]
filename = 'regression_fingerprint.csv'
data = pd.read_csv(filename, sep=',')
X = np.array(np.array(data.iloc[:, 1:]))
print('start')
gp.fit(X,y)