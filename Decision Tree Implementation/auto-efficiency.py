import pandas as pd
import numpy as np
import math
from metrics import * 
from tree.base import DecisionTree  
from tree.utils import *
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
np.random.seed(42)

# Reading the data
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
data = pd.read_csv(url, delim_whitespace=True, header=None,
                 names=["mpg", "cylinders", "displacement", "horsepower", "weight",
                        "acceleration", "model year", "origin", "car name"])


x, y = list(), list()
for index, row in data.iterrows():
    feat_val = row[:-1].values
    features = []
    valid = True
    for val in feat_val:
        if val == '?':   # handle missing values
            valid = False
            break
        features.append(float(val))
    
    if valid:   # only use rows without missing data
        x.append(features)
        y.append(float(row["mpg"]))


def skdt(X,y,max_depth=5,criterion='squared_error'):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20) 

    skmodel = DecisionTreeRegressor(criterion=criterion, max_depth=max_depth) 
    skmodel.fit(X_train, y_train)
    y_pred = skmodel.predict(X_test)

    rmse_val = math.sqrt(mean_squared_error(y_true=y_test, y_pred=y_pred))
    mae_val = mae_val = mae(y_test, y_pred)
    print("RMSE ::",rmse_val)
    print("MAE ::",mae_val)

nrows = len(data)

def builtdt(X,y,max_depth=5,criterion="information_gain"):
    s = ((nrows//100)*80)
    mymodel = DecisionTree(criterion=criterion,max_depth=max_depth)
    mymodel.fit(pd.DataFrame(X[0:s]),pd.Series(y[0:s]))
    y_pred = mymodel.predict(pd.DataFrame(X[s:]))

    y = pd.Series(y[s:])
    print("RMSE ::",rmse(y_pred,y))
    print("MAE ::",mae(y_pred,y))

# to print sequence
print("\n")
print("A. sklearn model ==> ")
skdt(x,y)

print("\n")
print("B. My decision tree model ==> ")
builtdt(x,y)
