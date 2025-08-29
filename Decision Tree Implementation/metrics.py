import pandas as pd
import numpy as np
import math

def accuracy(y_hat, y):
    assert(y_hat.size == y.size)
    
    if (isinstance(y_hat,pd.Series)):
        y_hat = y_hat.tolist()
    if (isinstance(y,pd.Series)):
        y = y.tolist()
    
    l = len(y_hat)
    count = 0
    for i in range(l):
        if (y_hat[i] == y[i]):
            count+=1
    
    ans = (count/l)*100
    return ans

def precision(y_hat, y, cls):
    if (isinstance(y_hat,pd.Series)):
        y_hat = y_hat.tolist()
    if (isinstance(y,pd.Series)):
        y = y.tolist()
    
    chosen_class = cls
    total_samples = len(y)

    pred_class_total = y_hat.count(chosen_class)
    
    correct_pred_count = 0
    for i in range(total_samples):
        if (y_hat[i] == chosen_class):
            if (y_hat[i] == y[i]):
                correct_pred_count+=1
    if (pred_class_total == 0):
        return None
    ans = (correct_pred_count/pred_class_total)*100
    return ans

def recall(y_hat, y, cls):
    if isinstance(y_hat,pd.Series):
        y_hat = y_hat.tolist()
    if isinstance(y,pd.Series):
        y = y.tolist()

    chosen_class = cls
    total_samples = len(y)

    total_samples_chosen = y.count(chosen_class)
    recall_count = 0

    for i in range(total_samples):
        if (y[i]==chosen_class):
            if (y_hat[i]==y[i]):
                recall_count+=1

    ans = (recall_count/total_samples_chosen)*100
    return ans

def rmse(y_hat, y):
    if isinstance(y_hat,pd.Series):
        y_hat = y_hat.tolist()
    if isinstance(y,pd.Series):
        y = y.tolist()
    diff = [0]*len(y)
    for i in range (len(diff)):
        diff[i]=(y_hat[i]-y[i])**2
    ans = sum(diff)/len(y)
    ans = math.sqrt(ans)
    return ans

def mae(y_hat, y):
    if isinstance(y_hat,pd.Series):
        y_hat = y_hat.tolist()
    if isinstance(y,pd.Series):
        y = y.tolist()
    diff = [0]*len(y)
    for i in range (len(diff)):
        diff[i]=abs(y_hat[i]-y[i])
    return sum(diff)/len(y)
