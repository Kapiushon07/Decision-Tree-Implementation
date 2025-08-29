import math 
import pandas as pd
import numpy as np

def entropy(Y):
    temp = np.unique(Y, return_counts=True)
    uniq_Y = list(temp[0])
    Y_count = list(temp[1])
    total = sum(Y_count)
    ent = 0
    for elem in uniq_Y:
        prob = Y_count[uniq_Y.index(elem)] / total
        ent -= (prob * (math.log2(prob)))
    return ent

def information_gain(Y, attr):
    initial_gain = entropy(Y)
    temp_Y = Y.tolist()
    temp_attr = attr.tolist()
    temp_attr = list(np.unique(attr))

    for a in temp_attr:
        l = []
        count = 0
        for j in attr:
            if (j == a):
                l.append(temp_Y[count])
            count+=1
        initial_gain -= ((len(l) / len(temp_Y)) * entropy(pd.Series(l)))
    return initial_gain

def variance(Y):
    if (isinstance(Y, pd.Series)):
        Y = Y.tolist()
    
    total_samples = len(Y)
    Y_squares = [i*i for i in Y]

    sq_mean = sum(Y_squares)/total_samples
    mean_sq = (sum(Y)/total_samples)**2
    return(sq_mean-mean_sq)

def variance_gain(Y, attr):
    initial_gain = variance(Y)
    Y = Y.tolist()
    attr = attr.tolist()
    attr_set = set(attr)
    attr_set = list(attr_set)
    
    for i in attr_set:
        l = []
        for j in range (len(attr)):
            if attr[j] == i:
                l.append(Y[j])
        initial_gain = initial_gain-(len(l)/len(Y))*variance(l)
    return initial_gain

def gini_index(Y):
    if (isinstance(Y, list) == False):
        temp_Y = Y.tolist()
    else:
        temp_Y = Y
        
    total_samples = len(temp_Y)
    temp = np.unique(Y, return_counts=True)
    Y_count = list(temp[1])
    Y_unique = list(temp[0])

    ans = 1
    for attr in Y_unique:
        g = Y_count[Y_unique.index(attr)] / total_samples
        ans -= (g**2)
    return ans

def gini_gain(Y, attr):
    Y = Y.tolist()
    attr = attr.tolist()
    attr_set = set(attr)
    attr_set = list(attr_set)
    initial_gain=0
    for i in attr_set:
        l = []
        for j in range (len(attr)):
            if attr[j] == i:
                l.append(Y[j])
        initial_gain = initial_gain+(len(l)/len(Y))*gini_index(l)
    return initial_gain

def loss(Y, split_index):
    if (isinstance(Y, list) == False):
        y = Y.tolist()
    
    total_samples = len(y)
    c1 = 0
    c2 = 0
    for i in range(total_samples):
        if (i <= split_index):
            c1 += y[i]
        else:
            c2 += y[i]
    c1 /= total_samples
    c2 /= total_samples

    loss = 0
    for i in range(total_samples):
        loss += ((y[i] - c1)**2 + (y[i]-c2)**2)
    return loss
        
