from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tree.utils import *

np.random.seed(42)

class Node:
    def __init__(self):
        self.feature_label = None
        self.pred_value = None
        self.children = dict()
        self.left = None
        self.right = None
        self.split_value = None

# @dataclass
class DecisionTree:
    criterion: Literal["information_gain", "gini_index"] 
    max_depth: int

    def __init__(self, criterion, max_depth=5):
        self.criterion = criterion
        self.max_depth = max_depth
        self.in_type = None 
        self.out_type = None

    def fit(self, X, y):
        self.in_type = X.dtypes[0].name
        self.out_type = y.dtype.name

        # case: DIDO
        if ("category" in X.dtypes.values and self.out_type == "category"):
            features = list(X.columns)
            temp_X = X.copy()
            temp_X["out"] = y
            self.tree = self.DI(temp_X, None, features, 0, mode="classification")

        # case: DIRO
        elif ("category" in X.dtypes.values and self.out_type != "category"):
            features = list(X.columns)
            temp_X = X.copy()
            temp_X["out"] = y
            self.tree = self.DI(temp_X, None, features, 0)

        # case: RIDO
        elif ("category" not in X.dtypes.values and self.out_type == "category"):
            self.X, self.y = X, y
            self.no_of_attributes = X.shape[1]
            self.no_of_out_classes = y.nunique()
            self.tree = self.RI(X, y, mode="classification")

        # case: RIRO
        else:  # ("category" not in X.dtypes.values and self.out_type != "category")
            self.no_of_attributes = X.shape[1]
            self.tree = self.RI(X, y, 0)

    def _split_riro(self, X, y):
        m = list(np.unique(y))
        if(len(m) <= 1):
            return m[0], None

        start_loss = 10**8
        best_feature, best_split_threshold = None, None

        for feature in list(X.columns):

            a = X[feature]
            a = pd.DataFrame(a)
            a['out'] = y
            a = a.sort_values(by=feature, ascending=True)
            a = a.reset_index()
            a = a.drop(['index'], axis=1)

            classes = a['out']
            a = a.drop(['out'],axis=1)
            cutoff_values = a

            for i in range(1, len(classes)):
                c = classes[i-1]

                curr_loss = loss(classes, i-1)

                if (curr_loss < start_loss):
                    start_loss = curr_loss
                    best_feature = feature
                    best_split_threshold = round(((cutoff_values.loc[i-1,feature] + cutoff_values.loc[i,feature])/2), 6)
        return best_feature, best_split_threshold

    def _best_split(self, X, y):

        m = list(np.unique(y))
        if(len(m) == 1):
            return m[0], None

        # initialization 
        best_feature, best_split_threshold = None, None
        if (self.criterion == "information_gain"):
            start_gain = -10**8
        else:
            start_gain = 10**8
        ##

        for feature in list(X.columns):
            # sorting output based on feature
            a = X[feature]
            a = pd.DataFrame(a)
            a['out'] = y
            a = a.sort_values(by=feature, ascending=True)
            a = a.reset_index()
            a = a.drop(['index'], axis=1)
            classes = a['out']
            ##

            temp_c = np.unique(classes, return_counts=True)
            temp_classes = list(temp_c[0])
            classes_count = list(temp_c[1])

            self.no_of_out_classes = len(temp_classes)

            a = a.drop(['out'],axis=1)
            cutoff_values = a

            # for tracking unique labels before and after split
            labels_before = dict()
            for i in range(self.no_of_out_classes):
                labels_before[temp_classes[i]] = 0

            labels_after = dict()
            for elem in range(self.no_of_out_classes):
                labels_after[temp_classes[elem]] = classes_count[elem]  
            ##

            for i in range(1, len(classes)):
                ## tracking labels on either side of split
                c = classes[i-1]
                labels_before[c]+=1
                labels_after[c]-=1

                if (self.criterion == "information_gain"):
                    gain_left = entropy(pd.Series(list(labels_before.values())))
                    gain_right = entropy(pd.Series(list(labels_after.values())))
                    gain_temp = entropy(y) - (i * gain_left + (len(classes) - i) * gain_right) / len(classes)
                    
                    if (start_gain < gain_temp):
                        start_gain = gain_temp
                        best_feature = feature
                        best_split_threshold = round(((cutoff_values.loc[i-1,feature] + cutoff_values.loc[i,feature])/2), 6)
                else:
                    gini_left = gini_index(pd.Series(list(labels_before.values())))
                    gini_right = gini_index(pd.Series(list(labels_after.values())))
                    gini_index_temp = (i * gini_left + (len(classes) - i) * gini_right) / len(classes)

                    if (gini_index_temp <= start_gain):
                        start_gain = gini_index_temp
                        best_feature = feature
                        best_split_threshold = round(((cutoff_values.loc[i-1,feature] + cutoff_values.loc[i,feature])/2), 6)
        return best_feature, best_split_threshold

    def RI(self, samples, output_vec, depth=0, parent_node=None, mode="regression"):

        if depth < self.max_depth:
            # different split functions
            if mode == "regression":
                feature, split_value = self._split_riro(samples, output_vec)
            else:
                feature, split_value = self._best_split(samples, output_vec)

            if (feature is not None and split_value is not None):
                samples['out'] = output_vec
                samples = samples.sort_values(by=feature, ascending=True).reset_index(drop=True)
                output_vec = samples['out']
                samples = samples.drop(['out'], axis=1)

                X_l, y_l, X_r, y_r = [], [], [], []
                for index in range(len(samples)):
                    if (samples.loc[index, feature] <= split_value):
                        X_l.append(samples.loc[index]); y_l.append(output_vec[index])
                    else:
                        X_r.append(samples.loc[index]); y_r.append(output_vec[index])

                X_l, X_r = pd.DataFrame(X_l).reset_index(drop=True), pd.DataFrame(X_r).reset_index(drop=True)
                y_l, y_r = pd.Series(y_l).reset_index(drop=True), pd.Series(y_r).reset_index(drop=True)

                node = Node()
                node.feature_label = feature
                node.split_value = split_value

                # difference in pred_value
                if mode == "regression":
                    node.pred_value = round(float(output_vec.mean()), 6)
                else:
                    uniq, counts = np.unique(output_vec, return_counts=True)
                    node.pred_value = uniq[np.argmax(counts)]

                if (len(X_l) != 0 and len(y_l) != 0):
                    node.left = self.RI(X_l, y_l, depth+1, node, mode)
                if (len(X_r) != 0 and len(y_r) != 0):
                    node.right = self.RI(X_r, y_r, depth+1, node, mode)

            elif (feature is not None and split_value is None):
                node = Node()
                if mode == "regression":
                    node.pred_value = round(float(output_vec.mean()), 6)
                else:
                    node.pred_value = np.unique(output_vec)[0]
                return node

            return node

        else:
            node = Node()
            if mode == "regression":
                node.pred_value = round(float(output_vec.mean()), 6)
            else:
                if len(output_vec) == 0:
                    node.pred_value = None
                else:
                    uniq, counts = np.unique(output_vec, return_counts=True)
                    node.pred_value = uniq[np.argmax(counts)]
            return node
        
    def DI(self, samples, target_attr, attributes, depth, mode="regression"):

        output_vec = samples['out']

        if (depth < self.max_depth):

            if (len(output_vec.unique()) <= 1):
                temp = output_vec.unique()[0]
                return temp

            elif (len(attributes) == 0):
                if (mode == "regression"):
                    temp = sum(output_vec)/len(output_vec)
                else:  # classification
                    temp = np.unique(output_vec)[np.argmax(np.unique(output_vec, return_counts=True)[1])]
                return temp

            else:
                a = list()

                if (mode == "regression"):
                    for x in attributes:
                        attr = samples[x]
                        var_gain = variance_gain(output_vec, attr)
                        a.append(var_gain)
                    best_attr = attributes[a.index(max(a))]

                elif (mode == "classification"):
                    if (self.criterion == "information_gain"):
                        for x in attributes:
                            attr = samples[x]
                            inf_gain = information_gain(output_vec, attr)
                            a.append(inf_gain)
                        best_attr = attributes[a.index(max(a))]

                    elif (self.criterion == "gini_index"):
                        for x in attributes:
                            attr = samples[x]
                            g_gain = gini_gain(output_vec, attr)
                            a.append(g_gain)
                        best_attr = attributes[a.index(max(a))]

                root = Node()
                root.feature_label = best_attr

                for x in samples[best_attr].unique():
                    new_data = samples[samples[best_attr] == x]
                    new_data = new_data.reset_index(drop = True)

                    if (len(new_data) == 0):
                        if (mode == "regression"):
                            root.children[x] = sum(output_vec)/len(output_vec)
                        else:
                            root.children[x] = np.unique(output_vec)[np.argmax(np.unique(output_vec,return_counts=True)[1])]
                    else:
                        temp_attr = []
                        for y in attributes:
                            if (y!=best_attr):
                                temp_attr.append(y)

                        subtree = self.DI(new_data, best_attr, temp_attr, depth+1, mode)

                        root.children[x] = subtree

                return root

        else:
            if (mode == "regression"):
                temp = sum(output_vec)/len(output_vec)
            else:
                temp = np.unique(output_vec)[np.argmax(np.unique(output_vec,return_counts=True)[1])]
            return temp 
        

    def predict(self, X):

            # case: DI
            if (self.in_type == "category"):
                y_hat = list()

                if (self.out_type == "category"):
                    if (type(self.tree) != Node):
                        y_hat.append(self.tree)
                        return pd.Series(y_hat)

                for i in range(len(X)):
                    tree = self.tree
                    data = list(X.loc[i])
                    # while True:
                    #     curr_feat = tree.feature_label
                    #     curr_val = data[curr_feat]

                    #     # fix unseen category
                    #     if curr_val not in tree.children:
                    #         y_hat.append(tree.pred_value)
                    #         break
                    #     ##
                    #     if (type(tree.children[curr_val]) == Node):
                    #         tree = tree.children[curr_val]
                    #     else:
                    #         y_hat.append(tree.children[curr_val])
                    #         break
                    while True:
                        # If tree is not a Node (leaf), just return it
                        if not isinstance(tree, Node):
                            y_hat.append(tree)
                            break

                        curr_feat = tree.feature_label
                        curr_val = data[curr_feat]

                        if curr_val not in tree.children:
                            y_hat.append(tree.pred_value)
                            break

                        tree = tree.children[curr_val]


                y_hat = pd.Series(y_hat)
                return y_hat
            
            # case: RI
            elif (self.in_type != "category"):
                y_hat = list()

                for i in range(len(X)):
                    tree = self.tree
                    while True:
                        curr_node_feature = tree.feature_label
                        if (curr_node_feature==None):
                            break

                        # fix: handle nodes with no split (leaf or early stop) ---
                        if (tree.split_value is None):
                            break
                        ##

                        sample_val = X.loc[i, curr_node_feature]

                        if (sample_val <= tree.split_value):
                            if (tree.left != None):
                                tree = tree.left
                            else:
                                break
                        else:
                            if (tree.right != None):
                                tree = tree.right
                            else:
                                break

                    y_hat.append(tree.pred_value)

                y_hat = pd.Series(y_hat)
                return y_hat

    def plot(self):
        if self.in_type == "category":
            tree = self.tree

            def printdict(d, indent=0):
                print("\t"*(indent-1) + "feature:" +str(d.feature_label))
                for key, value in d.children.items():
                    print('\t' * indent + "\t" + "feat_value:" +  str(key))
                    if isinstance(value, Node):
                        printdict(value, indent+1)
                    else:
                        print('\t' * (indent+1) + str(value))

            printdict(tree)

        elif (self.in_type != "category"):
            tree = self.tree
            def printdict(d, indent=0):

                if (isinstance(d.left, Node)):
                    print("\t"*(indent) + "feature:" +str(d.feature_label) + "\t" + "split value:" + str(d.split_value))
                    printdict(d.left, indent+1)

                if (isinstance(d.right, Node)):
                    print("\t"*(indent) + "feature:" +str(d.feature_label) + "\t" + "split value:" + str(d.split_value))
                    printdict(d.right, indent+1)

                if (isinstance(d.right, Node) == False and isinstance(d.left, Node) == False):
                    print('\t' * (indent+1) + str(d.pred_value))

            printdict(tree)
