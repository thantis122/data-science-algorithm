#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# --------------------------------------
# DATE: 2018/3/20
# --------------------------------------

import numpy as np
import pandas as pd
from sklearn import tree
import matplotlib.pyplot as plt
# import graphviz
from sklearn.datasets import load_iris


'''
the simple dataframe iris
'''
iris = load_iris()
iris_data = pd.DataFrame(data = iris.data, index = [ i+1 for i in range(len(iris.data))], columns = iris.feature_names)

iris_data_train = iris_data.sample(frac = 0.8)
iris_data_test = iris_data.sample(frac = 0.2)

clf = tree.DecisionTreeClassifier()
clf = clf.fit(iris.data, iris.target)
clf_pre = clf.predict(iris_data_test)


'''
dot_data = tree.export_graphviz(clf, out_file=None)
graph = graphviz.Source(dot_data)
graph.render("iris")
graph
'''


'''
another complex dataframe
Titanic

'''

#import the train dataset
data_titanic_train = pd.read_csv('train.csv', sep=',', encoding = 'utf-8')


def feature_treat(df_titanic):
    df = df_titanic
    df.set_index('PassengerId')
    
    
    
    return df

dfvv = feature_treat()







#import the test dataset
data_titanic_test = pd.read_csv('test.csv', sep=',', encoding = 'utf-8')




def fib3(n):
    a, b = 0, 1
    for i in range(n):
        a, b = b, a+b
    return a
 
for i in range(10):
    print(fib3(i))
    




