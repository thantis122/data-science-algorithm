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

'''






