#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# --------------------------------------
# DATE: 2018/3/20
# --------------------------------------

import numpy as np
import pandas as pd
from sklearn import tree
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris


iris = load_iris()
clf = tree.DecisionTreeClassifier()
clf = clf.fit(iris.data, iris.target)









