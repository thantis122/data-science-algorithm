#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# --------------------------------------
# DATE: 2018/3/15
# DEV:  Bin LIU
# DESC:
# --------------------------------------

'''

常用的几种决策树算法有ID3、C4.5、CART：

ID3：选择信息熵增益最大的feature作为node，实现对数据的归纳分类。
C4.5：是ID3的一个改进，比ID3准确率高且快，可以处理连续值和有缺失值的feature。
CART：使用基尼指数的划分准则，通过在每个步骤最大限度降低不纯洁度，CART能够处理孤立点以及能够对空缺值进行处理。

'''

# -*- coding: utf-8 -*-

from numpy import *


# 计算数据集的香农熵
def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    # 给所有可能分类创建字典
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    # 以2为底数计算香农熵
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt


# 对离散变量划分数据集，取出该特征取值为value的所有样本
def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis + 1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet


# 对连续变量划分数据集，direction规定划分的方向，
# 决定是划分出小于value的数据样本还是大于value的数据样本集
def splitContinuousDataSet(dataSet, axis, value, direction):
    retDataSet = []
    for featVec in dataSet:
        if direction == 0:
            if featVec[axis] > value:
                reducedFeatVec = featVec[:axis]
                reducedFeatVec.extend(featVec[axis + 1:])
                retDataSet.append(reducedFeatVec)
        else:
            if featVec[axis] <= value:
                reducedFeatVec = featVec[:axis]
                reducedFeatVec.extend(featVec[axis + 1:])
                retDataSet.append(reducedFeatVec)
    return retDataSet


# 选择最好的数据集划分方式
def chooseBestFeatureToSplit(dataSet, labels):
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1
    bestSplitDict = {}
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]
        # 对连续型特征进行处理
        if type(featList[0]).__name__ == 'float' or type(featList[0]).__name__ == 'int':
            # 产生n-1个候选划分点
            sortfeatList = sorted(featList)
            splitList = []
            for j in range(len(sortfeatList) - 1):
                splitList.append((sortfeatList[j] + sortfeatList[j + 1]) / 2.0)

            bestSplitEntropy = 10000
            slen = len(splitList)
            # 求用第j个候选划分点划分时，得到的信息熵，并记录最佳划分点
            for j in range(slen):
                value = splitList[j]
                newEntropy = 0.0
                subDataSet0 = splitContinuousDataSet(dataSet, i, value, 0)
                subDataSet1 = splitContinuousDataSet(dataSet, i, value, 1)
                prob0 = len(subDataSet0) / float(len(dataSet))
                newEntropy += prob0 * calcShannonEnt(subDataSet0)
                prob1 = len(subDataSet1) / float(len(dataSet))
                newEntropy += prob1 * calcShannonEnt(subDataSet1)
                if newEntropy < bestSplitEntropy:
                    bestSplitEntropy = newEntropy
                    bestSplit = j
            # 用字典记录当前特征的最佳划分点
            bestSplitDict[labels[i]] = splitList[bestSplit]
            infoGain = baseEntropy - bestSplitEntropy
        # 对离散型特征进行处理
        else:
            uniqueVals = set(featList)
            newEntropy = 0.0
            # 计算该特征下每种划分的信息熵
            for value in uniqueVals:
                subDataSet = splitDataSet(dataSet, i, value)
                prob = len(subDataSet) / float(len(dataSet))
                newEntropy += prob * calcShannonEnt(subDataSet)
            infoGain = baseEntropy - newEntropy
        if infoGain > bestInfoGain:
            bestInfoGain = infoGain
            bestFeature = i
    # 若当前节点的最佳划分特征为连续特征，则将其以之前记录的划分点为界进行二值化处理
    # 即是否小于等于bestSplitValue
    if type(dataSet[0][bestFeature]).__name__ == 'float' or type(dataSet[0][bestFeature]).__name__ == 'int':
        bestSplitValue = bestSplitDict[labels[bestFeature]]
        labels[bestFeature] = labels[bestFeature] + '<=' + str(bestSplitValue)
        for i in range(shape(dataSet)[0]):
            if dataSet[i][bestFeature] <= bestSplitValue:
                dataSet[i][bestFeature] = 1
            else:
                dataSet[i][bestFeature] = 0
    return bestFeature


# 特征若已经划分完，节点下的样本还没有统一取值，则需要进行投票
def majorityCnt(classList):
    classCount = {}

    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    return max(classCount)


# 主程序，递归产生决策树
def createTree(dataSet, labels, data_full, labels_full):
        classList = [example[-1] for example in dataSet]

    if classList.count(classList[0]) == len(classList):
            return classList[0]
    if len(dataSet[0]) == 1:
            return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet, labels)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel: {}}
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    if type(dataSet[0][bestFeat]).__name__ == 'str':
            currentlabel = labels_full.index(labels[bestFeat])
        featValuesFull = [example[currentlabel] for example in data_full]
        uniqueValsFull = set(featValuesFull)
    del (labels[bestFeat])
     # 针对bestFeat的每个取值，划分出一个子树。
    for value in uniqueVals:
            subLabels = labels[:]
        if type(dataSet[0][bestFeat]).__name__ == 'str':
                uniqueValsFull.remove(value)
        myTree[bestFeatLabel][value] = createTree(splitDataSet \
             (dataSet, bestFeat, value), subLabels, data_full, labels_full)
    if type(dataSet[0][bestFeat]).__name__ == 'str':
            for value in uniqueValsFull:
                    myTree[bestFeatLabel][value] = majorityCnt(classList)
    return myTree

df = pd.read_csv('watermelon_4_3.csv')
data = df.values[:, 1:].tolist()
data_full = data[:]
labels = df.columns.values[1:-1].tolist()
labels_full = labels[:]
myTree = createTree(data, labels, data_full, labels_full)

import matplotlib.pyplot as plt

decisionNode = dict(boxstyle="sawtooth", fc="0.8")
leafNode = dict(boxstyle="round4", fc="0.8")
arrow_args = dict(arrowstyle="<-")


# 计算树的叶子节点数量
def getNumLeafs(myTree):
    numLeafs = 0
    firstStr = myTree.keys()[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            numLeafs += getNumLeafs(secondDict[key])
        else:
            numLeafs += 1
    return numLeafs


# 计算树的最大深度
def getTreeDepth(myTree):
    maxDepth = 0
    firstStr = myTree.keys()[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:
            thisDepth = 1
        if thisDepth > maxDepth:
            maxDepth = thisDepth
    return maxDepth


# 画节点
def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    createPlot.ax1.annotate(nodeTxt, xy=parentPt, xycoords='axes fraction', \
                            xytext=centerPt, textcoords='axes fraction', va="center", ha="center", \
                            bbox=nodeType, arrowprops=arrow_args)


# 画箭头上的文字
def plotMidText(cntrPt, parentPt, txtString):
    lens = len(txtString)
    xMid = (parentPt[0] + cntrPt[0]) / 2.0 - lens * 0.002
    yMid = (parentPt[1] + cntrPt[1]) / 2.0
    createPlot.ax1.text(xMid, yMid, txtString)


def plotTree(myTree, parentPt, nodeTxt):
    numLeafs = getNumLeafs(myTree)
    depth = getTreeDepth(myTree)
    firstStr = myTree.keys()[0]
    cntrPt = (plotTree.x0ff + (1.0 + float(numLeafs)) / 2.0 / plotTree.totalW, plotTree.y0ff)
    plotMidText(cntrPt, parentPt, nodeTxt)
    plotNode(firstStr, cntrPt, parentPt, decisionNode)
    secondDict = myTree[firstStr]
    plotTree.y0ff = plotTree.y0ff - 1.0 / plotTree.totalD
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            plotTree(secondDict[key], cntrPt, str(key))
        else:
            plotTree.x0ff = plotTree.x0ff + 1.0 / plotTree.totalW
            plotNode(secondDict[key], (plotTree.x0ff, plotTree.y0ff), cntrPt, leafNode)
            plotMidText((plotTree.x0ff, plotTree.y0ff), cntrPt, str(key))
    plotTree.y0ff = plotTree.y0ff + 1.0 / plotTree.totalD


def createPlot(inTree):
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)
    plotTree.totalW = float(getNumLeafs(inTree))
    plotTree.totalD = float(getTreeDepth(inTree))
    plotTree.x0ff = -0.5 / plotTree.totalW
    plotTree.y0ff = 1.0
    plotTree(inTree, (0.5, 1.0), '')
    plt.show()


createPlot(myTree)

'''
autre modele de ID3 par Python

'''
# -*- coding: utf-8 -*-
from math import log
import operator
import pickle

'''
输入：原始数据集、子数据集（最后一列为类别标签，其他为特征列）
功能：计算原始数据集、子数据集（某一特征取值下对应的数据集）的香农熵
输出：float型数值（数据集的熵值）
'''


def calcShannonEnt(dataset):
    numSamples = len(dataset)
    labelCounts = {}
    for allFeatureVector in dataset:
        currentLabel = allFeatureVector[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    entropy = 0.0
    for key in labelCounts:
        property = float(labelCounts[key]) / numSamples
        entropy -= property * log(property, 2)
    return entropy


'''
输入：无
功能：封装原始数据集
输出：数据集、特征标签
'''


def creatDataSet():
    dataset = [[1, 1, 'yes'], [1, 1, 'yes'], [1, 0, 'no'], [0, 1, 'no'], [0, 0, 'no']]
    labels = ['no surfacing', 'flippers']
    return dataset, labels


'''
输入：数据集、数据集中的某一特征所在列的索引、该特征某一可能取值（例如，（原始数据集、0,1 ））
功能：取出在该特征取值下的子数据集（子集不包含该特征）
输出：子数据集
'''


def getSubDataset(dataset, colIndex, value):
    subDataset = []  # 用于存储子数据集
    for rowVector in dataset:
        if rowVector[colIndex] == value:
            # 下边两句实现抽取除第colIndex列特征的其他特征取值
            subRowVector = rowVector[:colIndex]
            subRowVector.extend(rowVector[colIndex + 1:])
            # 将抽取的特征行添加到特征子数据集中
            subDataset.append(subRowVector)
    return subDataset


'''
输入：数据集
功能：选择最优的特征，以便得到最优的子数据集（可简单的理解为特征在决策树中的先后顺序）
输出：最优特征在数据集中的列索引
'''


def BestFeatToGetSubdataset(dataset):
    # 下边这句实现：除去最后一列类别标签列剩余的列数即为特征个数
    numFeature = len(dataset[0]) - 1
    baseEntropy = calcShannonEnt(dataset)
    bestInfoGain = 0.0;
    bestFeature = -1
    for i in range(numFeature):  # i表示该函数传入的数据集中每个特征
        # 下边这句实现抽取特征i在数据集中的所有取值
        feat_i_values = [example[i] for example in dataset]
        uniqueValues = set(feat_i_values)
        feat_i_entropy = 0.0
        for value in uniqueValues:
            subDataset = getSubDataset(dataset, i, value)
            # 下边这句计算pi
            prob_i = len(subDataset) / float(len(dataset))
            feat_i_entropy += prob_i * calcShannonEnt(subDataset)
        infoGain_i = baseEntropy - feat_i_entropy
        if (infoGain_i > bestInfoGain):
            bestInfoGain = infoGain_i
            bestFeature = i
    return bestFeature


'''
输入：子数据集的类别标签列
功能：找出该数据集个数最多的类别
输出：子数据集中个数最多的类别标签
'''


def mostClass(ClassList):
    classCount = {}
    for class_i in ClassList:
        if class_i not in classCount.keys():
            classCount[class_i] = 0
        classCount[class_i] += 1
    sortedClassCount = sorted(classCount.iteritems(),
                              key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


'''
输入：数据集，特征标签
功能：创建决策树（直观的理解就是利用上述函数创建一个树形结构）
输出：决策树（用嵌套的字典表示）
'''


def creatTree(dataset, labels):
    classList = [example[-1] for example in dataset]
    # 判断传入的dataset中是否只有一种类别，是，返回该类别
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    # 判断是否遍历完所有的特征,是，返回个数最多的类别
    if len(dataset[0]) == 1:
        return mostClass(classList)
    # 找出最好的特征划分数据集
    bestFeat = BestFeatToGetSubdataset(dataset)
    # 找出最好特征对应的标签
    bestFeatLabel = labels[bestFeat]
    # 搭建树结构
    myTree = {bestFeatLabel: {}}
    del (labels[bestFeat])
    # 抽取最好特征的可能取值集合
    bestFeatValues = [example[bestFeat] for example in dataset]
    uniqueBestFeatValues = set(bestFeatValues)
    for value in uniqueBestFeatValues:
        # 取出在该最好特征的value取值下的子数据集和子标签列表
        subDataset = getSubDataset(dataset, bestFeat, value)
        subLabels = labels[:]
        # 递归创建子树
        myTree[bestFeatLabel][value] = creatTree(subDataset, subLabels)
    return myTree


'''
输入：测试特征数据
功能：调用训练决策树对测试数据打上类别标签
输出：测试特征数据所属类别
'''


def classify(inputTree, featlabels, testFeatValue):
    firstStr = inputTree.keys()[0]
    secondDict = inputTree[firstStr]
    featIndex = featlabels.index(firstStr)
    for firstStr_value in secondDict.keys():
        if testFeatValue[featIndex] == firstStr_value:
            if type(secondDict[firstStr_value]).__name__ == 'dict':
                classLabel = classify(secondDict[firstStr_value], featlabels, testFeatValue)
            else:
                classLabel = secondDict[firstStr_value]
    return classLabel


'''
输入：训练树，存储的文件名
功能：训练树的存储
输出：
'''


def storeTree(trainTree, filename):
    fw = open(filename, 'w')
    pickle.dump(trainTree, fw)
    fw.close()


def grabTree(filename):
    fr = open(filename)
    return pickle.load(fr)


if __name__ == '__main__':
    dataset, labels = creatDataSet()
    storelabels = labels[:]  # 复制label
    trainTree = creatTree(dataset, labels)
    classlabel = classify(trainTree, storelabels, [0, 1])
    print
    classlabel
