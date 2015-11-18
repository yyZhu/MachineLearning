#-*- coding: UTF-8 -*-
'''
Created on 2015��10��31��

@author: lenovo
'''
import copy
from math import *
from nt import remove
import random

from numpy.core.fromnumeric import shape
from numpy.matlib import ones, zeros
from numpy.matrixlib.defmatrix import mat

import numpy as np


# from numpy.matrixlib.defmatrix import mat
train_y = []
test_x = []
test_y = []
data = []
traindata = []
testdata = []
# dataMat = []
# labelMat = []
predictdata = []
def loadData():
    f = open("dataset.txt")
    lines = f.readlines()
    for line in lines:
        data.append(line.split()) 
    f.close()
    f2 = open("predictdata.txt")
    lines = f2.readlines()
    for line in lines:
        predictdata.append(line.split()) 
    f2.close()
    m,n = shape(predictdata)
    return data,predictdata
    

def divideData(dataSet):
    total = len(dataSet)   ####行
    ratio = int(total * 0.9)
    resultList = random.sample(range(0,total),total)  ######相当于打乱
#     testdata = copy.deepcopy(dataSet)
    dataMat = dataSet[resultList[0:ratio-1]] 
#     print "zuizhong datamat length",len(dataMat)
    labelMat = []
    testdata = dataSet[resultList[ratio:total]]
    for i in range(len(dataMat)):
        labelMat.append(dataMat.tolist()[i].pop())
    
#     labelMat = dataMat[:,dataMat.shape[1] - 1]
#     print "zuizhong labelMat",labelMat
    dataMat = dataMat[:,0:dataMat.shape[1] - 1]
    
   
    ####multi 
#     labelMat2 = labelMat.tolist()
    
#     labelMat3 = sorted(set(labelMat2),key=labelMat2.index)
    labelSet = []
    label = []
    for x in labelMat:
        label.append(x)
        if x not in labelSet :
            labelSet.append(x)
#     print "label label label label label label",label
    return label,dataMat,labelSet
        
######对数据进行归一化处理,读入的为str型的数据，若是数字的str，map成int，若是字母如汉字，则map成0,1normalize要具体写
def normalize(dataset):
    
    for i in range(len(dataset)):
        dataset[i].insert(0,'1.0')
        dataset[i] = map(eval, dataset[i])
    dataset = mat(dataset)
#         dataset[i] = mat(dataset[i])                
    for i in range(dataset.shape[1] - 1):   
        dataset[:,i] = dataset[:,i] / dataset[:,i].max()
#     print "dataset dataset is:",dataset.tolist()
    return dataset
#     for i in range(len(dataset) - 1):
#         if isinstance(dataset[i], float):
#             dataset[i] = 1


def sigMoid(x):
    return 1.0 / (1 + np.exp(-x))

def fakeSigMoid(x):
    m,n = shape(x)
#     print "m,n",m,n
    sum = 0
    results = zeros((m,n)).tolist()
    for i in range(m):
        for j in range(n):
            sum = sum + np.exp(x.tolist()[i][j])
            sum = np.exp(x.tolist()[i][j]) / (1.0 + sum)
            results[i][j] = (sum)
    return results

alpha = 0.001
def gradAscent(data,label,labelSet):
    hmat = zeros((data.shape[0],len(labelSet) - 1)).tolist()
#     print "hmat[2][1]",hmat[2][1]
    i = 0
#     print "ge zhong changdu",len(labelSet) - 1,data.shape[0]
    for k in range(len(labelSet) - 1):
        for j in range(data.shape[0]):
            if(label[j] == labelSet[k]):
                hmat[j][k] = 1
            
    iteration = 1
    error = 0.00001
    m,n = shape(data)
    labelNum = len(labelSet) - 1
    if(labelNum >= 2):
        weights = ones((n,labelNum))
    else:
        weights = ones((n,1))
    
#     for i in range(iteration):
    for i in range(labelNum):
        diff = 1
        hmat = mat(hmat)
        while(diff > error):
            if(labelNum == 1): 
                h = sigMoid(data * weights[:,i])
                h = mat(h)
                deri = mat(label).transpose() - h  ###11*1
            else:
                h = fakeSigMoid(data * weights)
                h = mat(h)
                deri = hmat[:,i] - h[:,i]  ###11*1
    #         cichu hai zhengchang
            formal = copy.deepcopy(weights[:,i])
            weights[:,i] = weights[:,i] + alpha * data.transpose() * deri   ####梯度下降法目标函数取最小值，所以此处为+
            diff = abs(formal.transpose() * formal - weights[:,i].transpose() * weights[:,i])
            print "diff = ",diff
    return weights

def predict(predictdata,weights,labelSet):
    m,n = shape(weights)
    predictresult = []  
    kk = labelSet.pop()       
    if n == 1:
        predictdata = normalize(predictdata)
        results = sigMoid(predictdata * weights)
        for i in range(len(results) - 1):
            if(results[i] > 0.5):
                print results[i]
                predictresult.append(1)
            else:
                predictresult.append(0)
    else:
        predictdata = normalize(predictdata)
        results = fakeSigMoid(predictdata * weights)
        for i in range(m):
            for j in range(n):
                if results[i][j] > 0.5:
                    predictresult.append(labelSet[j])
                    break
                else:
                    predictresult.append(kk)
            
        
    return predictresult
            
        
    