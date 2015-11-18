#-*- coding: UTF-8 -*-
'''
Created on 2015��10��31��

@author: lenovo
'''
from math import *
import random
from tokenize import Double

from numpy.matrixlib.defmatrix import mat

from LogisticReg.logisticRegression import *
import logisticRegression
import numpy as np


data,predictdata = loadData()
print type(data)
dataMat = normalize(data)
# print "DATAMAT normalize",dataMat
labelMat,dataMat,labelSet = divideData(dataMat)
# print "DATAMAT divided",dataMat

weights = gradAscent(dataMat, labelMat,labelSet)
predict = predict(predictdata, weights,labelSet)
print "weights:",weights
print "predict results:",predict
# print weights
