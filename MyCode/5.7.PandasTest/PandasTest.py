"""
#Online read the csv file. It doens't work in my mac. 
import pandas
df = pandas.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header = None)
df.tail()
"""

import csv
import random
import matplotlib.pyplot
import numpy
import operator
import math

def loadDataset(filename, split, trainingSet=[], testSet=[]):
    with open(filename, 'r') as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines)
        for x in range(len(dataset)-1):
            for y in range(4):
                dataset[x][y] = float(dataset[x][y])

            if random.random() < split:
                trainingSet.append(dataset[x])
            else:
                testSet.append(dataset[x])

def EuclidDist(instance1, instance2, length):
	distance = 0
	for x in range(length):
		distance += pow((instance1[x] - instance2[x]), 2)
	return math.sqrt(distance)

def getNeighbors(trainSet, testInstance, k):
	distances = []
	length = len(testInstance)-1
	for x in range(len(trainSet)):
		dist = EuclidDist(testInstance, trainSet[x], length)
		distances.append((trainSet[x], dist))
	distances.sort(key=operator.itemgetter(1))
	neighbors = []
	for x in range(k):
		neighbors.append(distances[x][0])
	return neighbors

trainingSet = []
testSet = []
loadDataset('iris.data', 0.80, trainingSet, testSet)
print('Training Samples: ' + repr(len(trainingSet)))
print('Test Samples:' + repr(len(testSet)))

neighbors = getNeighbors(trainingSet, testSet[0], 1)
print('Test one case: ' + repr(testSet[0]))
print('Found neighbor: ' + repr(neighbors))



