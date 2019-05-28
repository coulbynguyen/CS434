import csv
import random
import math
import operator
import numpy as np
import argparse
import matplotlib.pyplot as plt
parser = argparse.ArgumentParser()
parser.add_argument('trainingdatafile')
parser.add_argument('testingdatafile')
args = parser.parse_args()

k = 0


trainingdata = []

testingdata = []

myk = []
plotTrainingError = []
plotLOOCV = []
plotTestingError = []

#Read in the training data into a list
with open(args.trainingdatafile) as csvfile:
    reader = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC)
    for row in reader:
        trainingdata.append(row)
#Read in the testing data as a list
with open(args.testingdatafile) as csvfile:
    reader = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC)
    for row in reader:
        testingdata.append(row)


def euclideanDistance(instance1, instance2, length):
    distance = 0
    for x in range(length):
        #since the label is the first element
        #we dont want to include it in the euclidean distance calculation
        x = x+1
        distance += pow((instance1[x] - instance2[x]),2)
    return math.sqrt(distance)

def getNeighbors(trainingSet, testInstance, k):
    distances = []
    length = len(testInstance)-1
    for x in range(len(trainingSet)):
        dist = euclideanDistance(testInstance, trainingSet[x], length)
        distances.append((trainingSet[x], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors

def getResponse(neighbors):
    classVotes = {}
    for x in range(len(neighbors)):
        #get the label of the neihbors near the training data
        response = neighbors[x][0]
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1
    sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
    return sortedVotes[0][0]

def getError(testSet, predictions):
    correct = 0
    for x in range(len(testSet)):
        if testSet[x][0] == predictions[x]:
            correct += 1
    return 1-(correct/float(len(testSet)))

def main():

    predictions = []
    for x in range(len(trainingdata)):
        neighbors = getNeighbors(trainingdata, trainingdata[x], k)
        result = getResponse(neighbors)
        predictions.append(result)
        # print('> predicted=' + repr(result) + ', actual=' + repr(trainingdata[x][0]))
    trainingError = getError(trainingdata, predictions)


    predictions = []
    #leave one out cross validation
    for x in range(len(trainingdata)):
        validationExample = trainingdata.pop(0)
        neighbors = getNeighbors(trainingdata, validationExample, k)
        result = getResponse(neighbors)
        predictions.append(result)
        # print('> predicted=' + repr(result) + ', actual=' + repr(validationExample[0]))
        trainingdata.append(validationExample)
    leaveoneoutError = getError(trainingdata, predictions)


    predictions = []
    for x in range(len(testingdata)):
        neighbors = getNeighbors(trainingdata, testingdata[x], k)
        result = getResponse(neighbors)
        predictions.append(result)
        # print('> predicted=' + repr(result) + ', actual=' + repr(testingdata[x][0]))
    testingError = getError(testingdata, predictions)

    # print('Training Error: ' + repr(trainingError))
    # print('Leave One Out Cross Validation Error: ' + repr(leaveoneoutError))
    # print('Testing Error: ' + repr(testingError))
    myk.append(k);
    plotTrainingError.append(trainingError);
    plotLOOCV.append(leaveoneoutError);
    plotTestingError.append(testingError);

for i in range(1, 52, 2):
    k = i
    main()
# print(myk)
# print(plotTrainingError)

plt.plot(myk, plotTrainingError, 'bo', label='training error', markersize=9)
plt.plot(myk, plotLOOCV, 'ys', label="Leave One Out Cross Validation Error")
plt.plot(myk, plotTestingError, 'r^', label='testing error')
plt.xlabel("value of K")
plt.ylabel("%Error")
plt.legend(loc='upper left')
plt.show()
