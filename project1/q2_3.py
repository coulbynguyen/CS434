import csv
import math
import numpy as np
from numpy import linalg as LA
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('trainingdatafile')
parser.add_argument('testingdatafile')
parser.add_argument('lambdas')
args = parser.parse_args()
#this initializes the variables to empty lists

rawlambdas = args.lambdas.split(',')
lambdas = [float(i) for i in rawlambdas]

iterations = []
trainingaccuracy = []
testingaccuracy = []

for n in lambdas:

    rawtrainingdata = []
    trainingdata = []
    trainingdataY = []


    #this reads in the data from the training csv file and trainsforms it into a 2 dimensional matrix where
    #each row is one set of data
    with open(args.trainingdatafile) as csvfile:
        reader = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC)
        for row in reader:
            rawtrainingdata.append(row)

    #this also removes the labels from feature set and places it in a correalating list of expected values
    for x in rawtrainingdata:
        #This seperates the label from the features
        trainingdataY.append(x.pop())
        trainingdata.append(x)

    #this creates the w to be all 0's and is the size of all the features plus the dummy feature
    w = [0]*256




    #this iterates through algorithm in order to train w
    #it does this by calculating 1/(1+ e^(-w o x))
    #the dotwx is the dot product of w and x
    # then it raises e to the negative of this value
    # then for all features the change in the x values is calculated by multiplying the predicted Y - the expected Y
    # then the new gradient is calculated by adding each value of the change in X to the old gradient
    # then the gradient is updated
    # and after each iteration of the training data, w is updated by the gradient
    # it is updated by muliplying the gradient by a lamda value that is the learning rate
    #still need to change the gradient as a command line argument
    correct = 0.0
    total = 0.0
    traincorrect = 0.0
    traintotal = 0.0
    for count in range(10):
        #this creates the gradient and sets all values to 0 and is the exact same size as w
        gradientw = [0]*256
        for x,y in zip(trainingdata, trainingdataY):
            dotwx = np.dot(w, x)
            predictY = 1/(1 + math.e**(-dotwx))
            changeX = [i * (predictY - y) for i in x]
            newgradientw = [gradientw[i] + changeX[i] for i in range(len(x))]
            gradientw = newgradientw
        neww = [w[i] - n*(gradientw[i] + LA.norm(w) ) for i in range(len(w))]
        w = neww

        traincorrect = 0.0
        traintotal = 0.0
        for x,y in zip (trainingdata, trainingdataY):
            mysum = 0
            for i,j in zip(w, x):
                mysum += i*j
            # print("Predicted: " + str(mysum) + " Actual: " + str(y))
            if mysum < 0 and y == 0:
                traincorrect += 1
                traintotal += 1
            elif mysum > 0 and y == 1:
                traincorrect += 1
                traintotal += 1
            else:
                traintotal += 1


        #this creates and initalizes the variables to empty lists for the testing data
        rawtestingdata = []
        testingdata = []
        testingdataY = []

        #this reads the testing csv file data and adds it to raw testing data list
        with open(args.testingdatafile) as csvfile:
            reader = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC)
            for row in reader:
                rawtestingdata.append(row)

        #this creates the features matrix for the testing data and the correlating expected Y value
        for x in rawtestingdata:
            testingdataY.append(x.pop())
            testingdata.append(x)

        #this calculates the value that will determine which class the feature set should belong too
        #if the value mysum is < 0 it should be a 4
        #or if the value mysum is > 0 it should be a 9
        correct = 0.0
        total = 0.0
        for x,y in zip (testingdata, testingdataY):
            mysum = 0
            for i,j in zip(w, x):
                mysum += i*j
            # print("Predicted: " + str(mysum) + " Actual: " + str(y))
            if mysum < 0 and y == 0:
                correct += 1
                total += 1
            elif mysum > 0 and y == 1:
                correct += 1
                total += 1
            else:
                total += 1
    trainingaccuracy =  trainingaccuracy + [traincorrect/traintotal]
    testingaccuracy =  testingaccuracy + [correct/total]
    iterations = iterations + [n]

    print(iterations)
    print(trainingaccuracy)
    print(testingaccuracy)

plt.figure(1)
plt.title("Training Accuracy")
plt.plot(iterations, trainingaccuracy, 'ro')
# plt.axis([0.00001, 100, 0, 1])
plt.xlabel("The value of Lambda")
plt.ylabel("Accuracy Percentage")
plt.show()

plt.figure(2)
plt.title("Testing Accuracy")
plt.plot(iterations, testingaccuracy, 'b^')
# plt.axis([0.00001, 100, 0, 1])
plt.xlabel("The value of Lambda")
plt.ylabel("Accuracy Percentage")
plt.show()
