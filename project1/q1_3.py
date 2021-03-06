import sys
import numpy as np
from numpy.linalg import inv
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('trainingdatafile')
parser.add_argument('testingdatafile')
args = parser.parse_args()

#open the files for inputs
trainingfile = open(args.trainingdatafile, "r")

testingfile = open(args.testingdatafile, "r")


#initialize variables as empty lists
trainingdataX = []
trainingdataXt = []
trainingdataY = []

testingdataY = []
testingdatapredictY = []


#for each data set in the housing training file
#the data set is split by the spaces and represented as a list
#then the last item in the list gets removed, since that item is the label
#the label is added to the list of labels that correlate to the training features
#then the dummy variable and the rest of the features are added to a 2 dimensional matrix that contains all the traiing data
for x in trainingfile:
    myarray = list(map(float, x.split()))
    trainingdataY.append(myarray.pop())
    trainingdataX.append(myarray)

#this closes the file as we need the training data later to check the ASe
trainingfile.close()

#this line transforms the training data matrix into an np.ndarray in order to perform matrix operations such as transpose and inverse
trainingdataX = np.asarray(trainingdataX)
trainingdataXt = trainingdataX.transpose()

#this line calculates weight vector w from the equation (([Xt]*[X])^-1)*[Xt]*[Y]
w = np.matmul(np.matmul(inv(np.matmul(trainingdataXt, trainingdataX)), trainingdataXt),trainingdataY)

#this transforms the nd array back into a list for easier manipulation
w = w.tolist()
#this prints out the weight vector to std.out
print(w)


#this creates a list that will hold the predicted values of the training data
trainingdatapredictY = []
#this reopens the training data to be able to calculate the ASE
predicttrainingfile = open(args.trainingdatafile, "r")

#this calculates the predicted values using the weight vector for the training data set
#it does this by taking each feature and multiplying it by the corresponding weight
#and summing all of those values and storing it into the trainingpredictY list
for x in predicttrainingfile:
    myarray = list(map(float, x.split()))
    myarray.pop()
    i = 0
    predictedY = 0
    for n in myarray:
        predictedY += n*w[i]
        i += 1
    trainingdatapredictY.append(predictedY)

#this creates and initializes i and ASE to 0
i = 0.0
ASE = 0.0

#this calculates the ASE for the training data
#it does this by calculating the difference from the expected value and the predicted value
#squaring the difference and summing all of those values and dividing by the number of data points
for x,y in zip(trainingdataY, trainingdatapredictY):
    i += 1
    diff = x - y
    diff = diff**2
    ASE += diff

#this prints out the ASE for the training data
print("Training data ASE:" + str(ASE/i))


#this calculates the predicted values using the weight vector for the training data set
#it does this by taking each feature and multiploying it by the corresponding weight
#and summing all of those values and storing it into the trainingdatapredictY list
for x in testingfile:
    myarray = list(map(float, x.split()))
    testingdataY.append(myarray.pop())
    i = 0
    predictedY = 0
    for n in myarray:
        predictedY += n*w[i]
        i += 1
    testingdatapredictY.append(predictedY)

#this resets the i and ASE variables back to 0
i = 0.0
ASE = 0.0

#this calculates the ASE for the training data
#it does this by calculating the difference from the expected value and the predicted value
#squaring the difference and summing all of those values and dividing by the number of data point
for x,y in zip(testingdataY, testingdatapredictY):
    i += 1
    diff = x - y
    diff = diff**2
    ASE += diff
#this prints out the testing data ASE
print("Testing data ASE: " + str(ASE / i))
