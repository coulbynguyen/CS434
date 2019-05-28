import csv
import random
import math
import operator
import numpy as np
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('trainingdatafile')
parser.add_argument('testingdatafile')
args = parser.parse_args()

trainingdata = []

testingdata = []
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

def test_split(index, value, dataset):
    left, right = list(), list()
    for row in dataset:
        if row[index] < value:
            left.append(row)
        else:
            right.append(row)
    return left, right

def gini_index(groups, classes):
    n_instances = float(sum([len(group) for group in groups]))
    gini = 0.0
    for group in groups:
        size = float(len(group))
        if size == 0:
            continue
        score = 0.0
        for class_val in classes:
            p = [row[0] for row in group].count(class_val)/size
            score += p*p
        gini += (1.0 - score) * (size/n_instances)
    return gini

def predict(node, row):
    if row[node['index']] < node['value']:
        return node['left']
    else:
        return node['right']

def get_split(dataset):
    class_values = list(set(row[0] for row in dataset))
    b_index, b_value, b_score, b_groups = 999, 999, 999, None
    for index in range(len(dataset[0])-1):
        #since the first feature is the label
        index = index+1
        for row in dataset:
            groups = test_split(index, row[index], dataset)
            gini = gini_index(groups, class_values)
            # print('X%d < %.3f Gini=%f' %((index), row[index], gini))
            if gini < b_score:
                b_index, b_value, b_score, b_groups = index, row[index], gini, groups
    return {'index':b_index, 'value':b_value, 'groups':b_groups, 'left':-1, 'right': 1}

split = get_split(trainingdata)

correct = 0

for row in trainingdata:
    prediction = predict(split, row)
    if prediction == row[0]:
        correct += 1
print("Training Error: " + str(1 - float(correct)/len(trainingdata)))

correct = 0

for row in testingdata:
    prediction = predict(split, row)
    if prediction == row[0]:
        correct += 1
print("Testing Error: " + str(1 - float(correct)/len(testingdata)))
# print('Split: [X%d < %.3f]' % ((split['index']), split['value']))
