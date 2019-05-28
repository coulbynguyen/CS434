import csv
import random
import math
import operator
import numpy as np
import argparse
from scipy.stats import entropy
from math import log, e
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
    entropy = 0.0
    for group in groups:
        group_labels = [i[0] for i in group]
        entropy += entropy2(group_labels)

    return entropy

def entropy2(labels, base=None):
    """ Computes entropy of label distribution. """

    n_labels = len(labels)

    if n_labels <= 1:
        return 0

    value,counts = np.unique(labels, return_counts=True)
    probs = counts / n_labels
    n_classes = np.count_nonzero(probs)

    if n_classes <= 1:
        return 0

    ent = 0.

    # Compute entropy
    base = e if base is None else base
    for i in probs:
        ent -= i * log(i, base)

    return ent

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
            group_labels_left = [i[0] for i in groups[0]]
            group_labels_right = [i[0] for i in groups[1]]
            entropy = (entropy2(group_labels_left) + entropy2(group_labels_right)) / 2
            # print('X%d < %.3f Gini=%f' %((index), row[index], gini))
            if entropy < b_score:
                b_index, b_value, b_score, b_groups = index, row[index], entropy, groups


    group_labels = [i[0] for i in dataset]
    info_gain = entropy2(group_labels) - entropy
    print("The learned decision stump: Is Feature %d < %f" % (b_index, b_value))
    print("The computed information gain value is: %f" % info_gain)
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
