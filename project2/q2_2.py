import csv
import random
import math
import operator
import numpy as np
import argparse
import matplotlib.pyplot as plt
from scipy.stats import entropy
from math import log, e
parser = argparse.ArgumentParser()
parser.add_argument('trainingdatafile')
parser.add_argument('testingdatafile')
parser.add_argument('d')
args = parser.parse_args()

d = int(args.d)

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
        ent -= i * log(i, 2)

    return ent

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
            entropy = (entropy2(group_labels_left) + entropy2(group_labels_right))
            # print('X%d < %.3f Gini=%f' %((index), row[index], gini))
            if entropy < b_score:
                b_index, b_value, b_score, b_groups = index, row[index], entropy, groups
    return {'index':b_index, 'value':b_value, 'groups':b_groups}

# Create a terminal node value
def to_terminal(group):
	outcomes = [row[0] for row in group]
	return max(set(outcomes), key=outcomes.count)

# Create child splits for a node or make terminal
def split(node, max_depth, min_size, depth):
	left, right = node['groups']
	del(node['groups'])
	# check for a no split
	if not left or not right:
		node['left'] = node['right'] = to_terminal(left + right)
		return
	# check for max depth
	if depth >= max_depth:
		node['left'], node['right'] = to_terminal(left), to_terminal(right)
		return
	# process left child
	if len(left) <= min_size:
		node['left'] = to_terminal(left)
	else:
		node['left'] = get_split(left)
		split(node['left'], max_depth, min_size, depth+1)
	# process right child
	if len(right) <= min_size:
		node['right'] = to_terminal(right)
	else:
		node['right'] = get_split(right)
		split(node['right'], max_depth, min_size, depth+1)

# Build a decision tree
def build_tree(train, max_depth, min_size):
	root = get_split(train)
	split(root, max_depth, min_size, 1)
	return root

# Make a prediction with a decision tree
def predict(node, row):
	if row[node['index']] < node['value']:
		if isinstance(node['left'], dict):
			return predict(node['left'], row)
		else:
			return node['left']
	else:
		if isinstance(node['right'], dict):
			return predict(node['right'], row)
		else:
			return node['right']

#build_tree(trainingset, max_depth, mininum node size)
tree = build_tree(trainingdata, d, 1)

correct = 0

for row in trainingdata:
    prediction = predict(tree, row)
    if prediction == row[0]:
        correct += 1
print("Training Error: " + str(1 - float(correct)/len(trainingdata)))

correct = 0

for row in testingdata:
    prediction = predict(tree, row)
    if prediction == row[0]:
        correct += 1
print("Testing Error: " + str(1 - float(correct)/len(testingdata)))


# d = [1,2,3,4,5,6]
# training_error = []
# testing_error = []
#
# for i in d:
# tree = build_tree(trainingdata, d, 1)
#
# correct = 0
#
# for row in trainingdata:
#     prediction = predict(tree, row)
#     if prediction == row[0]:
#         correct += 1
# training_error.append(1-float(correct)/len(trainingdata))
#
# correct = 0
#
# for row in testingdata:
#     prediction = predict(tree, row)
#     if prediction == row[0]:
#         correct += 1
# testing_error.append(1-float(correct)/len(testingdata))


# plt.figure(1)
# plt.title("Decision Tree Error")
# plt.plot(d, training_error, 'ro', label='Training Error')
# plt.plot(d, testing_error, 'b^', label='Testing Error')
# plt.xlabel("Tree Depth")
# plt.ylabel("Error")
# plt.legend(loc='upper right')
# plt.show()
