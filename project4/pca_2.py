import csv
import math
import numpy as np
import argparse
import matplotlib.pyplot as plt
#parser = argparse.ArgumentParser()
#parser.add_argument('datafile')
#args = parser.parse_args()

data = []

mean_vector = []
#with open(args.datafile) as csvfile:
with open('p4-data.txt') as csvfile:
    reader = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC)
    for row in reader:
        data.append(row)

npdata = np.asarray(data)

for x in range(len(data[0])):
    mean_vector.append(npdata[:,x].mean())



covMat = np.cov(np.asarray(data).transpose())

e_val, e_vec = np.linalg.eigh(covMat)

eigen_pairs = [(np.abs(e_val[i]), e_vec[:,i]) for i in range(len(e_val))]
eigen_pairs.sort(key=lambda x: x[0], reverse=True)

ten_eigen_values = np.asarray([eigen_pairs[i][0] for i in range(10)])
ten_eigen_vector = np.asarray([eigen_pairs[i][1] for i in range(10)])

np_mean_vector = np.asarray(mean_vector)

plt.figure(1)
plt.title("Representation of the Mean Image")
plt.imshow(np_mean_vector.reshape((28,28)))
plt.show()

i = 1
for vec in ten_eigen_vector:
    plt.figure(i+1)
    plt.title("Eigen Vector #"+str(i))
    plt.imshow(vec.reshape((28,28)))
    plt.show()
    i += 1

