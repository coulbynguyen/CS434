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

reduced_repr = np.dot(ten_eigen_vector, npdata.transpose())
image_idxs = np.argmin(reduced_repr, axis=1)

fig = plt.figure(figsize=(25,5))
for x in range(10):
    plt.subplot(2, 10, x+1)
    plt.imshow(np.reshape(data[image_idxs[x]], (28,28)))
    plt.subplot(2, 10, x+11)
    plt.imshow(np.reshape(ten_eigen_vector[x].real, (28,28)))
plt.tight_layout()
try:
    plt.show()
except:
    print("cannot show graph")
plt.savefig("PCA_img_vec.png")



