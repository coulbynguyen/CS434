import csv
import math
import numpy as np
import argparse
#parser = argparse.ArgumentParser()
#parser.add_argument('datafile')
#args = parser.parse_args()

data = []

#with open(args.datafile) as csvfile:
with open('p4-data.txt') as csvfile:
    reader = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC)
    for row in reader:
        data.append(row)

covMat = np.cov(np.asarray(data).transpose())

e_val, e_vec = np.linalg.eigh(covMat)

eigen_pairs = [(np.abs(e_val[i]), e_vec[:,i]) for i in range(len(e_val))]
eigen_pairs.sort(key=lambda x: x[0], reverse=True)
for x in range(10):
    print(eigen_pairs[x][0])
