import pickle as pk
import scipy
import os.path
from os import path
import numpy as np
from scipy.sparse import coo_matrix, vstack

data = None
ranks = None
if(False):#path.exists("data.p")):
    #df = pk.load(open("data.p","rb"))
    data = pk.load(open("data.p","rb"))
    ranks = pk.load(open("ranks.p","rb"))
else:
    #df = pd.read_fwf("train_drugs.dat", infer_nrows=800)
    #df = pd.read_csv("train_drugs.dat", index_col=0, sep=" ",dtype=np.int32)
    data = []
    rows = []
    cols = []
    ranks = []
    max_len = 0
    max_val = 0
    with open('train_drugs.dat') as fp:
        count = 0
        for line in fp:
            #print(count)
            nums = [int(x) for x in line.split()]
            if(len(cols) == 0):
                cols = nums[1:]
                rows = [count] * len(cols)
                vals = [1] * len(cols)
                ranks = [nums[0]]
            else:
                cols.extend(nums[1:])
                rows.extend([count] * len(nums[1:]))
                vals.extend([1] * len(nums[1:]))
                ranks.append(nums[0])
            count+=1
            if(len(nums[1:])> max_len):
                max_len = len(nums[1:])
            if(max(nums[1:])> max_val):
                max_val = max(nums[1:])
    pk.dump(data, open("data.p","wb"))
    pk.dump(ranks, open("ranks.p","wb"))

print(len(vals))
print(len(rows))
print(len(cols))
print(len(ranks))
vals = np.array(vals,dtype=int)
rows = np.array(rows,dtype=int)
cols = np.array(cols,dtype=int)
print(vals.dtype)
print(rows.dtype)
print(cols.dtype)

#vals.reshape((1, -1))
#rows.reshape((1, -1))
#cols.reshape((1, -1))
#print(vals)
#print(rows)
#print(cols)
csr = coo_matrix(vals,(rows,cols))
print(csr)
