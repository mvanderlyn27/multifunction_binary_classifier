import pickle as pk
import scipy
import os.path
from os import path
import numpy as np
from scipy.sparse import coo_matrix, vstack,csr_matrix
from sklearn import feature_selection
from sklearn.feature_selection import RFECV
from sklearn.svm import SVR
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

def importData():
    data = None
    ranks = None
    if(path.exists("data.p")):
        data = pk.load(open("data.p","rb"))
        ranks = pk.load(open("ranks.p","rb"))
    else:
        data = []
        rows = []
        cols = []
        ranks = []
        max_len = 0
        max_val = 0
        with open('train_drugs.dat') as fp:
            count = 0
            for line in fp:
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
        vals = np.array(vals,dtype=int)
        rows = np.array(rows,dtype=int)
        cols = np.array(cols,dtype=int)
    dense = np.zeros((len(ranks),max_val+1))
    for c in range(len(cols)):
        dense[rows[c]][cols[c]] = 1
    csr = csr_matrix(dense)
    print()
    return csr,ranks

def rfecvFeatureSelection(sparse_matrix, ranks):
    estimator = SVR(kernel="linear")
    selector = RFECV(estimator, step=1, cv=5)
    selector = selector.fit(sparse_matrix.todense(), ranks) #might need to turn to dense
    return selector, csr_matrix(selector.transform(sparse_matrix))
def mutualInfoSelection(sparse_matrix, ranks): # probably must be done on individual sets of data 
    sparse_matrix,shape = mutual_info_classif(sparse_matrix.todense(),ranks,n_neighbors=5)
    return csr_matrix(sparse_matrix)

def chiSquareSelection(sparse_matrix, ranks):
    selector= SelectKBest(chi2,k=100).fit(sparse_matrix.todense(),ranks)
    return selector, csr_matrix(selector.transform(sparse_matrix.todense()))
def main():
    #import data from training file
    sparse_matrix,ranks = importData()
    #run variance filtering
    threshold = VarianceThreshold(threshold=(0.0975*(1-0.0975)))
    sparse_matrix = threshold.fit(sparse_matrix)
    #run diff feacture selection on sparse matrix
        #features selection using recursive features selection and cross validation
    selectorA,sparse_matrixA = rfecvFeatureSelection(sparse_matrix,ranks)
        #features selection mutual info 
    selectorA,sparse_matrixB = mutualInfoFeatureSelection(sparse_matrix,ranks)
        #run chi^2 selection
    sparse_matrixC = chiSquareSelection(sparse_matrix,ranks)
    #acount for imbalanced data with SMOTE

    #run classifier1 on data

    #run classifier 2 on data

    #somehow test the output