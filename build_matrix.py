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
from imblearn.over_sampling import SMOTE
def importData():
    data = None
    ranks = None
    if path.exists("data.p") and path.exists("ranks.p") and \
            path.exists("vals.p") and path.exists("cols.p") and \
            path.exists("rows.p") and path.exists("max_val.p"):
        data = pk.load(open("data.p","rb"))
        ranks = pk.load(open("ranks.p","rb"))
        vals = pk.load(open("vals.p","rb"))
        cols = pk.load(open("cols.p","rb"))
        rows = pk.load(open("rows.p","rb"))
        max_val = pk.load(open("max_val.p","rb"))
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
        pk.dump(vals, open("vals.p","wb"))
        rows = np.array(rows,dtype=int)
        pk.dump(rows, open("rows.p","wb"))
        cols = np.array(cols,dtype=int)
        pk.dump(cols, open("cols.p","wb"))
        pk.dump(max_val, open("max_val.p","wb"))
    dense = np.zeros((len(ranks),max_val+1))
    for c in range(len(cols)):
        dense[rows[c]][cols[c]] = 1
    csr = csr_matrix(dense)
    print()
    return csr,ranks
def varThreshold(s_m):
    threshold = VarianceThreshold(threshold=(0.0975*(1-0.0975)))
    s_m = threshold.fit_transform(s_m)
    return s_m, threshold
def rfecvFeatureSelection(sparse_matrix, ranks):
    estimator = SVR(kernel="linear")
    selector = RFECV(estimator, min_features_to_select=75, step=1, cv=3, verbose=1)
    selector = selector.fit(sparse_matrix.todense(), ranks) #might need to turn to dense
    return_sparse_matrix = csr_matrix(selector.transform(sparse_matrix))
    return selector, return_sparse_matrix
def mutualInfoFeatureSelection(sparse_matrix, ranks): # probably must be done on individual sets of data 
    sparse_matrix = mutual_info_classif(sparse_matrix,ranks,n_neighbors=5)
    return csr_matrix(sparse_matrix)
def chiSquareSelection(sparse_matrix, ranks):
    selector= SelectKBest(chi2,k=5).fit(sparse_matrix.todense(),ranks)
    return selector, csr_matrix(selector.transform(sparse_matrix.todense()))
def main():
    #import data from training file
    sparse_matrix,ranks = importData()
    #run variance filtering
    threshold_matrix = []
    threshold = None
    if(path.exists("threshold_matrix.p") and path.exists("varThreshold.p")):
        threshold_matrix = pk.load(open("threshold_matrix.p", "rb"))
        threshold = pk.load(open("varThreshold.p", "rb"))
    else:
        matrix,threshold = varThreshold(sparse_matrix)
        pk.dump(matrix, open("threshold_matrix.p", "wb"))
        pk.dump(csr_matrix(threshold), open("varThreshold.p", "wb"))
    #run diff feacture selection on sparse matrix
        #features selection using recursive features selection and cross validation
    selector_rfecv = None
    sparse_matrix_rfecv = []
    if(path.exists("selector_rfecv.p") and path.exists("sparse_matrix_rfecv.p")):
        selector_rfecv = pk.load(open("threshold_matrix.p", "rb"))
        sparse_matrix_rfecv = pk.load(open("varThreshold.p", "rb"))
    else:
        selector_rfecv,sparse_matrix_rfecv = rfecvFeatureSelection(threshold_matrix,ranks)
        print(selector_rfecv)
        print(sparse_matrix_rfecv)
        pk.dump(selector_rfecv, open("selector_rfecv.p", "wb"))
        pk.dump(sparse_matrix_rfecv, open("sparse_matrix_rfecv.p", "wb"))
        #features selection mutual info 
    # sparse_matrix_inf = []
    # if(path.exists("sparse_matrix_mutual.p")):
    #     sparse_matrix_mutual = pk.load(open("sparse_matrix_mutual.p","rb"))
    # else:
    #     sparse_matrix_mutual = mutualInfoFeatureSelection(threshold_matrix,ranks)
    #     pk.dump(sparse_matrix_mutual, open("sparse_matrix_mutual.p","wb"))
    #     #run chi^2 selection
    # sparse_matrix_chi = []
    # if(path.exists("sparse_matrix_chi.p")):
    #     sparse_matrix_chi = pickle.load(open("sparse_matrix_chi.p","rb"))
    # else:
    #     sparse_matrix_chi = chiSquareSelection(threshold_matrix,ranks)
    #     pk.dump(sparse_matrix_chi, open("sparse_matrix_chi.p","wb"))
    
    #print(threshold_matrix)
    print(sparse_matrix_rfecv)
    #print(sparse_matrix_mutual)
    #print(sparse_matrix_chi)

    #acount for imbalanced data with SMOTE
    Orig_X_resampled,Orig_y_resampled = SMOTE().fit_resample(sparse_matrix.todense(),ranks)
    Thresh_X_resampled,Thresh_y_resampled = SMOTE().fit_resample(threshold_matrix,ranks)
    RFECV_X_resampled,RFECV_y_resampled = SMOTE().fit_resample(sparse_matrix_rfecv,ranks)
    #run classifier1 on data

    #run classifier 2 on data

    #somehow test the output
main()