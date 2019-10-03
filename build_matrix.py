import pickle as pk
import scipy
import os.path
from os import path
import numpy as np
from scipy.sparse import coo_matrix, vstack,csr_matrix
#from sklearn import feature_selection
from sklearn.feature_selection import RFECV
from sklearn.svm import SVR
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import SMOTENC

from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import f1_score
def importData(file, keepRanks):
    data = None
    ranks = None
    if path.exists('./pickles/'+file+"/data.p") and path.exists('./pickles/'+file+"/ranks.p") and \
            path.exists('./pickles/'+file+"/vals.p") and path.exists('./pickles/'+file+"/cols.p") and \
            path.exists('./pickles/'+file+"/rows.p") and path.exists('./pickles/'+file+"/max_val.p"):
        data = pk.load(open('./pickles/'+file+"/data.p","rb"))
        ranks = pk.load(open('./pickles/'+file+"/ranks.p","rb"))
        vals = pk.load(open('./pickles/'+file+"/vals.p","rb"))
        cols = pk.load(open('./pickles/'+file+"/cols.p","rb"))
        rows = pk.load(open('./pickles/'+file+"/rows.p","rb"))
        max_val = pk.load(open('./pickles/'+file+"/max_val.p","rb"))
    else:
        data = []
        rows = []
        cols = []
        ranks = []
        max_len = 0
        max_val = 0
        with open(file) as fp:
            count = 0
            for line in fp:
                nums = [int(x) for x in line.split()]
                if(len(cols) == 0):
                    if(keepRanks):
                        cols = nums[1:]
                        rows = [count] * len(cols)
                        vals = [1] * len(cols)
                        ranks = [nums[0]]
                    else:
                        cols = nums
                        rows = [count] * len(cols)
                        vals = [1] * len(cols)
                else:
                    if(keepRanks):
                        cols.extend(nums[1:])
                        rows.extend([count] * len(nums[1:]))
                        vals.extend([1] * len(nums[1:]))
                        ranks.append(nums[0])
                    else:
                        cols.extend(nums)
                        rows.extend([count] * len(nums))
                        vals.extend([1] * len(nums))
                count+=1
                if(keepRanks):
                    if(len(nums[1:])> max_len):
                        max_len = len(nums[1:])
                    if(max(nums[1:])> max_val):
                        max_val = max(nums[1:])
                else:
                    if(len(nums)> max_len):
                        max_len = len(nums)
                    if(max(nums[1:])> max_val):
                        max_val = max(nums)
        pk.dump(data, open('./pickles/'+file+"/data.p","wb"))
        if(keepRanks):
            pk.dump(ranks, open('./pickles/'+file+"/ranks.p","wb"))
        vals = np.array(vals,dtype=int)
        pk.dump(vals, open('./pickles/'+file+"/vals.p","wb"))
        rows = np.array(rows,dtype=int)
        pk.dump(rows, open('./pickles/'+file+"/rows.p","wb"))
        cols = np.array(cols,dtype=int)
        pk.dump(cols, open('./pickles/'+file+"/cols.p","wb"))
        pk.dump(max_val, open('./pickles/'+file+"/max_val.p","wb"))
    dense = None
    if(keepRanks):
        dense = np.zeros((len(ranks),max_val+1))
    else:
        dense = np.zeros((count, max_val+1))
    for c in range(len(cols)):
        dense[rows[c]][cols[c]] = 1
    csr = csr_matrix(dense)
    print()
    if(keepRanks):
        return csr,ranks
    else:
        return csr
def varThreshold(s_m):
    threshold = VarianceThreshold(threshold=(0.0975*(1-0.0975)))
    s_m = threshold.fit_transform(s_m)
    return s_m, threshold
def rfecvFeatureSelection(sparse_matrix, ranks):
    estimator = SVR(kernel="linear")
    selector = RFECV(estimator, min_features_to_select=75, step=1, cv=3, verbose=1)
    selector = selector.fit(sparse_matrix, ranks) #might need to turn to dense
    return_sparse_matrix = csr_matrix(selector.transform(sparse_matrix))
    return selector, return_sparse_matrix
def mutualInfoFeatureSelection(sparse_matrix, ranks): # probably must be done on individual sets of data 
    sparse_matrix = mutual_info_classif(sparse_matrix,ranks,n_neighbors=5)
    return csr_matrix(sparse_matrix)
def chiSquareSelection(sparse_matrix, ranks):
    selector= SelectKBest(chi2,k=5).fit(sparse_matrix,ranks)
    return selector, csr_matrix(selector.transform(sparse_matrix))

def main():
    #import data from training file
    sparse_matrix,ranks = importData("train_drugs.dat",True)
    test_data = importData("test.dat", False)
    #run variance filtering
    threshold_matrix = []
    threshold = None
    if(path.exists("./pickles/threshold_matrix.p") and path.exists("./pickles/varThreshold.p")):
        threshold_matrix = pk.load(open("./pickles/threshold_matrix.p", "rb"))
        threshold = pk.load(open("./pickles/varThreshold.p", "rb"))
    else:
        matrix,threshold = varThreshold(sparse_matrix)
        pk.dump(matrix, open("./pickles/threshold_matrix.p", "wb"))
        pk.dump(threshold, open("./pickles/varThreshold.p", "wb"))
    #run diff feacture selection on sparse matrix
        #features selection using recursive features selection and cross validation
    selector_rfecv = None
    sparse_matrix_rfecv = []
    if(path.exists("./pickles/selector_rfecv.p") and path.exists("./pickles/sparse_matrix_rfecv.p")):
        selector_rfecv = pk.load(open("./pickles/selector_rfecv.p", "rb"))
        sparse_matrix_rfecv = pk.load(open("./pickles/sparse_matrix_rfecv.p", "rb"))
    else:
        selector_rfecv,sparse_matrix_rfecv = rfecvFeatureSelection(threshold_matrix,ranks)
        pk.dump(selector_rfecv, open("./pickles/selector_rfecv.p", "wb"))
        pk.dump(sparse_matrix_rfecv, open("./pickles/sparse_matrix_rfecv.p", "wb"))
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
    #print(sparse_matrix_rfecv)
    #print(sparse_matrix_mutual)
    #print(sparse_matrix_chi)

    #acount for imbalanced data with SMOTE
    category_mask1 = [True]*len(sparse_matrix.todense())
    #Orig_X_resampled,Orig_y_resampled = SMOTENC(categorical_features=category_mask1).fit_resample(sparse_matrix.todense(),ranks)
    Orig_X_resampled,Orig_y_resampled = SMOTE().fit_resample(sparse_matrix.todense(),ranks)
    Thresh_X_resampled,Thresh_y_resampled = SMOTE().fit_resample(threshold_matrix.todense(),ranks)
    RFECV_X_resampled,RFECV_y_resampled = SMOTE().fit_resample(sparse_matrix_rfecv.todense(),ranks)
    #run classifiers on data
        #decision tree

        #Bernoulli naive bayes
    clf_orig = BernoulliNB()
    print(Orig_X_resampled.shape)
    print(Orig_y_resampled.shape)
    clf_orig.fit(Orig_X_resampled, Orig_y_resampled)
    
    clf_thresh = BernoulliNB()
    print(Thresh_X_resampled.shape)
    print(Thresh_y_resampled.shape)
    clf_thresh.fit(Thresh_X_resampled, Thresh_y_resampled)
    
    clf_rcef = BernoulliNB()
    print(RFECV_X_resampled.shape)
    print(RFECV_y_resampled.shape)
    print(RFECV_X_resampled)
    print(RFECV_y_resampled)
    X = np.random.randint(2, size=(6, 100))
    Y = np.array([1, 2, 3, 4, 4, 5])
    clf_rcef.fit(X, Y)
    
    #somehow test the output f1 score
    orig_pred = clf_orig.predict(sparse_matrix)
    test_set1 = threshold.transform(sparse_matrix)
    thresh_pred = clf_thresh.predict(test_set1)
    test_set2 = selector_rfecv.transform(test_set1)

    
    #rcefv_pred = clf_rcef.predict(test_set2)

    orig_f1 = f1_score(ranks, orig_pred, average='macro') 
    thresh_f1 = f1_score(ranks, orig_pred, average='macro') 
    #rcefv_f1 = f1_score(ranks[:300], orig_pred, average='macro') 
    print('orig:',orig_f1,'threshold:',thresh_f1) #,'rcef',rcefv_f1)

    #test with testfile
    test_predict = clf_thresh.predict(threshold.transform(test_data))
    with open('test_file_prediction.dat', "w") as fp2:
        for num in test_predict:
            fp2.write(str(num)+'\n')
    print(test_predict)
main()