#=================Binary Classifier of Large Binary Feature Data ===================#
# Programmed By: Michael Vanderlyn                                                  #
# 9/3/19                                                                            #
#                                                                                   #
# Purpose: scan in sparse matrices with ~10,000 features, feature engineer data,    #
# and train at least 2 classifiers to predict test data with as much accuracy as    #
# possible.                                                                         #  
#===================================================================================#

#=============Imports==============#
import pickle as pk
import scipy
import time
import os.path
from os import path
import numpy as np
from scipy.sparse import coo_matrix, vstack,csr_matrix
from sklearn.feature_selection import RFE, RFECV
from sklearn.svm import SVR
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import SMOTENC
from sklearn.decomposition import TruncatedSVD
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import f1_score
from sklearn.tree import DecisionTreeClassifier
#=============Functions==============#

#turns file info into sparse matrix, if keepRanks is true, saves first column as
#binary rank, if false(for test data) doesn't keep rank, and sets width of sparse 
#matrix to width, which should be set to the same as the test data for consistency 
def importData(file, keepRanks,width):
    data = None
    ranks = None
    #checks if files has previous information stored
    if path.exists('./pickles/'+file+"/data.p") and path.exists('./pickles/'+file+"/ranks.p") and \
            path.exists('./pickles/'+file+"/vals.p") and path.exists('./pickles/'+file+"/cols.p") and \
            path.exists('./pickles/'+file+"/rows.p") and path.exists('./pickles/'+file+"/max_val.p"):
        data = pk.load(open('./pickles/'+file+"/data.p","rb"))
        ranks = pk.load(open('./pickles/'+file+"/ranks.p","rb"))
        vals = pk.load(open('./pickles/'+file+"/vals.p","rb"))
        cols = pk.load(open('./pickles/'+file+"/cols.p","rb"))
        rows = pk.load(open('./pickles/'+file+"/rows.p","rb"))
        max_val = pk.load(open('./pickles/'+file+"/max_val.p","rb"))
    #if not scan through data, save ranks and row,col info
    #to build sparse matrix
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
        #save information after computing to save time in repeated 
        #runs of the program
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
    #generates the sparse matrix in dense form, if keepRanks is
    #false then sets sparse matrix width to inputed width, and 
    #only returns the sparse matrix, otherwise returns the rank too
    dense = None
    if(keepRanks):
        dense = np.zeros((len(ranks),max_val+1))
    else:
        dense = np.zeros((count,width))
    for c in range(len(cols)):
        dense[rows[c]][cols[c]] = 1
    csr = csr_matrix(dense)
    print()
    if(keepRanks):
        return csr,ranks
    else:
        return csr

#runs recursive elimination features selection without validation
def rfeFeatureSelection(sparse_matrix, ranks):
    estimator = SVR(kernel="linear")
    selector = RFE(estimator,100,step=1000,verbose=1)
    selector = selector.fit(sparse_matrix, ranks) #might need to turn to dense
    return_sparse_matrix = csr_matrix(selector.transform(sparse_matrix))
    return selector, return_sparse_matrix

#runs recursive elemination features selection with cross validation 
def rfecvFeatureSelection(sparse_matrix,ranks):
    estimator = SVR(kernel="linear")
    selector = RFECV(estimator,min_features_to_select=10,step=1,cv=5, verbose=1)
    selector = selector.fit(sparse_matrix, ranks) #might need to turn to dense
    return_sparse_matrix = csr_matrix(selector.transform(sparse_matrix))
    return selector, return_sparse_matrix


#runs chi^2 analysis on features, selects k best features
def chiSquareSelection(sparse_matrix, ranks):
    selector= SelectKBest(chi2,k=5).fit(sparse_matrix,ranks)
    return selector, csr_matrix(selector.transform(sparse_matrix))

#Feature selects, and reduces dimensions on data, fits classifiers, predicts
#classes for training data, outputs f1 scores, and runs best classifier on 
#test data, saves output to ./test_file_prediction.dat
def main():
    #start timer
    start = time.time()
    #import data from training file and test file
    sparse_matrix,ranks = importData("train_drugs.dat",True,0)
    test_data = importData("test.dat", False,sparse_matrix.shape[1])
   
    #run dimensionality reduction on input data
    selector_tsvd = None
    sparse_matrix_tsvd = []
    if(path.exists("./pickles/selector_tsvd.p") and path.exists("./pickles/sparse_matrix_tsvd.p")):
        selector_tsvd = pk.load(open("./pickles/selector_tsvd.p", "rb"))
        sparse_matrix_tsvd = pk.load(open("./pickles/sparse_matrix_tsvd.p", "rb"))
    else:
        svd = TruncatedSVD(n_components=200, n_iter=7, random_state=42)
        selector_tsvd = svd.fit(sparse_matrix,ranks)
        sparse_matrix_tsvd = selector_tsvd.transform(sparse_matrix)
        pk.dump(selector_tsvd, open("./pickles/selector_tsvd.p", "wb"))
        pk.dump(sparse_matrix_tsvd, open("./pickles/sparse_matrix_tsvd.p", "wb"))
  
   #run features selection on data to remove unimportant features 
   #recursive features selection to remove most of the features that are least important
    selector_rfe = None
    sparse_matrix_rfe = []
    if(path.exists("./pickles/selector_rfe.p") and path.exists("./pickles/sparse_matrix_rfe.p")):
        selector_rfe = pk.load(open("./pickles/selector_rfe.p", "rb"))
        sparse_matrix_rfe = pk.load(open("./pickles/sparse_matrix_rfe.p", "rb"))
    else:
        selector_rfe,sparse_matrix_rfe = rfeFeatureSelection(sparse_matrix,ranks)
        pk.dump(selector_rfe, open("./pickles/selector_rfe.p", "wb"))
        pk.dump(sparse_matrix_rfe, open("./pickles/sparse_matrix_rfe.p", "wb"))
   
    #recursive features selection with cross validation to chose best of reamining features
    selector_rfecv = None
    sparse_matrix_rfecv = []
    if(path.exists("./pickles/selector_rfecv.p") and path.exists("./pickles/sparse_matrix_rfecv.p")):
        selector_rfecv = pk.load(open("./pickles/selector_rfecv.p", "rb"))
        sparse_matrix_rfecv = pk.load(open("./pickles/sparse_matrix_rfecv.p", "rb"))
    else:
        selector_rfecv,sparse_matrix_rfecv = rfecvFeatureSelection(sparse_matrix_rfe,ranks)
        pk.dump(selector_rfecv, open("./pickles/selector_rfecv.p", "wb"))
        pk.dump(sparse_matrix_rfecv, open("./pickles/sparse_matrix_rfecv.p", "wb"))
   
    #run chi^2 selection on original data to see how accurate it is 
    sparse_matrix_chi = []
    selector_chi = None
    if(path.exists("sparse_matrix_chi.p")):
        sparse_matrix_chi = pickle.load(open("./pickles/sparse_matrix_chi.p","rb"))
        selector_chi = pickle.load(open("/pickles/selector_chi.p","rb"))
    else:
        selector_chi,sparse_matrix_chi = chiSquareSelection(sparse_matrix,ranks)
        pk.dump(sparse_matrix_chi, open("./pickles/sparse_matrix_chi.p","wb"))
        pk.dump(selector_chi, open("./pickles/selector_chi.p","wb"))

    #account for imbalanced data with SMOTE over sampling
    Orig_X_resampled,Orig_y_resampled = SMOTE().fit_resample(sparse_matrix.todense(),ranks)
    
    TSVD_X_resampled,TSVD_y_resampled = SMOTE().fit_resample(sparse_matrix_tsvd,ranks)

    rfe_X_resampled,rfe_y_resampled = SMOTE().fit_resample(sparse_matrix_rfe,ranks)

    rfecv_X_resampled,rfecv_y_resampled = SMOTE().fit_resample(sparse_matrix_rfecv,ranks)

    chi_X_resampled,chi_y_resampled = SMOTE().fit_resample(sparse_matrix_chi,ranks)

    #set up classifiers, train on data
    #Bernoulli naive bayes
    nb_orig = BernoulliNB()
    nb_orig.fit(sparse_matrix,ranks)

    nb_orig_resampled = BernoulliNB()
    nb_orig_resampled.fit(Orig_X_resampled, Orig_y_resampled)
   
    nb_tsvd = BernoulliNB()
    nb_tsvd.fit(TSVD_X_resampled, TSVD_y_resampled)

    nb_tsvd_non_sampled = BernoulliNB()
    nb_tsvd_non_sampled.fit(sparse_matrix_tsvd, ranks)
    
    nb_rfec = BernoulliNB()
    nb_rfec.fit(selector_rfe.transform(sparse_matrix), ranks)

    nb_rfecv = BernoulliNB()
    nb_rfecv.fit(rfecv_X_resampled, rfecv_y_resampled)

    nb_rfecv_non_sampled = BernoulliNB()
    nb_rfecv_non_sampled.fit(sparse_matrix_rfecv, ranks)

    nb_chi = BernoulliNB()
    nb_chi.fit(chi_X_resampled, chi_y_resampled) 
    
    #decision tree classifier
    dt_rfecv_resampled = DecisionTreeClassifier(random_state=0)
    dt_rfecv_resampled.fit(rfecv_X_resampled,rfecv_y_resampled)

    dt_rfecv = DecisionTreeClassifier(random_state=0)
    dt_rfecv.fit(sparse_matrix_rfecv,ranks)

    dt_orig = DecisionTreeClassifier(random_state=0)
    dt_orig.fit(sparse_matrix,ranks)

    dt_orig_resampled = DecisionTreeClassifier(random_state=0)
    dt_orig_resampled.fit(Orig_X_resampled,Orig_y_resampled)
    
    dt_tsvd = DecisionTreeClassifier(random_state=0)
    dt_tsvd.fit(sparse_matrix_tsvd,ranks)

    dt_tsvd_resampled = DecisionTreeClassifier(random_state=0)
    dt_tsvd_resampled.fit(TSVD_X_resampled,TSVD_y_resampled)

    dt_chi = DecisionTreeClassifier(random_state = 0)
    dt_chi.fit(chi_X_resampled,chi_y_resampled)

    #run test predictions
    #run naive bayes predictions
    orig_pred = nb_orig.predict(sparse_matrix)
    orig_pred_resamp=nb_orig_resampled.predict(sparse_matrix)
    tsvd_pred = nb_tsvd.predict(selector_tsvd.transform(sparse_matrix)) 
    tsvd_non_sampled_pred = nb_tsvd_non_sampled.predict(selector_tsvd.transform(sparse_matrix))
    rfe_pred = nb_rfec.predict(selector_rfe.transform(sparse_matrix))
    rfecv_pred = nb_rfecv_non_sampled.predict(selector_rfecv.transform(selector_rfe.transform(sparse_matrix)))
    rfecv_pred_non_sampeld = nb_rfecv.predict(selector_rfecv.transform(selector_rfe.transform(sparse_matrix)))
    chi_pred = nb_chi.predict(selector_chi.transform(sparse_matrix))
    
    #run decision tree predictions
    dt_rfecv_resampled_pred = dt_rfecv_resampled.predict(selector_rfecv.transform(selector_rfe.transform(sparse_matrix))) 
    dt_rfecv_pred = dt_rfecv.predict(selector_rfecv.transform(selector_rfe.transform(sparse_matrix))) 
    dt_orig_pred = dt_orig.predict(sparse_matrix)
    dt_orig_resampled_pred = dt_orig_resampled.predict(sparse_matrix)
    dt_tsvd_pred = dt_tsvd.predict(selector_tsvd.transform(sparse_matrix))
    dt_tsvd_resampled_pred = dt_tsvd_resampled.predict(selector_tsvd.transform(sparse_matrix)) 
    dt_chi_pred = dt_chi.predict(selector_chi.transform(sparse_matrix))
    
    #test the output f1 score
    #test for naive bayes
    orig_f1 = f1_score(ranks, orig_pred, average='macro') 
    orig_f1_resampled=f1_score(ranks,orig_pred_resamp,average='macro')
    tsvd_f1 =f1_score(ranks,tsvd_pred,average='macro')
    tvsd_resampled_f1=f1_score(ranks,tsvd_non_sampled_pred,average='macro')
    rfe_f1=f1_score(ranks,rfe_pred,average='macro')
    rfecv_f1 =f1_score(ranks,rfecv_pred,average='macro') 
    rfecv_reasmple_f1_non_sampled =f1_score(ranks,rfecv_pred_non_sampeld,average='macro') 
    chi_f1 =f1_score(ranks,chi_pred,average='macro') 
    
    #test for decision trees
    dt_rfec_resampled_f1 =f1_score(ranks,dt_rfecv_resampled_pred,average='macro')      
    dt_rfecv_f1 =f1_score(ranks,dt_rfecv_pred,average='macro') 
    dt_orig_f1 =f1_score(ranks,dt_orig_pred,average='macro') 
    dt_orig_resampled_f1 =f1_score(ranks,dt_orig_resampled_pred,average='macro') 
    dt_tsvd_f1 =f1_score(ranks,dt_tsvd_pred,average='macro') 
    dt_tsvd_resampled_f1 =f1_score(ranks,dt_tsvd_resampled_pred,average='macro') 
    dt_chi_f1 = f1_score(ranks,dt_chi_pred,average='macro')
    
    #output the different test results
    print('orig:',orig_f1,'orig_resampled:',orig_f1_resampled,'tsvd:',tsvd_f1,'tvsd_resampled_f1:',tvsd_resampled_f1,'rfe_f1:',rfe_f1,'rfecv_f1:',rfecv_f1,'rfecv_reasmple_f1_non_sampled:', rfecv_reasmple_f1_non_sampled,'chi_f1:',chi_f1)
    print('dt_rfec_resampled_f1:',dt_rfec_resampled_f1,'dt_rfecv_f1:',dt_rfecv_f1,'dt_orig_f1:',dt_orig_f1,'dt_orig_resampled_f1:',dt_orig_resampled_f1,'dt_tsvd_f1:',dt_tsvd_f1,'dt_tsvd_resampled_f1:',dt_tsvd_resampled_f1)

    #test with testfile using best classifier
    transformed_data = selector_rfe.transform(test_data)
    test_predict = nb_chi.predict(selector_chi.transform(test_data))
    with open('test_file_prediction.dat', "w") as fp2:
        for num in test_predict:
            fp2.write(str(num)+'\n')
    print(len(test_predict))
    
    #print out time elapsed
    end = time.time()
    print(end - start)  
#=============End of file===============#
main()