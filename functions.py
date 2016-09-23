# functions for loading data into numpy / xgboost

#import utils
import csv
import numpy as np
#from sklearn.externals.joblib import Memory
from sklearn.datasets import load_svmlight_file




def load_svm_headers4scipy(path):
    headers = []
    #headers.append('pseudotarget') #first column in svmlight: target
    with open(path, 'rb') as f:
        g = csv.reader(f, dialect=csv.excel, delimiter=' ', quotechar='"')
        for row in g:
            if not row[0].startswith('#') and not row[0]=='':
                headers.append(row[1]) #Colname in 2nd column
    return headers

#mem = Memory(cache_dir)
#@mem.cache
def load_svmlight2scipy(path,dtype=np.uint16,n_features=None):
    data = load_svmlight_file(path,n_features=n_features, zero_based=False,dtype=dtype)
    return data[0] #, data[1] # returns X,y

def zero_rows(data):
    ### zero indices
    msk = data==0
    return msk.nonzero()[0]