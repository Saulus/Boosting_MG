# Build big AMTS model

import utils
import settings
import h2o
from h2o.estimators.glm import H2OGeneralizedLinearEstimator
#from h2o.connection import H2OConnection
import csv
import sys
import os
import math
import time
import pandas as pd
import gc


####### SETTINGS ###########
#settings = settings.development()
settings = settings.production()

# save stdout
old_stdout = sys.stdout
#sys.stdout = open(settings.out_log, 'w')



def load_headers (path):
    headers = []
    headers.append('pseudotarget') #first column in svmlight: target
    with open(path, 'rb') as f:
        g = csv.reader(f, dialect=csv.excel, delimiter=' ', quotechar='"')
        for row in g:
            if not row[0].startswith('#') and not row[0]=='':
                headers.append(row[1]) #Colname in 2nd column
    return headers

def import_svmlight(path,headers=""):
    raw = h2o.lazy_import(path)
    if settings.debug and len(headers)<100: print utils.time() + "import with headers: " + str(headers)
    #parsesetup = h2o.parse_setup(raw,column_names=headers)
    parsesetup = h2o.parse_setup(raw) # Issue: H2O 3.8 tests length of header vs. columns, but still imports the "pseudotarget" additionally
    parsesetup['parse_type'] = 'SVMLight'
    loaded_frame = h2o.parse_raw(parsesetup)
    if settings.debug:
        print "......HEader length: " + str(len(headers))
        print "......Frame imported: "+ str(loaded_frame.ncol)
    if (len(headers) > loaded_frame.ncol):
        n = len(headers) - loaded_frame.ncol
        print "Remove last "+str(n)+" header entries"
        del headers[-n:]
    loaded_frame.set_names(headers) #Workaround, Set names now
    print "First column: " + loaded_frame.names[0] #needed because lazy name setting
    if settings.debug and len(headers)<100: loaded_frame.head(show=True)
    loaded_frame.pop(0) #remove first ('pseudotarget') columnn 
    #if loaded_frame.ncol>len(headers)-1: #workaround: H2O reads info from svmlight into columns -> remove everything that is not in headers
    #    delete = []
    #    for i in xrange(len(headers)-1,loaded_frame.ncol):
    #        delete.append(loaded_frame.names[i])
    #    loaded_frame = remove_vecs(loaded_frame,delete)
    if settings.debug and len(headers)<100: loaded_frame.head(show=True)
    return loaded_frame
    
def upload_svmlight(path,headers=""):
    loaded_frame = h2o.upload_file(path)
    if settings.debug and len(headers)<100: print utils.time() + "import with headers: " + str(headers)
    if settings.debug:
        print "......HEader length: " + str(len(headers))
        print "......Frame imported: "+ str(loaded_frame.ncol)
    if (len(headers) > loaded_frame.ncol):
        n = len(headers) - loaded_frame.ncol
        print "Remove last "+str(n)+" header entries"
        del headers[-n:]
    loaded_frame.set_names(headers) 
    print "First column: " + loaded_frame.names[0] #needed because lazy name setting
    if settings.debug and len(headers)<100: loaded_frame.head(show=True)
    loaded_frame.pop(0) #remove first ('pseudotarget') columnn 
    if settings.debug and len(headers)<100: loaded_frame.head(show=True)
    return loaded_frame

####
#### STEP 0: clean output dir
####
if settings.clean_output:
    print utils.time() + "Loesche Files im Output Verzeichnis "+settings.out_directory_client
    filelist = [ f for f in os.listdir(settings.out_directory_client) if os.path.isfile(os.path.join(settings.out_directory_client, f)) ]
    for f in filelist:
        os.remove(os.path.join(settings.out_directory_client, f))

####
#### STEP 1: init h2o, load data
####
print utils.time() + "Initialisiere H2O Server: " + settings.ip + ":"+str(settings.port)
h2o.init(ip=settings.ip, port=settings.port, start_h2o=False)
#h2o.remove_all()

###Output file
if settings.start_icd=="":
    with open(settings.out_model, 'w') as f:
            f.write(';'.join(['target','train_auc','train_threshhold','train_precision','train_recall','valid_auc','valid_precision','valid_recall','glm_auc','model_id']) + '\n')

    with open(settings.out_nodes, 'w') as f:
            f.write(';'.join(["Node","prevalence","incidence","Mean age of incident","Mean age prevalence"]) + '\n')

    with open(settings.out_edges, 'w') as f:
            f.write(';'.join(["Source","Target","OR","beta","p-value","number relations","proportion of incidents have Source","proportion Source get Target","Mean age of incident patients with Condition Source"]) + '\n')
           
        
#Load headers
headerfile =  settings.in_directory_client+'/'+settings.colnames_file
print utils.time() + "Lade Header File "+headerfile
headers=load_headers(headerfile)
print "......HEader length: " + str(len(headers))
if settings.debug:
    print "......"+headers[0]
    print "......"+headers[1]

#Load features
print utils.time() + "Lade Feature File "+settings.mat_file
if settings.do_upload:
    featurefile=settings.in_directory_client+'/'+settings.mat_file
    matrix = upload_svmlight(featurefile,headers)
else:
    featurefile=settings.in_directory_server+'/'+settings.mat_file
    matrix = import_svmlight(featurefile,headers)
print "......Matrix: " + str(matrix.nrow) + " x " +str(matrix.ncol)
if settings.debug:
    print "......"+matrix.names[0]
    print "......"+matrix.names[1]
    print "......"+matrix.names[15]
    #raw_input("Press Enter to continue...")

################################TEST ###################################
#pre_col_vector = matrix["PRE_ICD_M54"] == 0
#pre_col_vector =  matrix["PRE_ICD_M54"].which()
#print "pre_col_vector: " + str(pre_col_vector.nrow)
#possible_incidents = matrix[pre_col_vector==0]
#possible_incidents = matrix[pre_col_vector, :]
#print "Possible incidents: " + str(possible_incidents.nrow)
#raw_input("Press Enter to continue...")
#index_cols = ["IDX_ICD_M54","IDX_ICD_B37","IDX_ICD_C88"]
#idx_means_h = matrix[index_cols].apply(lambda col: col.mean(), axis = 0)
#print "... Means: " + str(idx_means_h.ncol) + " cols"  + str(idx_means_h.nrow)
#print idx_means_h.summary()
#for a in idx_means_h:
#df = idx_means_h.as_data_frame(use_pandas=False)
#print df
#idx_means = {a: df[1][i] for i,a in enumerate(df[0])}
#    print a[0,0]
#    print a.names[0]
    #print a[-1]
#idx_means = {a.names[0]: a[0,0] for a in idx_means_h}

#test_vector =  matrix[1:10,"PRE_ICD_M54"]
#p = test_vector.as_data_frame(use_pandas=False)
#print idx_means
#raw_input("Press Enter to continue...")

################################TEST END###################################
    
####
#### STEP 2: Data prep: remove columns, and change Target to factor
####
print utils.time() + "Data preparation (find columns)"
cols_to_be_deleted = []
index_cols = []
index_ATC_cols = []
index_ICD_cols = []
pre_cols = []
post_cols = []
for i_c in xrange(0,matrix.ncol):
    if matrix.names[i_c] in settings.del_cols:
        cols_to_be_deleted.append(matrix.names[i_c])
    if matrix.names[i_c].startswith(settings.index_prefix):
        index_cols.append(matrix.names[i_c])
        if "ATC" in matrix.names[i_c]:
            index_ATC_cols.append(matrix.names[i_c])
        elif "ICD" in matrix.names[i_c]:
            index_ICD_cols.append(matrix.names[i_c])
    if matrix.names[i_c].startswith(settings.pre_prefix):
        pre_cols.append(matrix.names[i_c])
    if matrix.names[i_c].startswith(settings.post_prefix):
        post_cols.append(matrix.names[i_c])
if settings.debug:
    print "......To be deleted: " + str(len(cols_to_be_deleted)) + " cols"
    print "......Index Cols " + str(len(index_cols)) + " cols"
    print "......Pre Cols " + str(len(pre_cols)) + " cols"
    print "......Post Cols " + str(len(post_cols)) + " cols"
#matrix = remove_vecs(matrix,cols_to_be_deleted)

#raw_input("Press Enter to continue...")
#add count cols
print utils.time() + "add count cols"
IDX_COUNT_ATC = matrix[index_ATC_cols].apply(lambda row: row.sum(), axis = 1)
IDX_COUNT_ATC = IDX_COUNT_ATC.set_names(["IDX_COUNT_ATC"])
IDX_COUNT_ICD = matrix[index_ICD_cols].apply(lambda row: row.sum(), axis = 1)
IDX_COUNT_ICD = IDX_COUNT_ICD.set_names(["IDX_COUNT_ICD"])
index_cols.append("IDX_COUNT_ATC")
index_cols.append("IDX_COUNT_ICD")
matrix = matrix.cbind(IDX_COUNT_ATC)
matrix = matrix.cbind(IDX_COUNT_ICD)

#raw_input("Press Enter to continue...")
#calc prevalences and related mean ages on matrix by apply
print utils.time() + "Start calculation Prevalences / Mean Ages"
prevalences = matrix[index_cols].apply(lambda col: col.mean(), axis = 0) #requires index to be 2010 (1 year)
prevalences_count = matrix[index_cols].apply(lambda col: col.sum(), axis = 0) #requires index to be 2010 (1 year)
prevalences_ages_frame = matrix[index_cols]*matrix[settings.age]
prevalences_ages_sum = prevalences_ages_frame.apply(lambda col: col.sum(), axis = 0) 
prevalences_ages = prevalences_ages_sum / prevalences_count
print "... prevalences: " + str(prevalences.ncol) + " cols"
print "... Ages: " + str(prevalences_ages.ncol) + " cols"
if settings.debug:
        print prevalences.summary()
        print prevalences_ages.summary()


#raw_input("Press Enter to continue...")
#clear h2o
h2o.remove(IDX_COUNT_ATC)
IDX_COUNT_ATC = None
h2o.remove(IDX_COUNT_ICD)
IDX_COUNT_ICD = None
h2o.remove(prevalences_count)
prevalences_count = None
h2o.remove(prevalences_ages_frame)
prevalences_ages_frame = None
h2o.remove(prevalences_ages_sum)
prevalences_ages_sum = None
#gc python
gc.collect()
#tell back-end cluster nodes to do three back-to-back JVM full GCs.
h2o.connection.H2OConnection.post("GarbageCollect")
h2o.connection.H2OConnection.post("GarbageCollect")
h2o.connection.H2OConnection.post("GarbageCollect")

#loop through post_cols
start = False
for c_i, post_col in enumerate(post_cols):
    if post_col==settings.start_icd or settings.start_icd=="":
        start = True
    if start:
        print utils.time() + 'Work on ' + post_col
        # calculate node statistics
        node = post_col.replace(settings.post_prefix,"")
        idx_col= settings.index_prefix + node
        if post_col=="POST_DEATH":
            possible_incidents = h2o.deep_copy(matrix,"poss_inc")
        else:
            pre_col=settings.pre_prefix + node
            pre_col_vector = matrix[pre_col]
            print "...Anteil possible incidents: " + str(pre_col_vector.mean()) #needed for tricking h2o to not lazy calc this vector -> MEM issues
            possible_incidents = matrix[pre_col_vector==0]
            h2o.remove(pre_col_vector)
            pre_col_vector=None
        print "...Possible incidents: " + str(possible_incidents.nrow) #needed for tricking h2o to not lazy calc this vector -> MEM issues
        post_col_vector = possible_incidents[post_col]
        print "...Anteil real incidents: " + str(post_col_vector.mean())
        real_incidents = possible_incidents[post_col_vector>0]
        print "...Real incidents: " + str(real_incidents.nrow)
        
        print "...calc node prevalence + incidence"
        if post_col=="POST_DEATH":
            prevalence = 0
            mean_age_prevalents = 0
        else:
            prevalence = prevalences[idx_col].min()
            mean_age_prevalents = prevalences_ages[idx_col].min()
            
        if settings.debug:
            print prevalence
            print mean_age_prevalents
        
        incidence4y = post_col_vector.mean()[0]
        incidence1y = incidence4y/settings.years
        mean_age_incidents = real_incidents[settings.age].mean()[0]+2 #add 2 as Age comes from 2010, incedence is 2011 - 2014

        #pre-calc means and sums for idx (=all possible coeffs) on real incidence
        print "...Pre-calc means and sums for idx (=all possible coeffs) on real incidence"
        idx_means_h = real_incidents[index_cols].apply(lambda col: col.mean(), axis = 0)
        idx_sums_h = real_incidents[index_cols].apply(lambda col: col.sum(), axis = 0)
        # pre-calc mean ages for idx (=all possible coeffs) on real incidence, using matrix apply
        idx_ages_frame = real_incidents[index_cols]*real_incidents[settings.age]
        idx_ages_sum = idx_ages_frame.apply(lambda col: col.sum(), axis = 0) 
        idx_ages_h = idx_ages_sum / idx_sums_h + 2 #add 2 as Age comes from 2010, incidence is 2011 - 2014
        print "... Means: " + str(idx_means_h.ncol) + " cols"
        print "... Sums: " + str(idx_sums_h.ncol) + " cols"
        print "... Ages: " + str(idx_ages_h.ncol) + " cols"
        if settings.debug:
            print idx_means_h.summary()
            print idx_sums_h.summary()
            print idx_ages_h.summary()
        
        df = idx_means_h.as_data_frame(use_pandas=False)
        idx_means = {a: df[1][i] for i,a in enumerate(df[0])}
        df = idx_sums_h.as_data_frame(use_pandas=False)
        idx_sums = {a: df[1][i] for i,a in enumerate(df[0])}
        df = idx_ages_h.as_data_frame(use_pandas=False)
        idx_ages = {a: df[1][i] for i,a in enumerate(df[0])}
        df = {}
        h2o.remove(idx_ages_frame)
        idx_ages_frame = None
        h2o.remove(idx_ages_sum)
        idx_ages_sum = None
        h2o.remove(idx_ages_h)
        idx_ages_h = None
        h2o.remove(idx_means_h)
        idx_means_h = None
        h2o.remove(idx_sums_h)
        idx_sums_h = None
        

        # change target to factor
        if settings.debug:
            print "change target to factor"
        possible_incidents[post_col] = possible_incidents[post_col].asfactor()
        
        #split training and test
        ratio = 0.7
        print utils.time() + "Separiere Training und Val Daten"
        r1=possible_incidents.runif(seed=123456)
        if settings.debug:
            print r1.summary()
        training = possible_incidents[r1<ratio] # split out X% for training
        val = possible_incidents[r1>=ratio]
        if settings.debug:
            print "......Training matrix: " + str(training.nrow) + " x " +str(training.ncol)
            print "......Training matrix: " + training.frame_id
            print "......Val matrix: " + str(val.nrow) + " x " +str(val.ncol)


        #model1: GLM with ElasticNet
        print utils.time() + 'Erstelle Modell 1 ' + post_col
        model1 = H2OGeneralizedLinearEstimator(model_id=post_col,
                                              family = 'binomial', solver='IRLSM', alpha = 0.99, lambda_search=True, max_active_predictors=400, #max_active_predictors not working in H2o 3.8 or 3.10, so need to set max runtime... working at all?
                                              max_runtime_secs=60,
                                              #nfolds=5, fold_assignment='Random',
                                              standardize=True, intercept=True)

        model1.train(x=index_cols, y=post_col, training_frame=training, validation_frame=val, max_runtime_secs=60)
        #if settings.debug:
        #    print possible_incidents.frame_id
        #    print(possible_incidents['IDX_ICD_E51'].sum())

        coeffs = model1.coef()
        print "...All coeffs: " + str(len(coeffs))
        if settings.debug:
            print(coeffs)
        coeffs_selected= [key for key,value in coeffs.iteritems() if value != 0 and key != "Intercept"]
        print "...Selected coeffs: " + str(len(coeffs_selected))
        if settings.debug:
            print (coeffs_selected)
            #raw_input("Press Enter to continue...")

        #model2: STandard-GLM (for p-values)
        print utils.time() + 'Erstelle Modell 2 ' + post_col
        model2 = H2OGeneralizedLinearEstimator(model_id=post_col+'_GLM',
                                              family = 'binomial', solver='IRLSM', Lambda = 0,
                                              standardize=False, intercept=True, compute_p_values=True,
                                              remove_collinear_columns=True)

        model2.train(x=coeffs_selected, y=post_col, training_frame=possible_incidents)

        coeff_pvalues = model2._model_json["output"]["coefficients_table"].cell_values
        p_values = {a[0]: a[-1] for a in coeff_pvalues}

        #change target back to numeric (reason: statistics) -> not needed
        # possible_incidents[post_col] = possible_incidents[post_col].asnumeric()

        #output
        #Model parameters
        #quality parameters: AUC, Precision, Recall
        train_auc1 = model1.auc(train=True)
        train_threshhold1 = model1.find_threshold_by_max_metric("f1",train=True)
        train_precision1 = model1.precision(thresholds=[train_threshhold1],train=True)[0][1]
        train_recall1 = model1.recall(thresholds=[train_threshhold1],train=True)[0][1]
        valid_auc1 = model1.auc(valid=True)
        valid_precision1 = model1.precision(thresholds=[train_threshhold1],valid=True)[0][1]
        valid_recall1 = model1.recall(thresholds=[train_threshhold1],valid=True)[0][1]

        glm_auc = model2.auc(train=True)

        with open(settings.out_model, 'a') as f:
            f.write(';'.join([post_col,str(train_auc1),str(train_threshhold1),str(train_precision1),str(train_recall1),str(valid_auc1),str(valid_precision1),str(valid_recall1),str(glm_auc),model1.model_id]) + '\n')

        with open(settings.out_nodes, 'a') as f:
            f.write(';'.join([node,str(prevalence),str(incidence1y),str(mean_age_incidents),str(mean_age_prevalents)]) + '\n')

        edges = []
        edges.append(["Intercept",node,math.exp(coeffs["Intercept"]),coeffs["Intercept"],p_values["Intercept"],0,0,0,0,0])
        
        print "...Calculate Keys"
        for key in coeffs_selected:
                edges.append([key.replace(settings.index_prefix,""),node,math.exp(coeffs[key]),coeffs[key],p_values[key],idx_sums[key],idx_means[key],"na",idx_ages[key]])

        with open(settings.out_edges, 'a') as f:
            writer = csv.writer(f, delimiter=';', quoting=csv.QUOTE_NONE, lineterminator='\n')
            writer.writerows(edges)

        #clear H2O
        h2o.remove(possible_incidents)
        possible_incidents=None
        h2o.remove(post_col_vector)
        post_col_vector=None
        h2o.remove(real_incidents)
        real_incidents=None
        h2o.remove(model1)
        model1=None
        h2o.remove(model2)
        model2=None
        h2o.remove(r1)
        r1=None
        h2o.remove(training)
        training=None
        h2o.remove(val)
        val=None
        
        #gc python
        gc.collect()
        #tell back-end cluster nodes to do three back-to-back JVM full GCs.
        h2o.connection.H2OConnection.post("GarbageCollect")
        h2o.connection.H2OConnection.post("GarbageCollect")
        h2o.connection.H2OConnection.post("GarbageCollect")
        #time.sleep(10) # delay for 30 seconds
    


sys.stdout.close()
#set back
sys.stdout = old_stdout

#os.system('python ddi.py.py')