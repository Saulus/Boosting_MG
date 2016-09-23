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


####### SETTINGS ###########
#settings = settings.development()
settings = settings.production()

# save stdout
old_stdout = sys.stdout
sys.stdout = open(settings.out_log, 'w')



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
h2o.remove_all()

###Output file
with open(settings.out_model, 'w') as f:
        f.write(';'.join(['target','train_auc','train_threshhold','train_precision','train_recall','valid_auc','valid_precision','valid_recall','glm_auc','model_id']) + '\n')

with open(settings.out_edges, 'w') as f:
        f.write(';'.join(["Source","Target","OR","beta","p-value","number relations","proportion of incidents have Source","proportion Source get Target","Mean age of incident patients with Condition Source"]) + '\n')

with open(settings.out_nodes, 'w') as f:
        f.write(';'.join(["Node","prevalence","incidence","Mean age of incident","Mean age prevalence"]) + '\n')

#Load headers
headerfile =  settings.in_directory_client+'/'+settings.colnames_file
print utils.time() + "Lade Header File "+headerfile
headers=load_headers(headerfile)
print "......HEader length: " + str(len(headers))
if settings.debug:
    print "......"+headers[0]
    print "......"+headers[1]

#Load features
featurefile=settings.in_directory_server+'/'+settings.mat_file
print utils.time() + "Lade Feature File "+featurefile
matrix = import_svmlight(featurefile,headers)
print "......Matrix: " + str(matrix.nrow) + " x " +str(matrix.ncol)
if settings.debug:
    print "......"+matrix.names[0]
    print "......"+matrix.names[1]
    print "......"+matrix.names[15]
    #raw_input("Press Enter to continue...")

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


#add count cols
IDX_COUNT_ATC = matrix[index_ATC_cols].apply(lambda row: row.sum(), axis = 1)
IDX_COUNT_ATC = IDX_COUNT_ATC.set_names(["IDX_COUNT_ATC"])
IDX_COUNT_ICD = matrix[index_ICD_cols].apply(lambda row: row.sum(), axis = 1)
IDX_COUNT_ICD = IDX_COUNT_ICD.set_names(["IDX_COUNT_ICD"])
index_cols.append("IDX_COUNT_ATC")
index_cols.append("IDX_COUNT_ICD")
matrix = matrix.cbind(IDX_COUNT_ATC)
matrix = matrix.cbind(IDX_COUNT_ICD)

#loop through post_cols
for c_i, post_col in enumerate(post_cols):
    print utils.time() + 'Work on ' + post_col
    # calculate node statistics
    node = post_col.replace(settings.post_prefix,"")
    idx_col= settings.index_prefix + node
    idx_col_vector = matrix[idx_col] #requires index to be 2010 (1 year)
    pre_col=settings.pre_prefix + node
    pre_col_vector = matrix[pre_col]
    possible_incidents = matrix[pre_col_vector==0]
    post_col_vector = possible_incidents[post_col]
    real_incidents = possible_incidents[post_col_vector>0]
    prevalents = matrix[idx_col_vector>0]
    
    if settings.debug:
        print "calc node prevalence + incidence"
    prevalence = idx_col_vector.mean()[0] #requires index to be 2010 (1 year)
    mean_age_prevalents = prevalents[settings.age].mean()[0]
    incidence4y = post_col_vector.mean()[0]
    incidence1y = incidence4y/settings.years
    mean_age_incidents = real_incidents[settings.age].mean()[0]+2 #add 2 as Age comes from 2010, incedence is 2011 - 2014


    # change target to factor
    if settings.debug:
        print "change target to factor"
    possible_incidents[post_col] = possible_incidents[post_col].asfactor()


    #model1: GLM with ElasticNet
    print utils.time() + 'Erstelle Modell 1 ' + post_col
    model1 = H2OGeneralizedLinearEstimator(model_id=post_col,
                                          family = 'binomial', solver='L_BFGS', alpha = 0.9, lambda_search=False,
                                          nfolds=5, fold_assignment='Random',
                                          standardize=True, intercept=True)

    model1.train(x=index_cols, y=post_col, training_frame=possible_incidents)
    #if settings.debug:
    #    print possible_incidents.frame_id
    #    print(possible_incidents['IDX_ICD_E51'].sum())

    coeffs = model1.coef()
    if settings.debug:
        print "All coeffs: " + str(len(coeffs))
        print(coeffs)
    coeffs_selected= [key for key,value in coeffs.iteritems() if value != 0 and key != "Intercept"]
    if settings.debug:
        print "Selected coeffs: " + str(len(coeffs_selected))
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
    valid_auc1 = model1.auc(xval=True)
    valid_precision1 = model1.precision(thresholds=[train_threshhold1],xval=True)[0][1]
    valid_recall1 = model1.recall(thresholds=[train_threshhold1],xval=True)[0][1]

    glm_auc = model2.auc(train=True)

    with open(settings.out_model, 'a') as f:
        f.write(';'.join([post_col,str(train_auc1),str(train_threshhold1),str(train_precision1),str(train_recall1),str(valid_auc1),str(valid_precision1),str(valid_recall1),str(glm_auc),model1.model_id]) + '\n')

    with open(settings.out_nodes, 'a') as f:
        f.write(';'.join([node,str(prevalence),str(incidence1y),str(mean_age_incidents),str(mean_age_prevalents)]) + '\n')

    with open(settings.out_edges, 'a') as f:
        f.write(';'.join(["Intercept",node,str(math.exp(coeffs["Intercept"])),str(coeffs["Intercept"]),str(p_values["Intercept"]),str(0),str(0),str(0),str(0),str(0)]) + '\n')

        for key in coeffs_selected:
            print "...Calculate " + key
            if key not in [settings.age]: #, settings.gender, settings.hosp
                key_vec1 = possible_incidents[key]
                possible_incidents_source = possible_incidents[key_vec1>0]
                key_vec2 = real_incidents[key]
                real_incidents_source = real_incidents[key_vec2>0]

                number = real_incidents_source.nrow
                number_by_incidents = float(number)/ real_incidents.nrow
                #proportion = float(number) / matrix[key>0].nrow -> not calculated here
                mean_age_inc_source = real_incidents_source[settings.age].mean()[0]+2 #add 2 as Age comes from 2010, incedence is 2011 - 2014
            else:
                proportion = 0
                mean_age_inc_source = 0

            f.write(';'.join([key.replace(settings.index_prefix,""),node,str(math.exp(coeffs[key])),str(coeffs[key]),str(p_values[key]),str(number),str(number_by_incidents),"na",str(mean_age_inc_source)]) + '\n')

            #clear H2O
            #h2o.remove(key_vec1)
            #h2o.remove(possible_incidents_source)
            #h2o.remove(real_incidents_source)
            #h2o.remove(key_vec2)

    #clear H2O
    #h2o.remove(idx_col_vector)
    #h2o.remove(pre_col_vector)
    #h2o.remove(possible_incidents)
    #h2o.remove(post_col_vector)
    #h2o.remove(real_incidents)
    #h2o.remove(prevalents)

sys.stdout.close()
#set back
sys.stdout = old_stdout

#os.system('python ddi.py.py')