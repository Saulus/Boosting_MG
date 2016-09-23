# Build MG with boosting

import utils
import settings3 as settings
import functions
import sys
import os
import csv

mingw_path = 'C:\\Program Files\\mingw-w64\\x86_64-6.2.0-posix-seh-rt_v5-rev1\\mingw64\\bin'
os.environ['PATH'] = mingw_path + ';' + os.environ['PATH']
import xgboost as xgb
from scipy.sparse import csr_matrix, hstack
from sklearn.cross_validation import train_test_split
from sklearn.feature_selection import f_classif
from indexed import IndexedOrderedDict
import math


####### SETTINGS ###########
settings = settings.development()
#settings = settings.production()

# save stdout
old_stdout = sys.stdout
#sys.stdout = open(settings.out_log, 'w')


####
#### STEP 0: clean output dir
####
if settings.clean_output:
    print utils.time() + "Loesche Files im Output Verzeichnis "+settings.out_directory_client
    filelist = [ f for f in os.listdir(settings.out_directory_client) if os.path.isfile(os.path.join(settings.out_directory_client, f)) ]
    for f in filelist:
        os.remove(os.path.join(settings.out_directory_client, f))

####
#### STEP 1: load data
#### Result: scikit-learn csr matrix, headers
####
headerfile = settings.in_directory_client+'/'+settings.colnames_file
print utils.time() + "Lade Header File "+headerfile
headers=functions.load_svm_headers4scipy(headerfile)
print "......Header length: " + str(len(headers))
if settings.debug:
    print "......"+headers[0]
    print "......"+headers[1]
#Load features
featurefile=settings.in_directory_client+'/'+settings.mat_file
print utils.time() + "Lade Feature File "+featurefile
matrix = functions.load_svmlight2scipy(featurefile, dtype=settings.inputtype, n_features=len(headers))
print "......Matrix: ", matrix.get_shape()

####
#### STEP 2: init output
####
###Output files
if settings.start_icd=="":
    with open(settings.out_model, 'w') as f:
            f.write(';'.join(['target','train_auc','train_error','train_precision (map)','train_no_top1%','train_precision_top1% (map@x)','eval_auc','eval_error','eval_precision (map)','eval_no_top1%','eval_precision_top1% (map@x)']) + '\n')

    with open(settings.out_nodes, 'w') as f:
            f.write(';'.join(["Node","prevalence","incidence","Mean age of incident","Mean age prevalence"]) + '\n')

    with open(settings.out_edges, 'w') as f:
            f.write(';'.join(["Source","Target","OR","beta","p-value","number relations","proportion of incidents have Source","proportion Source get Target","Mean age of incident patients with Condition Source"]) + '\n')

    
####
#### STEP 2: identify columns
####
print utils.time() + "Data preparation (find columns)"
#cols_to_be_deleted = {}
index_cols = IndexedOrderedDict()
index_ATC_cols = IndexedOrderedDict()
index_ICD_cols = IndexedOrderedDict()
pre_cols = IndexedOrderedDict()
post_cols = IndexedOrderedDict()
#dict format: "ICD_E11": 5 (i.e. without prefix, pointing to matrix column index)
for i_c in xrange(0,len(headers)):
    #if headers[i_c] in settings.del_cols:
    #   cols_to_be_deleted[headers[i_c]]=i_c
    if headers[i_c].startswith(settings.index_prefix):
        index_cols[headers[i_c].replace(settings.index_prefix,"")]=i_c
        if "ATC" in headers[i_c]:
            index_ATC_cols[headers[i_c].replace(settings.index_prefix,"")]=i_c
        elif "ICD" in headers[i_c]:
            index_ICD_cols[headers[i_c].replace(settings.index_prefix,"")]=i_c
    if headers[i_c].startswith(settings.pre_prefix):
        pre_cols[headers[i_c].replace(settings.pre_prefix,"")]=i_c
    if headers[i_c].startswith(settings.post_prefix):
        post_cols[headers[i_c].replace(settings.post_prefix,"")]=i_c

if settings.debug:
    #print "......To be deleted: " + str(len(cols_to_be_deleted)) + " cols"
    print "......Index Cols " + str(len(index_cols)) + " cols"
    print "......Pre Cols " + str(len(pre_cols)) + " cols"
    print "......Post Cols " + str(len(post_cols)) + " cols"


####
#### STEP 3: add count columns, scale age by 5
####
#raw_input("Press Enter to continue...")
print utils.time() + "add count cols"
IDX_COUNT = matrix[:,index_ATC_cols.values()].sum(axis = 1)
matrix = hstack([matrix,IDX_COUNT],format ="csr")
headers.append("IDX_COUNT_ATC")
index_cols["COUNT_ATC"]=len(headers)-1

IDX_COUNT = matrix[:,index_ICD_cols.values()].sum(axis = 1)
matrix = hstack([matrix,IDX_COUNT],format ="csr")
headers.append("IDX_COUNT_ICD")
index_cols["COUNT_ICD"]=len(headers)-1

####
#### STEP 4: pre-calc prevalences and related mean ages
####
print utils.time() + "Start calculation Prevalences / Mean Ages"
prevalences = matrix[:,index_cols.values()].mean(axis = 0).A1 #requires index to be 2010 (1 year)
prevalences_count = matrix[:,index_cols.values()].sum(axis = 0).A1
prevalences_ages_sum = matrix[:,index_cols.values()].multiply(matrix[:,index_cols[settings.age]]).sum(axis = 0).A1
prevalences_ages = prevalences_ages_sum / prevalences_count

print "... prevalences: ", prevalences.shape
print "... Ages: ", prevalences_ages.shape

IDX_COUNT = None
prevalences_count = None
prevalences_ages_sum = None

####
#### STEP 5: Feature scaling (only age: div by age_scaler = 5)
####
matrix [:,index_cols[settings.age]] = matrix [:,index_cols[settings.age]]/settings.age_scaler #scales age to allow better fitting

####
#### STEP 6: loop through post_cols
####
#loop through post_cols
start = False
for node in post_cols.keys():
    if node==settings.start_icd or settings.start_icd=="":
        start = True
    if start:
        print utils.time() + 'Work on ' + node
        ####
        #### STEP 7: calculate node statistics: get possible and real incidents, calc prevalence, incidence, mean_age
        ####
        try:
            if node=="DEATH":
                possible_incidents = matrix
            else:
                possible_incidents = matrix[functions.zero_rows(matrix[:,pre_cols[node]]),:]  #rows,cols = matrix.nonzero()

            print "...Possible incidents: ", possible_incidents.get_shape()
            real_incidents = possible_incidents[possible_incidents[:,post_cols[node]].nonzero()[0],:]
            print "...Real incidents: ", real_incidents.shape

            print "...calc node prevalence + incidence"
            if node==settings.death:
                prevalence = 0
                mean_age_prevalents = 0
            else:
                prevalence = prevalences[index_cols.keys().index(node)]
                mean_age_prevalents = prevalences_ages[index_cols.keys().index(node)]

            if settings.debug:
                print "Prevalence: ", prevalence
                print "Mean age: ", mean_age_prevalents

            incidence4y = float(real_incidents.shape[0])/possible_incidents.shape[0] #possible_incidents[:,post_cols[node]].mean()
            incidence1y = incidence4y/settings.years
            mean_age_incidents = int(real_incidents[:,index_cols[settings.age]].mean()*settings.age_scaler)+2 #add 2 as Age comes from 2010, incedence is 2011 - 2014 #scale back age

            ####
            #### STEP 8:pre-calc means and sums for all possible coeffs on real incidence
            ####
            print "...Pre-calc means and sums for all possible coeffs on real incidence"
            coeff_means = real_incidents[:,index_cols.values()].mean(axis = 0).A1
            coeff_count = real_incidents[:,index_cols.values()].sum(axis = 0).A1
            # pre-calc mean ages for all possible coeffs on real incidence, using matrix apply
            coeff_ages_sum = real_incidents[:,index_cols.values()].multiply(real_incidents[:,index_cols[settings.age]]).sum(axis = 0).A1
            coeff_ages = coeff_ages_sum / coeff_count
            coeff_ages = coeff_ages * settings.age_scaler + 2 #add 2 as Age comes from 2010, incedence is 2011 - 2014 #scale back age

            print "... Means: ", coeff_means.shape
            print "... Sums: ", coeff_count.shape
            print "... Ages: ", coeff_ages.shape
            coeff_ages_sum = None


            ####
            #### STEP 9: model with linear boosting, for variable selection and coeff/OR calculation
            ####

            #split training and test
            X_train, X_test, y_train, y_test = train_test_split(possible_incidents[:,index_cols.values()], possible_incidents[:,post_cols[node]].A[:,0], train_size=.7, stratify=possible_incidents[:,post_cols[node]].A[:,0], random_state=0)
            dtrain = xgb.DMatrix(X_train, label=y_train)
            dtest = xgb.DMatrix(X_test, label=y_test)

            print "... X_train: ", X_train.shape
            print "... y_train: ", y_train.shape
            print "... X_test: ", X_test.shape

            #model: linear boosting
            silent_mode=1
            if settings.debug:
                silent_mode=0
            top1percent_train = max(1,int(X_train.shape[0] * 0.01))
            top1percent_eval = max(1,int(X_test.shape[0] * 0.01))
            params={'silent': silent_mode,
                'nthread': settings.nthread,
                'eval_metric':['error','map','map@'+str(top1percent_train),'map@'+str(top1percent_eval),'auc'], #error:#(wrong cases)/#(all cases). #map=Mean average precision  #threshhold is always 0.5
                'objective': 'binary:logistic',
                'booster': 'gblinear',
                'lambda': 0, #L2 regularization  = none
                'alpha': 5} #L1 regularization (LASSO)

            print utils.time() + 'Erstelle Modell 1 ' + node
            quality = {}
            booster = xgb.train( params, dtrain, num_boost_round=settings.boosting_iterations, evals=[(dtrain,'train'),(dtest,'eval')], early_stopping_rounds=10, evals_result =quality)

            ####
            #### STEP 10: find all selected coeffs, and prepare edge information
            ####
            edges = []
            coeffs_idx = []
            coeffs_raw = booster.get_dump(with_stats=False)[0]
            counter = 0
            all_lines = iter(coeffs_raw.splitlines())
            for line in all_lines:
                if line == 'bias:':
                    coeff = float(next(all_lines))
                    edges.append(["Intercept",node,math.exp(coeff),coeff,0,0,0,"na",0])
                elif line == 'weight:':
                    counter = 0
                else:
                    coeff= float(line)
                    if coeff<>0:
                        edges.append([index_cols.keys()[counter],node,math.exp(coeff),coeff,"p_values",coeff_count[counter],coeff_means[counter],"na",coeff_ages[counter]])
                        coeffs_idx.append(counter)
                        #ToDO calc proportion source get target (now "nA")
                    counter += 1

            ####
            #### STEP 11: find p-values (no additional GLM but Computing the ANOVA F-value, i.e.Analysis of Variance )
            ####
            F, pvals = f_classif(X_train[:,coeffs_idx],y_train)
            for j, pval in enumerate(pvals):
                edges[j+1][4]=pval #edges start from 1 because intercept; position 4 is p-value

            ####
            #### STEP 12: OUTPUT models, nodes, edges
            ####
            train_auc = str(quality['train']['auc'][-1])
            train_error = str(quality['train']['error'][-1])
            train_precision = str(quality['train'][''][-1])
            train_precision_top1 = str(quality['train']['map@'+str(top1percent_train)][-1])
            eval_auc = str(quality['eval']['auc'][-1])
            eval_error = str(quality['eval']['error'][-1])
            eval_precision = str(quality['eval'][''][-1])
            eval_precision_top1 = str(quality['eval']['map@'+str(top1percent_eval)][-1])

            with open(settings.out_model, 'a') as f:
                f.write(';'.join([node,train_auc,train_error,train_precision,str(top1percent_train),train_precision_top1,eval_auc,eval_error,eval_precision,str(top1percent_eval),eval_precision_top1]) + '\n')

            with open(settings.out_nodes, 'a') as f:
                f.write(';'.join([node,str(prevalence),str(incidence1y),str(mean_age_incidents),str(mean_age_prevalents)]) + '\n')

            with open(settings.out_edges, 'a') as f:
                writer = csv.writer(f, delimiter=';', quoting=csv.QUOTE_NONE, lineterminator='\n')
                writer.writerows(edges)

            if settings.debug:
                booster.dump_model(settings.out_directory_client+'/'+node+'dump.raw.txt',with_stats=True)
                with open(settings.out_directory_client+'/'+node+'pvals.txt', 'w') as f:
                    for pval in pvals:
                        f.write("%s\n" % pval)
        except Exception, e:
            print '#################### Fehler bei Modell ' +  node
            my_old_stdout = sys.stdout
            sys.stdout = open(settings.out_directory_client + '/'+node+'.failed.txt', 'w')
            print 'Model-Name: ' + node
            print "Couldn't do it: %s" % e
            sys.stdout.close()
            sys.stdout = old_stdout


sys.stdout.close()
#set back
sys.stdout = old_stdout

#os.system('python ddi.py.py')