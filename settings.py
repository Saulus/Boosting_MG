__author__ = 'Paul Hellwig'


class Settings:
    #Java Model building
    in_directory_server = ''
    in_directory_client = ''
    out_directory_server = ''
    out_directory_client = ''
    mat_file = ''
    colnames_file = ''

    #H2O instance
    ip = 'localhost'
    port = 54321

    #Flags
    debug = False
    clean_output = False
    years=4
    do_upload = True
    start_icd = "POST_ICD_U50" #ICD_R04

    # Clean Columns
    del_cols = [] #'time_end','reference_group'
    index_prefix = 'IDX_'
    pre_prefix = 'PRE_'
    post_prefix = 'POST_'
    age = 'IDX_AGE'
    gender = 'IDX_GENDER'
    #death = 'POST_DEATH'
    hosp = 'IDX_HOSP'

    #files
    f_allmodels = 'all_models.csv'
    f_edges = 'edges.csv'
    f_nodes = 'nodes.csv'
    f_log = 'mg.log'

    out_edges = f_edges
    out_model = f_allmodels
    out_nodes = f_nodes
    out_log = f_log

    def refresh(self):
        self.out_model =  self.out_directory_client+'/'+ self.f_allmodels
        self.out_edges =  self.out_directory_client+'/'+ self.f_edges
        self.out_nodes =  self.out_directory_client+'/'+ self.f_nodes
        self.out_log =  self.out_directory_client+'/'+ self.f_log


def development():
    settings = Settings()

    settings.in_directory_server = 'C:/Users/hellwigp/Documents/4 Technology/00 Development/zufall/medgraph'
    settings.in_directory_client = 'C:/Users/hellwigp/Documents/4 Technology/00 Development/zufall/medgraph'
    settings.out_directory_server = 'C:/Users/hellwigp/Documents/4 Technology/00 Development/zufall/medgraph'
    settings.out_directory_client = 'C:/Users/hellwigp/Documents/4 Technology/00 Development/zufall/medgraph'
    settings.mat_file = 'mgprofil_svmlight_100k.svm'
    settings.colnames_file = 'mgprofil_svmlight_head.svm'


    #H2O instance
    settings.ip = 'localhost'
    settings.port = 54321

    #Flags
    settings.debug = True
    settings.clean_output = False

    settings.refresh()
    return settings

def production():
    settings = Settings()


    settings.in_directory_server = '/sasfs/sas_temp/dev_temp/saswork/Paul/MG'
    settings.in_directory_client = 'E:/MGResults'
    settings.out_directory_server = '/sasfs/sas_temp/dev_temp/saswork/Paul/MG/out'
    settings.out_directory_client = 'E:/MGResults/out'
    settings.mat_file = 'mgprofil_svmlight.svm'
    settings.colnames_file = 'mgprofil_svmlight_head.svm'

    #H2O instance
    settings.ip = '10.55.143.205'
    settings.port = 54321

    #Flags
    settings.debug = False
    settings.clean_output = False

    settings.refresh()
    return settings