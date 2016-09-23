__author__ = 'Paul Hellwig'

import datetime
import subprocess
import random
import gc
import h2o



#helper functions
def run_command(args,executable=None):
    if executable == None:
        p = subprocess.check_call(args)
    else:
        p = subprocess.check_call(args,executable=executable)
    if p==0:
        return 0
    else:
        return p.returncode

def time():
    return datetime.datetime.strftime(datetime.datetime.now(), '%H:%M:%S') + ' '


   
