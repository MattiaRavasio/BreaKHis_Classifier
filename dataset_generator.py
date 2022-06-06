import shutil
import os
import pandas as pd
import numpy as np

def dataset_generator(df):

    for i in range(df.count()[0]):
        file = df.filename[i]
        
        file_source = os.path.join('archive/BreaKHis_v1',file)
        file_destination = os.path.join('data', os.path.join(df.grp[i], df.class[i]))
   
        #get_files = os.listdir(file_source)
        shutil.copy(file_source, file_destination)
    
    print('folder succesfully generated')