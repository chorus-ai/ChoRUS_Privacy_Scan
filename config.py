# setup coonfiguration data here
import os

available_dbs = {'PSQL_MIMIC': ["userid:password@192.168.0.199:5432/mimic","mimiciii"],
                'LOCAL_TEXT_FILES': 'LOCAL_TEXT_FILES'
        }
 
data_profile_sample_size = 500  

PHI_SCAN_MODEL = './phi_scan/XGBClassifier(V220230227).json'


target_models = {
}

dest_db = ""

target_dbs ={
}


