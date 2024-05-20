import json
import sys
import os

def load_config():
    # Path to the directory where the executable or script resides
    base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(base_path, 'config.json')
    
    try:
        with open(config_path, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        print("Configuration file not found. Ensure 'config.json' is in the correct directory.")
        return {}
    except json.JSONDecodeError:
        print("Error decoding 'config.json'. Ensure the file is formatted correctly.")
        return {}

# Load the configuration at startup
config = load_config()

# Assign configuration data to variables
available_dbs = config.get('available_dbs', {})
data_profile_sample_size = config.get('data_profile_sample_size', 1000)  # default to 1000 if not specified
PHI_SCAN_MODEL = config.get('PHI_SCAN_MODEL', './phi_scan/default_model.json')  # provide a default model path if not specified


text_file_location = config.get('text_file_location', './data_folder')
selected_db = config.get('selected_db', 'LOCAL_TEXT_FILES')
tables_to_scan  = config.get('tables_to_scan', [])

output_folder = config.get('output_folder', '.')
result_file = config.get('result_file', 'phi_scan_result.xls')
result_file_path = os.path.join(output_folder, result_file)
log_file_path = os.path.join(output_folder, 'warning.log')

target_models = config.get('target_models', {})
dest_db = config.get('dest_db', '')  # default to an empty string if not specified
target_dbs = config.get('target_dbs', {})
scan_tables= config.get('scan_tables', [])




 
