from pathlib import Path
import json
import os

import pandas as pd

import xlwt
from xlwt import Workbook, XFStyle, Font, easyxf

import arial10  # Ensure you have this library or remove if unnecessary

from config import *
from db_tools import *
from phi_scan import phi_scan


def main():
    global tables_to_scan

    print("Step 1: Confirm tables to scan")

    all_tables = get_tables(selected_db, text_file_location)

    print("All tables")

    if len(tables_to_scan) == 0:
        tables_to_scan = all_tables
    else:
        if "TXT" in selected_db:
            tables_to_scan = [table for table in tables_to_scan if table.endswith(".csv")]
        else:
            tables_to_scan = [table for table in tables_to_scan if table in all_tables]

    print("Step 1: Tables to scan")
    print(tables_to_scan)

    print("Step 2: Data Profiling")

    source_mapping = {}
    source_profile = {}

    for profile_table in tables_to_scan:
        print(profile_table)
        table_mapping = {}
        profile_table_result = get_table_profile(
            selected_db, profile_table, data_profile_sample_size, text_file_location
        )
        source_profile[profile_table] = profile_table_result['data_stats']
        for column_stats in profile_table_result['data_stats']:
            if column_stats['categorical']:
                table_mapping[column_stats['column_name']] = [
                    x for x in column_stats['statistics']['categorical_count']
                ]
            else:
                table_mapping[column_stats['column_name']] = []
        source_mapping[profile_table] = table_mapping

    #print(json.dumps(source_mapping, indent=4))
    #print(json.dumps(source_profile, indent=4))

    data_profile_result = {}    
    for source_table in source_profile:
        for tb_item in source_profile[source_table]:
            data_profile_item_result = {}  
            data_profile_col_name = source_table+'.'+tb_item['column_name']
            data_profile_item_result['table'] = source_table
            data_profile_item_result['column'] = tb_item['column_name']
            data_profile_item_result['unique_count'] = tb_item['statistics']['unique_count']
            data_profile_item_result['unique_ratio'] = tb_item['statistics']['unique_ratio']
            if tb_item['categorical'] :
                categorical_samples = []
                x = 0
                for y in tb_item['statistics']['categorical_count'] :
                        categorical_samples.append( str(y)+":"+str(tb_item['statistics']['categorical_count'][y]))
                        x += 1
                        if x >= 5 :
                            break
                categorical_samples = ' ; '.join(categorical_samples)                         
                if len(tb_item['statistics']['categorical_count']) > 5 :
                        categorical_samples += '...'
                data_profile_item_result['Value Counts'] = categorical_samples
            else:     
                    data_profile_item_result['Value Counts'] = "Not available for non categorical items"
            data_profile_result[data_profile_col_name] = data_profile_item_result

    print(json.dumps(data_profile_result, indent=4))

    print("Step 2: Data Profiling - Done")

    print( "step 3 : PHI SCAN")

    phi_scan_result_json={}

    for phi_scan_table in tables_to_scan:
        original_data_path = './data_profile/table_{}_sample.csv'.format(phi_scan_table)
        json_file_path = './data_profile/table_{}_profile.json'.format(phi_scan_table)
        model_path = PHI_SCAN_MODEL
        output_path = './data_profile/table_{}_phi.csv'.format(phi_scan_table)
        print(phi_scan_table)
        phi_scan(original_data_path,json_file_path,model_path,output_path)
        phi_scan_result = pd.read_csv(output_path,index_col=0)          
        #phi_scan_result = phi_scan_result[phi_scan_result['ML prediction result 0/1']==1]

        for index,row in  phi_scan_result.iterrows():
            phi_scan_result_json['{}.{}'.format(phi_scan_table,index)] = {'method':'0','offset':'','predict_probability':row['ML prediction result'],'predict_result':row['ML prediction result 0/1']}


    print(json.dumps(phi_scan_result_json,indent = 4))  

    print( "step 3 : PHI SCAN - Done")

    print( "step 4 : Save PHI SCAN Result")

    wbk = Workbook()
    sheet = wbk.add_sheet("sheet", cell_overwrite_ok=True)

    # Style for bold headers
    style = XFStyle()
    font = Font()
    font.bold = True
    style.font = font

    # Style for highlighted cells
    st = easyxf('pattern: pattern solid;')
    st.pattern.pattern_fore_colour = 43  

    # Define the output columns
    output_cols = ['Table','Column','Predicted PHI Probability','Predicted Result','Unique Values','Unique Ratio', 'Value Counts']

    # Write the header
    for c in range(len(output_cols)):
        sheet.write(0, c, output_cols[c], style=style)

    results = []

    # Assuming phi_scan_result_json and data_profile_result are defined and populated
    for table_column in phi_scan_result_json:
        item_current_table = data_profile_result[table_column]['table']
        item_current_table_col = data_profile_result[table_column]['column']                
        item_current_probability = str(phi_scan_result_json[table_column]['predict_probability'])
        item_current_result = str(phi_scan_result_json[table_column]['predict_result'])
        item_current_u_counts = str(data_profile_result[table_column]['unique_count'])
        item_current_u_ratio = str(data_profile_result[table_column]['unique_ratio'])
        item_current_v_counts = data_profile_result[table_column]['Value Counts']   
        results.append([item_current_table, item_current_table_col, item_current_probability, item_current_result, item_current_u_counts, item_current_u_ratio, item_current_v_counts])

    # Write the results to the sheet
    for r in range(len(results)):
        for c in range(len(output_cols)):
            text = results[r][c]
            if len(text) > 200:
                text = "Text too long to save to a file"
            if c == 3 and text == '1.0':
                sheet.write(r+1, c, text, st)
            else:
                sheet.write(r+1, c, text)

    wbk.save(result_file_path)
    print("step 4 : Save PHI SCAN Result - Done",result_file_path )
if __name__ == "__main__":
    main()


 