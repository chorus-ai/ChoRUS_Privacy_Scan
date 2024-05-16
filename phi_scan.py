#!/usr/bin/env python
# coding: utf-8
# %%


import pandas as pd
import numpy as np
from tqdm import tqdm
import re

# import matplotlib.pyplot as plt
import json
from xgboost import XGBClassifier
import datetime
import argparse


# %%


# from AutoDeidentifyNet import *

from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix,accuracy_score, roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.metrics import roc_auc_score, confusion_matrix,accuracy_score, roc_curve, auc, precision_recall_curve

from pandas.api.types import is_numeric_dtype


# %%


from sklearn.metrics import roc_curve, confusion_matrix , auc, precision_recall_curve, average_precision_score
from sklearn import metrics

def print_metrics(y_true, y_pred): # threshold must be pre-determined by the training program, otherwise, this test data may contain all y=1 or all y=0, and the threshold will be different from the training program
    
    false_positive_rate, recall, thresholds = roc_curve(y_true, y_pred)
    roc_auc = auc(false_positive_rate, recall)
    auprc = average_precision_score(y_true, y_pred)
    print('AUC: ',roc_auc, "AUPRC: ", auprc)

    print("threshold: 0.994889")
    y_pred_01 = y_pred.apply(lambda x: int(x >= 0.994889))
    

    return y_pred_01


# %%
# ### flatten dict 
# https://www.freecodecamp.org/news/how-to-flatten-a-dictionary-in-python-in-4-different-ways/
def create_training_data(json_data):
    from collections.abc import MutableMapping

    def _flatten_dict_gen(d, parent_key, sep):
        for k, v in d.items():
            new_key = parent_key + sep + k if parent_key else k
            if isinstance(v, MutableMapping):
                yield from flatten_dict(v, new_key, sep=sep).items()
            else:
                yield new_key, v


    def flatten_dict(d: MutableMapping, parent_key: str = '', sep: str = '.'):
        return dict(_flatten_dict_gen(d, parent_key, sep))

    feature_list_1 = ['column_name', 'data_type', 'categorical', 'order'] 
    feature_list_2 = set()

    train_data_all = []

    for i in tqdm(range(len(json_data['data_stats']))):
        train_data = flatten_dict(json_data['data_stats'][i])
        train_data_all.append(train_data)


    # remove some json features
    feature_list = set([j for i in train_data_all for j in i.keys()])
    feature_list_2 = []
    for j in feature_list:
        if 'categorical_count' in j: continue
        if 'null_types_index' in j: continue
        if  'null_types' in j: continue
        if 'mode' in j : continue
        if 'samples' in j: continue
        if 'categories' in j : continue
        if 'bin_edges' in j : continue 
        if 'bin_counts' in j: continue
        if 'column_name' in j: continue
        if 'format' in j: continue
        if 'order' in j: continue
        if 'categorical' in j: continue
            
        feature_list_2.append(j)

    # build training dataframe
    df_train_data_all = pd.DataFrame(columns = feature_list_2)
    for data in train_data_all:
        df_data = pd.DataFrame(data, columns = feature_list_2, index=[data['column_name']])
        df_train_data_all = df_train_data_all._append(df_data)

    # create 0/1 mask data
    df_train_data_all_mask = df_train_data_all.notnull().astype('int')
    df_train_data_all_mask.columns = [i + str('_01') for i in df_train_data_all_mask.columns]

    df_train_data_all = df_train_data_all.join(df_train_data_all_mask)
    return df_train_data_all


with open('HIPPA.txt', 'r') as f:
    HIPPA = f.readlines()
HIPPA = [line.strip().replace('\t', '') for line in HIPPA]

# fix_column_name =  ['statistics.quantiles.0', 'statistics.median_abs_deviation', 'statistics.mean', 'statistics.min', 'statistics.unalikeability', 'statistics.histogram', 'statistics.times.sum', 'statistics.median', 'statistics.num_negatives', 'statistics.times.min', 'statistics.data_type_representation.float', 'statistics.max', 'statistics.sum', 'statistics.unique_ratio', 'statistics.null_count', 'statistics.num_zeros', 'statistics.data_type_representation.datetime', 'statistics.data_type_representation.int', 'data_type', 'statistics.variance', 'statistics.times.datetime', 'statistics.times.skewness', 'statistics.sample_size', 'statistics.quantiles.1', 'statistics.times.kurtosis', 'statistics.quantiles.2', 'statistics.kurtosis', 'statistics.times.histogram_and_quantiles', 'statistics.stddev', 'statistics.gini_impurity', 'statistics.skewness', 'statistics.times.num_negatives', 'statistics.unique_count', 'statistics.times.num_zeros', 'statistics.times.max', 'statistics.times.variance', 'statistics.quantiles.0_01', 'statistics.median_abs_deviation_01', 'statistics.mean_01', 'statistics.min_01', 'statistics.unalikeability_01', 'statistics.histogram_01', 'statistics.times.sum_01', 'statistics.median_01', 'statistics.num_negatives_01', 'statistics.times.min_01', 'statistics.data_type_representation.float_01', 'statistics.max_01', 'statistics.sum_01', 'statistics.unique_ratio_01', 'statistics.null_count_01', 'statistics.num_zeros_01', 'statistics.data_type_representation.datetime_01', 'statistics.data_type_representation.int_01', 'data_type_01', 'statistics.variance_01', 'statistics.times.datetime_01', 'statistics.times.skewness_01', 'statistics.sample_size_01', 'statistics.quantiles.1_01', 'statistics.times.kurtosis_01', 'statistics.quantiles.2_01', 'statistics.kurtosis_01', 'statistics.times.histogram_and_quantiles_01', 'statistics.stddev_01', 'statistics.gini_impurity_01', 'statistics.skewness_01', 'statistics.times.num_negatives_01', 'statistics.unique_count_01', 'statistics.times.num_zeros_01', 'statistics.times.max_01', 'statistics.times.variance_01', 'statistics.precision.margin_of_error', 'statistics.precision.confidence_level', 'statistics.precision.var', 'statistics.precision.mean', 'statistics.precision.max', 'statistics.times.precision', 'statistics.precision.sample_size', 'statistics.precision.min', 'statistics.precision.std', 'statistics.precision.margin_of_error_01', 'statistics.precision.confidence_level_01', 'statistics.precision.var_01', 'statistics.precision.mean_01', 'statistics.precision.max_01', 'statistics.times.precision_01', 'statistics.precision.sample_size_01', 'statistics.precision.min_01', 'statistics.precision.std_01'] ['statistics.quantiles.2', 'statistics.histogram', 'statistics.data_type_representation.float', 'statistics.times.min', 'statistics.times.sum', 'statistics.sample_size', 'statistics.median_abs_deviation', 'statistics.null_count', 'statistics.sum', 'statistics.median', 'statistics.data_type_representation.int', 'statistics.num_negatives', 'data_type', 'statistics.max', 'statistics.times.kurtosis', 'statistics.times.skewness', 'statistics.min', 'statistics.gini_impurity', 'statistics.times.num_zeros', 'statistics.mean', 'statistics.unalikeability', 'statistics.quantiles.1', 'statistics.num_zeros', 'statistics.quantiles.0', 'statistics.data_type_representation.datetime', 'statistics.unique_ratio', 'statistics.skewness', 'statistics.kurtosis', 'statistics.times.datetime', 'statistics.times.variance', 'statistics.times.num_negatives', 'statistics.times.max', 'statistics.variance', 'statistics.unique_count', 'statistics.stddev', 'statistics.times.histogram_and_quantiles', 'statistics.quantiles.2_01', 'statistics.histogram_01', 'statistics.data_type_representation.float_01', 'statistics.times.min_01', 'statistics.times.sum_01', 'statistics.sample_size_01', 'statistics.median_abs_deviation_01', 'statistics.null_count_01', 'statistics.sum_01', 'statistics.median_01', 'statistics.data_type_representation.int_01', 'statistics.num_negatives_01', 'data_type_01', 'statistics.max_01', 'statistics.times.kurtosis_01', 'statistics.times.skewness_01', 'statistics.min_01', 'statistics.gini_impurity_01', 'statistics.times.num_zeros_01', 'statistics.mean_01', 'statistics.unalikeability_01', 'statistics.quantiles.1_01', 'statistics.num_zeros_01', 'statistics.quantiles.0_01', 'statistics.data_type_representation.datetime_01', 'statistics.unique_ratio_01', 'statistics.skewness_01', 'statistics.kurtosis_01', 'statistics.times.datetime_01', 'statistics.times.variance_01', 'statistics.times.num_negatives_01', 'statistics.times.max_01', 'statistics.variance_01', 'statistics.unique_count_01', 'statistics.stddev_01', 'statistics.times.histogram_and_quantiles_01', 'statistics.precision.sample_size', 'statistics.precision.min', 'statistics.precision.margin_of_error', 'statistics.times.precision', 'statistics.precision.std', 'statistics.precision.max', 'statistics.precision.mean', 'statistics.precision.var', 'statistics.precision.confidence_level', 'statistics.precision.sample_size_01', 'statistics.precision.min_01', 'statistics.precision.margin_of_error_01', 'statistics.times.precision_01', 'statistics.precision.std_01', 'statistics.precision.max_01', 'statistics.precision.mean_01', 'statistics.precision.var_01', 'statistics.precision.confidence_level_01']


def main(model, df, df_json):
    X = create_training_data(df_json)
    
    ## Columns name needs to be fixed
    # add these fix_column_name

    fix_column_name = model.get_booster().feature_names

    X = pd.concat([pd.DataFrame(columns = fix_column_name), X]) 
    
    # X_join could possibily contain unseen columns -> delete
    X = X[fix_column_name]
    assert X.shape[1] == 90



    # create HIPPA label
    X['HIPPA'] = 0
    X[X.index.isin(HIPPA)] = 1
    #
    # change object to float
    for i in X.columns:
        X[i] = pd.to_numeric(X[i],errors='coerce')
    
    
    # sampe 5000 to form text
    data_1_5000 = df.loc[:5000, :]

    data_1_5000 = data_1_5000.replace('Unknown', np.nan) # unknown -> nan
    data_1_5000 = data_1_5000.replace('Other', np.nan)  # other -> nan

    print(X.shape)

    feature_names = model.get_booster().feature_names
    # print(set(fix_column_name) - set(feature_names))
    # print(set(feature_names) - set(fix_column_name) )
    # input()

    df_pred_result = pd.DataFrame({'ML prediction result': model.predict_proba(X.drop(columns=['HIPPA']))[:, 1]}).set_index([X.index])
#     df_pred_result['REgular expression result'] = pd.DataFrame.from_dict(y_pred_2, orient='index').set_index([X.index])
    df_pred_result = df_pred_result.join(X[['HIPPA']])

    predict_01 = print_metrics(df_pred_result['HIPPA'], df_pred_result['ML prediction result'])

    df_pred_result['ML prediction result 0/1'] = predict_01
    
    df_pred_result = df_pred_result[['HIPPA', 'ML prediction result', 'ML prediction result 0/1']]

    """ ========================= regular expression========================"""
    import re
    def re_find_count(text):

        # phone, fax
        # text = "738-345-3453, 343-234-2342, 3434543543"
        pattern_number = r'\b(?:\d[-.()]*?){10}\b'
        result_text = re.findall(pattern_number, text)
        number_text = len(result_text)
        # print(result1)

        # text = "(123)456-7890, (345)678-9012"
        pattern_number = r'\(\d{3}\)\d{3}-\d{4}'
        result_number = re.findall(pattern_number, text)
        number_number = len(result_number)
        # print(result1)
        # print(number1)

        # ID
        # text = "43573, 23423,34234, 1234567, 12345678, 123456789,"
        pattern_7digits = r'\b\d{7,}\b'
        result_ID = re.findall(pattern_7digits, text)
        number_ID = len(result_ID)
        # print(number3)

        # zip
        # text = '87878-3049, 34948'
        pattern_postal = r'\b(\d{5}(-\d{4})?)\b'
        result_zip = re.findall(pattern_postal, text)
        number_zip = len(result_zip)
        # print(result3)

        # date
        # text = "2021-01-07 18:45:00 , 2021-01-07 18:45:00 , 2021-01-07 18:45:00 , 2021-01-07 18:45:00"
        pattern_date = r'\d{4}-\d{2}-\d{2}(?: \d{2}:\d{2}:\d{2})?'
        result_date= re.findall(pattern_date, text)
        number_date = len(result_date)
        # print(result4)

        # email
        # text = "john.doe@example.com, jane_doe123@gmail.com, info@company.co.uk"
        pattern_email = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        result_emails = re.findall(pattern_email, text)
        number_emails = len(result_emails)
        # print(result_emails)
        # print(number_emails)

        # utl
        # text = "Visit us at http://www.example.com, check out https://example.org, or go to www.example.net"
        pattern_url = r'\b(?:https?://)?(?:www\.)?[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}(?:\.[A-Za-z]{2,})?\b'
        result_urls = re.findall(pattern_url, text)
        number_urls = len(result_urls)
        # print(result_urls)
        # print(number_urls)

        # IP
        # text = "Connect to 192.168.1.1, ping 10.0.0.1, or visit http://[2001:db8::1] for IPv6."
        pattern_ip = r'\b(?:\d{1,3}\.){3}\d{1,3}\b|\b(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}\b'
        result_ips = re.findall(pattern_ip, text)
        number_ips = len(result_ips)
        # print(result_ips)
        # print(number_ips)

        
        # text = 'Hello my name is Chandler Muriel Bing. I have a friend who is named Pieter van den Woude and he has another friend, A. A. Milne. Gandalf the Gray joins us. Together, we make up the Friends Cast and Crew.'
        pattern_name = r'[A-Z]([a-z]+|\.)(?:\s+[A-Z]([a-z]+|\.))*(?:\s+[a-z][a-z\-]+){0,2}\s+[A-Z]([a-z]+|\.)'
        result_name = re.findall(pattern_name, text)
        number_name = len(result_name)
        # print(result_name)
        # print(number_name)


        # print(result1, result2, result3, result4)
        return np.max([number_text, number_number, number_ID ,number_zip, number_date ,number_emails, number_urls, number_ips, number_name])
        # return number4


    # sampe 5000 to form text
    y_pred_RE = {}
    for i, data in enumerate([df]):
        print('Processing RE on ', i)
        sample = np.max([5000, data.shape[0]])
        data_5000 = data.iloc[:sample, :]

        data_5000 = data_5000.replace('Unknown', np.nan)  # unknown -> nan
        data_5000 = data_5000.replace('Other', np.nan)  # other -> nan

        # nan to random choice
        import random
        for c in data_5000.columns:
            l = list(set(data_5000[c][data_5000[c].notna()].tolist()))
            if not l:
                l = [0]
            data_5000[c].fillna(random.choice(l), inplace=True)
        print('this is the line')
        for c in tqdm(data_5000.columns):
            print('columns is: ', c)
            text = c + ' , '
            for r in data_5000[c].tolist():
                text += str(r) + ' , '
            print('test is:', text[:100])
            count = re_find_count(text)
            print('count is: ', count)
            p = min(count / sample, 1)
            y_pred_RE[c] = p

    print('RE result: ', y_pred_RE)
    for k,v in y_pred_RE.items():
        if v > 0:
            print(' k is:', k)
            df_pred_result.loc[k, 'HIPPA'] = 1
            df_pred_result.loc[k, 'ML prediction result'] = 1
            df_pred_result.loc[k, 'ML prediction result 0/1'] = 1
    """
    detect features names
    """
    strings_to_flag = ['date', 'time', 'name', 'id', 'zip', 'address', 'phone', 'fax', 'email', 'ssn', 'mrn', 'account']

    flagged_columns = {}

    for col in data_5000.columns:
        if any(string in col for string in strings_to_flag):
            flagged_columns[col] = 1
    print('flagged_columns is:', flagged_columns)
    for k,v in flagged_columns.items():
        if v > 0:
            df_pred_result.loc[k, 'HIPPA'] = 1
            df_pred_result.loc[k, 'ML prediction result'] = 1
            df_pred_result.loc[k, 'ML prediction result 0/1'] = 1

    """
    pandas to datetime
    """

    successful_columns = {}
    for c in df.columns:
        # Skip columns that contain numeric values
        if np.issubdtype(df[c].dtype, np.number):
            print(c, 'skipped (numeric)')
            continue

        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("error")
                a = pd.to_datetime(df[c])
            successful_columns[c] = 1
            print(c, 'success')
        except pd.errors.ParserWarning as e:
            print(f"Warning for column '{c}': {str(e)}")
        except:
            print(c, 'fail')

    print("Columns successfully converted without any warnings:")
    print(successful_columns)

    for k,v in successful_columns.items():
        if v > 0:
            df_pred_result.loc[k, 'HIPPA'] = 1
            df_pred_result.loc[k, 'ML prediction result'] = 1
            df_pred_result.loc[k, 'ML prediction result 0/1'] = 1

    return df_pred_result




# %%

def phi_scan(original_data_path,json_file_path,model_path,output_path):
    
    # load json data
    with open(json_file_path) as f:
        json_data = json.load(f)
    print('Finish loading data 1 json')


    df_data = pd.read_csv(original_data_path)
    print('Finish loading data 1')

    
    # load model
    model_xgb = XGBClassifier()
    model_xgb.load_model(model_path)
    
    csv_result = main(model_xgb, df_data, json_data)
    
    # csv_result.to_csv('Prediction_result_' + str(datetime.datetime.now()) +'.csv')
    
    # csv_result.to_csv('{}_prediction_result.csv'.format(args.json_file_path.split('profile')[0]))
    csv_result.to_csv(output_path)
    

# %%
