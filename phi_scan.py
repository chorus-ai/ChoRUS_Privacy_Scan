#!/usr/bin/env python
# coding: utf-8
# %%


import pandas as pd
import numpy as np
from tqdm import tqdm
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


    '''fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred)

    # calculate the g-mean for each threshold
    gmeans = np.sqrt(tpr * (1-fpr))
    ix = np.argmax(gmeans)
    print('Best Threshold=%f, G-Mean=%.3f' % (thresholds[ix], gmeans[ix]))'''


#     tn, fp, fn, tp = confusion_matrix(y_true, y_pred > thresholds[ix]).ravel()
#     print('TN: ', tn, ", FP: ",fp, ", FN:", fn, ", TP:", tp)
#     print("==> Sensitivity (Recall, TPR): %.3f"%(tp/(tp+fn)))
#     print("==> Specifity: %.3f"%(tn/(tn+fp)))
#     print("==> Positive Predictive Value (PPV) (Precision): %.3f"%(tp / (tp + fp)))
#     print("==> Negative Predictive Value (NPV): %.3f"%(tn / (tn + fn)))
#     print("==> Accuracy: %.3f"%((tp+tn)/(tn+ fp+ fn+tp)))
#     print("==> F1 score: %.3f"%((2*tp)/(2*tp + fp + fn)))

#     ns_probs = [0 for _ in range(len(y_true))]
    
#     # calculate roc curves
#     ns_fpr, ns_tpr, _ = roc_curve(y_true, ns_probs)
#     lr_fpr, lr_tpr, _ = roc_curve(y_true, y_pred)
#     # plot the roc curve for the model
#     plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill, AUC = %0.2f' % 0.5)
#     plt.plot(lr_fpr, lr_tpr, marker='.', label = 'Our model: AUC = %0.2f' % roc_auc)
#     # axis labels
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     # show the legend
#     plt.legend()
#     # show the plot
#     plt.show()
    
#     lr_precision, lr_recall, _ = precision_recall_curve(y_true, y_pred)
#     lr_auc = auc(lr_recall, lr_precision)
#     # summarize scores
# #     print('Logistic: f1=%.3f auc=%.3f' % (lr_f1, lr_auc))
#     # plot the precision-recall curves
#     y_true = np.array(y_true)
#     no_skill = len(y_true[y_true==1]) / len(y_true)
#     plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No skill, AUPRC = %0.2f' % no_skill)
#     plt.plot(lr_recall, lr_precision, marker='.',label = 'Our model: AUPRC = %0.2f' % auprc)
#     # axis labels
#     plt.xlabel('Recall')
#     plt.ylabel('Precision')
#     # show the legend
#     plt.legend(loc = 'upper right')
#     # show the plot
#     plt.show()

    '''y_pred_01 = y_pred.apply(lambda x: int(x >= thresholds[ix]))'''
    
    print("threshold: 0.966910")
    y_pred_01 = y_pred.apply(lambda x: int(x >= 0.966910))
    

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


# %%
fix_column_name = ['statistics.kurtosis', 'statistics.mean', 'statistics.histogram',
       'statistics.times.sum', 'statistics.null_count',
       'statistics.unique_count', 'statistics.data_type_representation.int',
       'statistics.quantiles.0', 'statistics.stddev', 'statistics.sample_size',
       'statistics.median_abs_deviation', 'statistics.times.kurtosis',
       'statistics.data_type_representation.float', 'statistics.unique_ratio',
       'statistics.sum', 'statistics.num_zeros', 'statistics.gini_impurity',
       'statistics.times.min', 'data_type', 'statistics.skewness',
       'statistics.times.variance', 'statistics.times.datetime',
       'statistics.quantiles.2', 'statistics.times.histogram_and_quantiles',
       'statistics.variance', 'statistics.quantiles.1',
       'statistics.times.num_zeros', 'statistics.unalikeability',
       'statistics.times.num_negatives',
       'statistics.data_type_representation.datetime', 'statistics.median',
       'statistics.min', 'statistics.num_negatives', 'statistics.max',
       'statistics.times.skewness', 'statistics.times.max',
       'statistics.kurtosis_01', 'statistics.mean_01',
       'statistics.histogram_01', 'statistics.times.sum_01',
       'statistics.null_count_01', 'statistics.unique_count_01',
       'statistics.data_type_representation.int_01',
       'statistics.quantiles.0_01', 'statistics.stddev_01',
       'statistics.sample_size_01', 'statistics.median_abs_deviation_01',
       'statistics.times.kurtosis_01',
       'statistics.data_type_representation.float_01',
       'statistics.unique_ratio_01', 'statistics.sum_01',
       'statistics.num_zeros_01', 'statistics.gini_impurity_01',
       'statistics.times.min_01', 'data_type_01', 'statistics.skewness_01',
       'statistics.times.variance_01', 'statistics.times.datetime_01',
       'statistics.quantiles.2_01',
       'statistics.times.histogram_and_quantiles_01', 'statistics.variance_01',
       'statistics.quantiles.1_01', 'statistics.times.num_zeros_01',
       'statistics.unalikeability_01', 'statistics.times.num_negatives_01',
       'statistics.data_type_representation.datetime_01',
       'statistics.median_01', 'statistics.min_01',
       'statistics.num_negatives_01', 'statistics.max_01',
       'statistics.times.skewness_01', 'statistics.times.max_01',
       'statistics.precision.var', 'statistics.precision.max',
       'statistics.times.precision', 'statistics.precision.confidence_level',
       'statistics.precision.min', 'statistics.precision.margin_of_error',
       'statistics.precision.sample_size', 'statistics.precision.mean',
       'statistics.precision.std', 'statistics.precision.var_01',
       'statistics.precision.max_01', 'statistics.times.precision_01',
       'statistics.precision.confidence_level_01',
       'statistics.precision.min_01',
       'statistics.precision.margin_of_error_01',
       'statistics.precision.sample_size_01', 'statistics.precision.mean_01',
       'statistics.precision.std_01']

# %%
def main(model, df, df_json):
    X = create_training_data(df_json)
    
    ## Columns name needs to be fixed
    # add these fix_column_name
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

#     # nan to random choice
#     import random 
#     for c in data_1_5000.columns:
#         l = list(set(data_1_5000[c][data_1_5000[c].notna()].tolist()))
#     #     print(l)
#         if not l:
#             l = [0]
#         data_1_5000[c].fillna(random.choice(l), inplace=True)

#     y_pred_2 = {}

#     whitelist_file = 'whitelist.pkl'

#     start_time_all = time.time()
#     with open(whitelist_file, "rb") as fin:
#         whitelist = pickle.load(fin)

#     for c in data_1_5000.columns:
#         print(c)
#         text = c + ' , '
#         for r in data_1_5000[c].tolist()[:100]:
#             text += str(r) + ' , '
#         try:
#             p = filter_task(text, whitelist)
#         except: 
#             print(c, ' is not english')
#             p = 0
#         print("{:.2f}%".format(p*100))
#         y_pred_2[c] = p

    print(X.shape)
    df_pred_result = pd.DataFrame({'ML prediction result': model.predict_proba(X.drop(columns=['HIPPA']))[:, 1]}).set_index([X.index])
#     df_pred_result['REgular expression result'] = pd.DataFrame.from_dict(y_pred_2, orient='index').set_index([X.index])
    df_pred_result = df_pred_result.join(X[['HIPPA']])

    predict_01 = print_metrics(df_pred_result['HIPPA'], df_pred_result['ML prediction result'])

    df_pred_result['ML prediction result 0/1'] = predict_01
    
    df_pred_result = df_pred_result[['HIPPA', 'ML prediction result', 'ML prediction result 0/1']]

    """ ========================= regular expression========================"""
    import re
    def re_find_count(text):
        # text = "738-345-3453, 343-234-2342"
        # pattern_number = r'\b(?:\d[-.()]*?){10}\b'
        # result1 = re.findall(pattern_number, text)
        # number1 = len(result1)
        #
        # # text = "43573, 23423,34234,"
        # pattern_5digits = r'\b\d{5}\b'
        # result2 = re.findall(pattern_5digits, text)
        # number2 = len(result2)
        #
        # # text = '87878-3049, 34948'
        # pattern_postal = r'\b(\d{5}(-\d{4})?)\b'
        # result3 = re.findall(pattern_postal, text)
        # number3 = len(result3)

        #     text = "2021-01-07 18:45:00 , 2021-01-07 18:45:00 , 2021-01-07 18:45:00 , 2021-01-07 18:45:00"
        pattern_date = r'\d{4}-\d{2}-\d{2}(?: \d{2}:\d{2}:\d{2})?'
        result4 = re.findall(pattern_date, text)
        number4 = len(result4)

        # print(result1, result2, result3, result4)
        # return np.max([number1, number2, number3, number4])
        return number4


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
    return df_pred_result



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
